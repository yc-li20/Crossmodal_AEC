import os
import time
import nltk
import random
import numpy as np
from torch import nn
import torch, transformers
from torch.optim import AdamW
import torch.nn.functional as F
from nltk.metrics import edit_distance
from nltk.tokenize import word_tokenize
from torch.nn.parallel import DataParallel
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel


model_name = "roberta-base"
config = RobertaConfig.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
roberta_model = RobertaModel.from_pretrained(model_name)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


"""
Data preparation
"""

data_truth = 'your_groundtruth_path.txt'
data_trans = 'your_asrtranscript_path.txt'

with open(data_truth, 'r', encoding='utf-8') as f1:
    lines_truth = [line.strip().lower() for line in f1.readlines()]
with open(data_trans, 'r', encoding='utf-8') as f2:
    lines_trans = [line.strip().lower() for line in f2.readlines()]

combined = list(zip(lines_truth, lines_trans)) # make sure the truth and trans are in pairs
random.shuffle(combined) # if you need to shuffle your data

train_data = combined[:you_training_data_amount]
val_data = combined[you_training_data_amount:]

train_truth, train_trans = zip(*train_data)
val_truth, val_trans = zip(*val_data)


"""
ASR error correction model -- text only
"""

class Text2TextModel(nn.Module):
    """
        * `encoder`- encoder of the sequence-to-sequence model, e.g. "bert-base-uncased".
        * `decoder`- decoder of the sequence-to-sequence model, usually initialized randomly.
        * `config`- config file of encoder model, should be obtained from hugging face pre-trained checkpoints.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target during training and generation.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size, max_length, sos_id, eos_id):
        super(Text2TextModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(512, 512)))  # max_length cannot exceed 512
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, source_ids, source_mask, target_ids, target_mask): # all inputs' shape: bsz * max_len
        outputs = self.encoder(source_ids, attention_mask=source_mask) # outputs[0]: bsz * max_len * hidden
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous() # max_len * bsz * hidden
        if target_ids is not None:  # training, teacher-forcing
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous() # max_len * bsz * hidden
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask, memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states) # bsz * max_len * vocab
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1 # indices that are not masked, flattened ((max_len - 1) * bsz)
            shift_logits = lm_logits[..., :-1, :].contiguous() # bsz * (max_len -1) * vocab
            shift_labels = target_ids[..., 1:].contiguous() # bsz * (max_len -1)
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss], shift_labels.view(-1)[active_loss])

            return loss, loss * active_loss.sum(), active_loss.sum()

        else:  # autoregressive decoding
        # generate text
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask, memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            return torch.cat(preds, 0)

class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


"""
Model training
"""

encoder = roberta_model
decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

model = Text2TextModel(encoder=encoder, decoder=decoder, config=config, beam_size=5, max_length=128,
                       sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
model.load_state_dict(torch.load('path_to_your_commonvoice_ted_model.bin'), strict=False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)


batch_size = 64
num_epochs = 40
learning_rate = 2e-5
adam_epsilon = 1e-8
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)


# define WER calculation and BLEU, GLEU calculation
def wer_calculation(output_texts, target_texts):
    total_distance = 0
    total_length = 0

    for output_text, target_text in zip(output_texts, target_texts):
        
        output_tokens = word_tokenize(output_text.lower())
        target_tokens = word_tokenize(target_text.lower())

        distance = edit_distance(output_tokens, target_tokens)
        total_distance += distance
        total_length += len(target_tokens)

    wer = total_distance / total_length

    return wer

def bleu_gleu_calculation(output_texts, target_texts):
    
    outputs = []
    targets = []
    
    for output_text, target_text in zip(output_texts, target_texts):
        
        output_tokens = word_tokenize(output_text.strip().lower())
        target_tokens = [word_tokenize(target_text.strip().lower())]
        
        outputs.append(output_tokens)
        targets.append(target_tokens)
    
    bleu_score = corpus_bleu(targets, outputs)
    gleu_score = corpus_gleu(targets, outputs)
    
    return bleu_score, gleu_score


# tokenization
max_source_length = 128
max_target_length = 128

train_source_ids = []
train_source_mask = []
train_target_ids = []
train_target_mask = []
val_source_ids = []
val_source_mask = []
val_target_ids = []
val_target_mask = []

for input_text in train_trans:
    source_tokens = tokenizer.tokenize(input_text)[:max_source_length - 2]
    source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_mask = [1] * (len(source_tokens))
    padding_length = max_source_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask += [0] * padding_length

    train_source_ids.append(source_ids)
    train_source_mask.append(source_mask)

for target_text in train_truth:            
    target_tokens = tokenizer.tokenize(target_text)[:max_target_length - 2]
    target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] * len(target_ids)
    padding_length = max_target_length - len(target_ids)
    target_ids += [tokenizer.pad_token_id] * padding_length
    target_mask += [0] * padding_length

    train_target_ids.append(target_ids)
    train_target_mask.append(target_mask)
    
for input_text in val_trans:
    source_tokens = tokenizer.tokenize(input_text)[:max_source_length - 2]
    source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_mask = [1] * (len(source_tokens))
    padding_length = max_source_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_mask += [0] * padding_length

    val_source_ids.append(source_ids)
    val_source_mask.append(source_mask)

train_source_ids = torch.tensor(train_source_ids)
train_source_mask = torch.tensor(train_source_mask)
train_target_ids = torch.tensor(train_target_ids)
train_target_mask = torch.tensor(train_target_mask)
val_source_ids = torch.tensor(val_source_ids)
val_source_mask = torch.tensor(val_source_mask)


# generate metric values for your original ASR transcript
wer = wer_calculation(val_trans, val_truth)
bleu, gleu = bleu_gleu_calculation(val_trans, val_truth)
print(f"Trans WER: {wer:.4f}, Trans BLEU: {bleu:.4f}, Trans GLEU: {gleu:.4f}")


# refine the val_truth data. If no change after doing this, then not necessary
val_truth_revised = []

for target_text in val_truth:
    encoded = tokenizer(target_text)
    decoded = tokenizer.decode(encoded["input_ids"], clean_up_tokenization_spaces=True, skip_special_tokens=True)
    val_truth_revised.append(decoded)


# model training
wers = []
bleus = []
gleus = []

torch.cuda.empty_cache()

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    total_loss = 0
    
    for i in range(0, len(train_data), batch_size):        
        batch_source_ids = train_source_ids[i:i+batch_size].to(DEVICE)
        batch_source_mask = train_source_mask[i:i+batch_size].to(DEVICE)
        batch_target_ids = train_target_ids[i:i+batch_size].to(DEVICE)
        batch_target_mask = train_target_mask[i:i+batch_size].to(DEVICE)
                
        loss, _, _ = model(batch_source_ids, batch_source_mask, batch_target_ids, batch_target_mask)
        
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        total_loss += loss.sum().item()
    
    avg_train_loss = total_loss / (len(train_data) // batch_size)

# evaluation
    model.eval()
    val_loss = 0.0
    generated_texts = []
    
    for i in range(0, len(val_data), batch_size):
        
        batch_source_ids = val_source_ids[i:i+batch_size].to(DEVICE)
        batch_source_mask = val_source_mask[i:i+batch_size].to(DEVICE)

        with torch.no_grad():
            preds = model(batch_source_ids, batch_source_mask, None, None)
            
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                text = text.strip()
                generated_texts.append(text)
#     print(generated_texts)                

    wer = wer_calculation(generated_texts, val_truth_revised)
    bleu, gleu = bleu_gleu_calculation(generated_texts, val_truth_revised)
    wers.append([epoch+1, wer])
    bleus.append([epoch+1, bleu])
    gleus.append([epoch+1, gleu])
    
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, WER: {wer:.4f}, BLEU: {bleu:.4f}, GLEU: {gleu:.4f}, Time: {epoch_time:.2f}s")

    torch.cuda.empty_cache()
    
best_wer = min(wers, key=lambda x: x[1])
best_bleu = max(bleus, key=lambda x: x[1])
best_gleu = max(gleus, key=lambda x: x[1])

print(f"Best WER:", best_wer)
print(f"Best BLEU:", best_bleu)
print(f"Best GLEU:", best_gleu)
