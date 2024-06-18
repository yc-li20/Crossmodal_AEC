import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from nltk.translate.bleu_score import corpus_bleu, corpus_gleu
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
from torch.nn import TransformerDecoderLayer, TransformerDecoder, MultiheadAttention

class Text2TextModel(nn.Module):
    def __init__(self, encoder, decoder, config, beam_size, max_length, sos_id, eos_id):
        super(Text2TextModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(512, 512)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        self.attn = nn.MultiheadAttention(768, 8, batch_first=False)

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        
    def forward(self, source_ids, source_mask, target_ids, target_mask, audio):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        audio_output = audio.permute([1, 0, 2]).contiguous() # max_len * bsz * hidden
        audio_output, _ = self.attn(encoder_output,audio_output,audio_output)
        encoder_output = encoder_output+audio_output
        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask, memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss], shift_labels.view(-1)[active_loss])
            return loss, loss * active_loss.sum(), active_loss.sum()
        else:
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
        self.scores = self.tt.FloatTensor(size).zero_()
        self.prevKs = []
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = sos
        self._eos = eos
        self.eosTop = False
        self.finished = []

    def getCurrentState(self):
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        return self.prevKs[-1]

    def advance(self, wordLk):
        numWords = wordLk.size(1)

        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished =
