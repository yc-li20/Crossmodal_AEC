import torch
from torch import nn

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
        audio_output = audio.permute([1, 0, 2]).contiguous()
        audio_output, _ = self.attn(encoder_output, audio_output, audio_output)
        encoder_output = encoder_output + audio_output

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
            return self.generate(source_ids, source_mask, encoder_output)

    def generate(self, source_ids, source_mask, encoder_output):
        preds = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        for i in range(source_ids.shape[0]):
            context = encoder_output[:, i:i + 1]
            context_mask = source_mask[i:i + 1, :]
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            input_ids = beam.getCurrentState().clone().detach().cuda()
            context = context.repeat(1 * self.beam_size, 1, 1)
            context_mask = context_mask.repeat(self.beam_size, 1)
            for _ in range(self.max_length):
                if beam.done():
                    break
                attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask, memory_key_padding_mask=(1 - context_mask).bool())
                hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
                lm_logits = self.lsm(self.lm_head(hidden_states[:, -1, :].contiguous())).data
                beam.advance(lm_logits)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), dim=1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))) for p in pred]
            preds.append(pred)
        preds = [torch.stack(p) for p in preds]
        return preds

class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self._done = False
        self.scores = torch.FloatTensor(size).zero_()
        self.all_scores = []
        self.prev_ks = []
        self.next_ys = [torch.LongTensor(size).fill_(0)]
        self.next_ys[0][0] = sos
        self.eos = eos

    def getCurrentState(self):
        batch = self.next_ys[-1].view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        return self.prev_ks[-1]

    def advance(self, word_lk):
        num_words = word_lk.size(1)
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]
        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        self.all_scores.append(self.scores)
        self.scores = best_scores
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        if self.next_ys[-1][0] == self.eos:
            self._done = True

    def done(self):
        return self._done

    def getHyp(self, k):
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return hyp[::-1]

    def buildTargetTokens(self, preds):
        tokens = []
        for tok in preds:
            tokens.append(tok)
            if tok == self.eos:
                break
        return tokens
