import torch.nn as nn
from .transformer_component.Block import Encoder, Decoder
import torch


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,
                           bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq_enc, len_max_seq_dec,
            d_word_vec=512, d_model=512, d_inner=1024,
            n_layers_enc=6, n_layers_dec=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=False,
            emb_src_tgt_weight_sharing=False):

        super().__init__()
        self.num_classes = n_tgt_vocab
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq_enc,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers_enc, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq_dec,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers_dec, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
                "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, batch_max_length, is_train=True):
        if is_train:
            tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
            enc_output, *_ = self.encoder(src_seq, src_pos)
            dec_output, * \
                _ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
            seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale
            return seq_logit
        else:
            batch_size = src_seq.size(0)
            num_steps = batch_max_length + 1
            seq_logit = torch.cuda.FloatTensor(
                batch_size, num_steps, self.num_classes).fill_(0)

            enc_output, *_ = self.encoder(src_seq, src_pos)
            if tgt_pos is not None:
                pos = tgt_pos
            else:
                pos = torch.arange(
                    1, num_steps+1, dtype=torch.long, device='cuda').expand(batch_size, -1)
            ys = torch.zeros(batch_size, num_steps+1).long().cuda()
            ys[:, 0] = Constants.BOS
            for i in range(num_steps):
                out, *_ = self.decoder(ys[:, :i+1],
                                       pos[:, :i+1], src_seq, enc_output)
                prob = self.tgt_word_prj(out) * self.x_logit_scale
                seq_logit[:, i, :] = prob[:, -1, :]
                _, next_word = torch.max(prob[:, -1, :], dim=1)
                ys[:, i+1] = next_word

            return seq_logit
