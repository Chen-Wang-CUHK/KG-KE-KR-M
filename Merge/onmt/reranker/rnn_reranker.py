"""Define RNN-based selectors."""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

from onmt.utils.misc import sequence_mask


class RNNReRanker(EncoderBase):
    """ A generic selector based on RNNs.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNReRanker, self).__init__()
        assert embeddings is not None
        assert bidirectional

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.hidden_size = hidden_size
        self.embeddings = embeddings

        self.src_rnn, self.src_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0.0,
                        bidirectional=bidirectional)

        self.tgt_rnn, self.tgt_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0.0,
                        bidirectional=bidirectional)

        self.mlp_f = self._mlp_layers(2 * self.hidden_size, 2 * self.hidden_size, dropout=dropout)
        self.mlp_g = self._mlp_layers(4 * self.hidden_size, 2 * self.hidden_size, dropout=dropout)
        self.mlp_h = self._mlp_layers(4 * self.hidden_size, 2 * self.hidden_size, dropout=dropout)

        self.final_linear = nn.Linear(2 * self.hidden_size, 1, bias=True)

        self.dropout_m = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

    def _mlp_layers(self, input_dim, output_dim, dropout):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=dropout))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=dropout))
        mlp_layers.append(nn.Linear(
            output_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        return nn.Sequential(*mlp_layers)  # * used to unpack list

    def forward(self, input, lengths_=None):
        assert isinstance(input, tuple)
        src, tgt = input

        if lengths_ is not None:
            assert isinstance(lengths_, tuple)
            src_lengths, tgt_lengths = lengths_
        else:
            src_lengths = None
            tgt_lengths = None

        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, src_lengths)
        self._check_args(tgt, tgt_lengths)

        # encoding the src
        emb = self.embeddings(src)
        src_len, batch_size, _ = emb.size()

        packed_emb = emb
        if src_lengths is not None and not self.src_no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            # src_lengths = src_lengths.view(-1).tolist()
            packed_emb = pack(emb, src_lengths.view(-1).tolist())

        # encoder_final: [2(directions), batch_size, hidden_dim]
        # memory_bank: [seq_len, batch_size, 2 * hidden_dim]
        src_memory_bank, src_encoder_final = self.src_rnn(packed_emb)

        if src_lengths is not None and not self.src_no_pack_padded_seq:
            src_memory_bank = unpack(src_memory_bank)[0]

        # encoding the tgt
        emb = self.embeddings(tgt)
        tgt_len, batch_size, _ = emb.size()

        # sort retrieved keys w.r.t tgt_lengths
        sorted_tgt_lengths, idx_sort = torch.sort(tgt_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        packed_emb = emb.index_select(1, idx_sort)
        if tgt_lengths is not None and not self.tgt_no_pack_padded_seq:
            sorted_tgt_lengths = sorted_tgt_lengths.view(-1).tolist()
            packed_emb = pack(packed_emb, sorted_tgt_lengths)

        tgt_memory_bank, tgt_encoder_final = self.tgt_rnn(packed_emb)

        if tgt_lengths is not None and not self.tgt_no_pack_padded_seq:
            tgt_memory_bank = unpack(tgt_memory_bank)[0]
            tgt_memory_bank = tgt_memory_bank.index_select(1, idx_unsort)
            tgt_encoder_final = tgt_encoder_final.index_select(1, idx_unsort)

        # if self.use_bridge:
        #     src_encoder_final = self._bridge(src_encoder_final)
        #     tgt_encoder_final = self._bridge(tgt_encoder_final)

        # attention
        '''
            sent_linear: batch_size x length x hidden_size
        '''
        src_mask = sequence_mask(src_lengths, max_len=src_len)  # [batch, src_len]
        #src_mask = src_mask.unsqueeze(2)  # Make it broadcastable. [batch, src_len, 1]
        tgt_mask = sequence_mask(tgt_lengths, max_len=tgt_len)  # [batch, tgt_len]
        #tgt_mask = tgt_mask.unsqueeze(1)  # Make it broadcastable. [batch, 1, tgt_len]
        #final_mask = torch.bmm(src_mask.unsqueeze(2).float(), tgt_mask.unsqueeze(1).float()).byte()    # [batch, src_len, tgt_len]

        sent1_linear = src_memory_bank.transpose(0, 1).contiguous()
        sent2_linear = tgt_memory_bank.transpose(0, 1).contiguous()
        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)
        assert src_len == len1
        assert tgt_len == len2

        '''attend'''

        f1 = self.mlp_f(sent1_linear.view(-1, 2 * self.hidden_size))
        f2 = self.mlp_f(sent2_linear.view(-1, 2 * self.hidden_size))

        f1 = f1.view(-1, len1, 2 * self.hidden_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, 2 * self.hidden_size)
        # batch_size x len2 x hidden_size

        score = torch.bmm(f1, f2.transpose(1, 2)) # batch_size x len1 x len2

        #score1 = score
        score1 = score.masked_fill((1 - tgt_mask.unsqueeze(1)).byte(), -float('inf'))
        # e_{ij} batch_size x len1 x len2
        score1 = score1.contiguous()

        prob1 = F.softmax(score1.view(-1, len2), dim=-1).view(-1, len1, len2)
        # batch_size x len1 x len2

        score2 = score.transpose(1, 2)
        # e_{ij} batch_size x len2 x len1
        score2 = score2.masked_fill((1 - src_mask.unsqueeze(1)).byte(), -float('inf'))
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1), dim=-1).view(-1, len2, len1)
        # batch_size x len2 x len1

        sent1_combine = torch.cat(
            (sent1_linear, torch.bmm(prob1, sent2_linear)), 2)
        # batch_size x len1 x (hidden_size x 4)
        sent2_combine = torch.cat(
            (sent2_linear, torch.bmm(prob2, sent1_linear)), 2)
        # batch_size x len2 x (hidden_size x 4)

        '''sum'''
        g1 = self.mlp_g(sent1_combine.view(-1, 4 * self.hidden_size))
        g2 = self.mlp_g(sent2_combine.view(-1, 4 * self.hidden_size))
        g1 = g1.view(-1, len1, 2 * self.hidden_size)
        g1 = g1 * src_mask.unsqueeze(2).float()
        # batch_size x len1 x (hidden_size * 2)
        g2 = g2.view(-1, len2, 2 * self.hidden_size)
        g2 = g2 * tgt_mask.unsqueeze(2).float()
        # batch_size x len2 x (hidden_size * 2)

        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x (hidden_size * 2)
        # sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x (hidden_size * 2)
        # sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (4 * hidden_size)
        h = self.mlp_h(input_combine)
        # batch_size * (2 *hidden_size)

        # if sample_id == 15:
        #     print '-2 layer'
        #     print h.data[:, 100:150]

        logits = self.final_linear(h)
        # batch_size

        # print 'final layer'
        # print h.data

        probs = self.sigmoid(logits)

        return logits, probs

