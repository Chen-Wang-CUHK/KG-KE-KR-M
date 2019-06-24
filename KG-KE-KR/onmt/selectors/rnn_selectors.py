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


class RNNSelector(EncoderBase):
    """ A generic selector based on RNNs.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
       sel_classifier (string): the chosen classifier type
       detach_sel_probs (bool): If true, detach the predicted sel_probs when training
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None, sel_classifier='simple_fc', detach_sel_probs=False):
        super(RNNSelector, self).__init__()
        assert embeddings is not None
        assert bidirectional

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.hidden_size = hidden_size
        self.embeddings = embeddings

        self.detach_sel_probs = detach_sel_probs

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0.0,
                        bidirectional=bidirectional)

        # add a simplest classifier first to build the whole framework
        # activation function: sigmoid
        self.sel_classifier = sel_classifier
        if sel_classifier == 'simple_fc':
            self.word_feats_linear = nn.Linear(2 * hidden_size, hidden_size)
            self.simple_classifier = nn.Linear(hidden_size, 1)
        else:
            # for complex_Nallapati classifier, refer to https://arxiv.org/pdf/1611.04230.pdf
            self.word_feats_linear = nn.Linear(2 * hidden_size, hidden_size)
            self.art_feats_linear = nn.Linear(2 * hidden_size, hidden_size)

            self.w_content = nn.Linear(hidden_size, 1)
            self.w_salience = nn.Linear(hidden_size, hidden_size, bias=False)
            self.w_novelty = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout_m = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        seq_len, batch_size, _ = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        # encoder_final: [2(directions), batch_size, hidden_dim]
        # memory_bank: [seq_len, batch_size, 2 * hidden_dim]
        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        memory_bank = self.dropout_m(memory_bank)
        if self.sel_classifier == 'simple_fc':
            word_feats = self.tanh(self.word_feats_linear(memory_bank))
            logits = self.simple_classifier(word_feats) # [seq_len, batch_size, 1]
            logits = logits.view(seq_len, batch_size)  # [seq_len, batch_size]
            probs = self.sigmoid(logits)  # [seq_len, batch_size]
        else:
            # [corrected]
            word_feats = self.tanh(self.word_feats_linear(memory_bank)) # [seq_len, batch_size, hidden_size]

            # use the concatenation of the final state as the article feature
            art_feats = torch.cat([encoder_final[0], encoder_final[1]], dim=-1)

            art_feats = self.art_feats_linear(art_feats)    # [batch_size, hidden_size]
            art_feats = self.tanh(art_feats)
            s = torch.zeros(batch_size, self.hidden_size).cuda()

            logits = []
            probs = []
            for i in range(seq_len):
                content_feats = self.w_content(word_feats[i, :, :]) # [batch_size, 1]
                salience_feats = torch.sum(self.w_salience(word_feats[i, :, :]) * art_feats, dim=1, keepdim=True) # [batch_size, 1]
                novelty_feats = torch.sum(self.w_novelty(word_feats[i, :, :]) * self.tanh(s), dim=1, keepdim=True)
                logit = content_feats + salience_feats - novelty_feats # [batch_size, 1]
                logits.append(logit)

                prob = self.sigmoid(logit)
                probs.append(prob)

                s = s + word_feats[i, :, :] * prob

            logits = torch.cat(logits, dim=1).transpose(0, 1) # [seq_len, batch_size]
            probs = torch.cat(probs, dim=1).transpose(0, 1) # [seq_len, batch_size]

        if self.detach_sel_probs:
            probs = probs.detach()

        return logits, probs, encoder_final, memory_bank

