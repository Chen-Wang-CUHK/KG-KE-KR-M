"""Define RNN-based encoders."""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

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
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        # self.use_retrieved_keys = use_retrieved_keys
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers != 1 else 0.0,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, src_lengths=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, src_lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if src_lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            src_lengths = src_lengths.view(-1).tolist()
            packed_emb = pack(emb, src_lengths)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if src_lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)

        return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class RKRNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

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
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RKRNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rk_rnn, self.rk_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers != 1 else 0.0,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, retrieved_keys, rk_lengths=None):
        "See :obj:`EncoderBase.forward()`"
        # encoding retrieved keyphrases
        assert retrieved_keys is not None
        self._check_args(retrieved_keys, rk_lengths)

        rk_len, _, _ = retrieved_keys.size()
        # print('rk_len: {}'.format(rk_len))
        rk_emb = self.embeddings(retrieved_keys)
        # rk_len, batch, emb_dim = rk_emb.size()
        # print(rk_len)

        # sort retrieved keys w.r.t rk_lengths
        sorted_rk_lengths, idx_sort = torch.sort(rk_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        rk_packed_emb = rk_emb.index_select(1, idx_sort)
        if rk_lengths is not None and not self.rk_no_pack_padded_seq:
            sorted_rk_lengths = sorted_rk_lengths.view(-1).tolist()
            rk_packed_emb = pack(rk_packed_emb, sorted_rk_lengths)

        rk_memory_bank, rk_encoder_final = self.rk_rnn(rk_packed_emb)

        if rk_lengths is not None and not self.rk_no_pack_padded_seq:
            rk_memory_bank = unpack(rk_memory_bank)[0]
            rk_memory_bank = rk_memory_bank.index_select(1, idx_unsort)
            rk_encoder_final = rk_encoder_final.index_select(1, idx_unsort)

        if self.use_bridge:
            rk_encoder_final = self._bridge(rk_encoder_final)

        return rk_encoder_final, rk_memory_bank

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs