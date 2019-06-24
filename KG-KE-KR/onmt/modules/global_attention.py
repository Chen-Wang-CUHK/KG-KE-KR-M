""" Global attention modules (Luong / Bahdanau) """
import torch
import torch.nn as nn

from onmt.utils.misc import aeq, sequence_mask

# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, coverage=False, attn_type="dot", no_sftmax_bf_rescale=False, use_retrieved_keys=False):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        self.no_sftmax_bf_rescale = no_sftmax_bf_rescale
        self.use_retrieved_keys = use_retrieved_keys

        # assert (self.attn_type in ["dot", "general", "mlp"]), (
        #     "Please select a valid attention type.")

        assert (self.attn_type == "general"), "Currently, only general attention is supported."

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
            if self.use_retrieved_keys:
                self.rk_linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if self.use_retrieved_keys:
            self.linear_out = nn.Linear(dim * 3, dim, bias=out_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s, rk=False):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                if not rk:
                    h_t_ = self.linear_in(h_t_)
                else:
                    h_t_ = self.rk_linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, rk_memory_bank, memory_lengths=None,
                rk_memory_lengths=None, coverage=None, rk_final_state=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          rk_memory_bank (`FloatTensor`): retrieved keys vectors `[batch x rk_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          rk_memory_lengths (`LongTensor`): the rk lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)
          rk_final_state (`FloatTensor`): final state of the retrieved keys `[batch x dim]`

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank = memory_bank + self.linear_cover(cover).view_as(memory_bank)
            # memory_bank = self.tanh(memory_bank)

        # compute attention scores for the context, as in Luong et al.
        if rk_final_state is not None:
            align = self.score(source + rk_final_state.unsqueeze(1), memory_bank, rk=False)
        else:
            align = self.score(source, memory_bank, rk=False)

        mask = None
        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        if not self.no_sftmax_bf_rescale:
            # Softmax to normalize attention weights
            align_vectors = self.softmax(align.view(batch*target_l, source_l))
            align_vectors = align_vectors.view(batch, target_l, source_l)
        else:
            # Summation to normalize the sigmoided attention weights
            align = self.sigmoid(align)

            # calculate the norescaled_align_vectors
            if mask is not None:
                masked_align = align * mask.float()
            normalize_term = torch.sum(masked_align, dim=-1, keepdim=True)  # [batch x tgt_len x 1]
            align_vectors = masked_align / normalize_term

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        rk_c = None
        if self.use_retrieved_keys:
            _, rk_source_l, _ = rk_memory_bank.size()
            rk_align = self.score(source, rk_memory_bank, rk=True)

            rk_mask = None
            if rk_memory_lengths is not None:
                rk_mask = sequence_mask(rk_memory_lengths, max_len=rk_align.size(-1))
                rk_mask = rk_mask.unsqueeze(1)  # Make it broadcastable.
                rk_align.masked_fill_(1 - rk_mask, -float('inf'))

            # Softmax to normalize attention weights
            # add for RK
            rk_align_vectors = self.softmax(rk_align.view(batch * target_l, rk_source_l))
            rk_align_vectors = rk_align_vectors.view(batch, target_l, rk_source_l)

            """
            if not self.no_sftmax_bf_rescale:
                # Softmax to normalize attention weights
                # add for RK
                rk_align_vectors = self.softmax(rk_align.view(batch * target_l, rk_source_l))
                rk_align_vectors = rk_align_vectors.view(batch, target_l, rk_source_l)
            else:
                # Summation to normalize the sigmoided attention weights
                # add for RK
                rk_align = self.sigmoid(rk_align)
                # calculate the norescaled_align_vectors
                if rk_mask is not None:
                    rk_masked_align = rk_align * rk_mask.float()
                rk_normalize_term = torch.sum(rk_masked_align, dim=-1, keepdim=True)  # [batch x tgt_len x 1]
                rk_align_vectors = rk_masked_align / rk_normalize_term
            """

            # each rk context vector rk_c_t is the weighted average
            # over all the rk source hidden states
            rk_c = torch.bmm(rk_align_vectors, rk_memory_bank)

        # concatenate
        if rk_c is None:
            concat_c = torch.cat([c, source], 2).view(batch * target_l, dim * 2)
        else:
            concat_c = torch.cat([rk_c, c, source], 2).view(batch * target_l, dim * 3)

        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            target_l_, batch_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        return attn_h, align_vectors


class MyGlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, coverage=False, attn_type="dot",
                 not_use_sel_probs=False, no_sftmax_bf_rescale=False,
                 use_retrieved_keys=False, only_rescale_copy=False):
        super(MyGlobalAttention, self).__init__()

        self.not_use_sel_probs = not_use_sel_probs
        self.no_sftmax_bf_rescale = no_sftmax_bf_rescale
        self.use_retrieved_keys = use_retrieved_keys
        self.only_rescale_copy = only_rescale_copy
        self.dim = dim
        self.attn_type = attn_type

        # assert (self.attn_type in ["dot", "general", "mlp"]), (
        #     "Please select a valid attention type.")
        assert (self.attn_type == "general"), "Currently, only general attention is supported."

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
            if self.use_retrieved_keys:
                self.rk_linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if self.use_retrieved_keys:
            self.linear_out = nn.Linear(dim * 3, dim, bias=out_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s, rk=False):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                if not rk:
                    h_t_ = self.linear_in(h_t_)
                else:
                    h_t_ = self.rk_linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, rk_memory_bank, probs, memory_lengths=None,
                rk_memory_lengths=None, coverage=None, rk_final_state=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          probs (`FloatTensor`): key probabilities `[batch x src_len]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank = memory_bank + self.linear_cover(cover).view_as(memory_bank)
            # memory_bank = self.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        if rk_final_state is not None:
            align = self.score(source + rk_final_state.unsqueeze(1), memory_bank, rk=False)
        else:
            align = self.score(source, memory_bank, rk=False)  # [batch x tgt_len x src_len]


        # align_tmp = align.clone()

        mask = None
        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))    # [batch x src_len]
            mask = mask.unsqueeze(1)  # Make it broadcastable.  # [batch x 1 x src_len]
            align.masked_fill_(1 - mask, -float('inf'))

        if not self.no_sftmax_bf_rescale:
            # Softmax to normalize attention weights
            norescale_align_vectors = self.softmax(align.view(batch*target_l, source_l))
            norescale_align_vectors = norescale_align_vectors.view(batch, target_l, source_l)

            # changed for KE_KG
            if not self.not_use_sel_probs:
                probs = probs.unsqueeze(1)  # [batch x src_len] -> [batch x 1 x src_len]
                scaled_align = norescale_align_vectors * probs  # [batch x tgt_len x src_len]
            else:
                scaled_align = norescale_align_vectors * 1.0
        else:
            # Summation to normalize the sigmoided attention weights
            align = self.sigmoid(align)

            # calculate the norescaled_align_vectors
            if mask is not None:
                masked_align = align * mask.float()
            normalize_term = torch.sum(masked_align, dim=-1, keepdim=True)  # [batch x tgt_len x 1]
            norescale_align_vectors = masked_align / normalize_term

            if not self.not_use_sel_probs:
                probs = probs.unsqueeze(1)  # [batch x src_len] -> [batch x 1 x src_len]
                scaled_align = align * probs    # [batch x tgt_len x src_len]
            else:
                scaled_align = align * 1.0

        # probs = probs.unsqueeze(1)  # [batch x src_len] -> [batch x 1 x src_len]
        # scaled_align = norescale_align_vectors * probs  # [batch x tgt_len x src_len]

        if mask is not None:
            # scaled_align.masked_fill_(1 - mask, 0.0)
            scaled_align = scaled_align * mask.float()
        # normalize rescaled attention weights
        # note that softmax do not work well here!
        # align_vectors = self.softmax(scaled_align.view(batch * target_l, source_l))
        # align_vectors = align_vectors.view(batch, target_l, source_l)
        normalize_term = torch.sum(scaled_align, dim=-1, keepdim=True) # [batch x tgt_len x 1]
        align_vectors = scaled_align / normalize_term


        # # analysis the combination of the extraction probs ans generation attentions
        # src_i = 'definition and recognition of rib features in aircraft structural part . <eos> in this research , a new type of manufacturing feature that is commonly observed in aircraft structural parts , known as ribs , is defined and implemented using the object oriented software engineering approach . the rib feature type is defined as a set of constrained and adjacent faces of a part which are associated with a set of specific rib machining operations . computerized numerical control ( cnc ) operation experience and the machining knowledge are leveraged by analysing typical geometry interactions when generating machining tool paths where such knowledge and experience are abstracted as rules of process planning . then those abstracted machining process rules are implemented in a feature recognition algorithm on top of an existing and holistic attribute adjacency graph solution to extract seed faces , identify individual local rib elements and further cluster these newly identified local rib elements into groups for the ease of machining operations . out of the potentially different combinations of local rib elements , those optimised cluster groups are merged into the top level rib features . the enhanced recognition algorithm is presented in details . a pilot system has already been developed and applied for machining many advanced aircraft structural parts in a large aircraft manufacturer . observations and conclusions are presented at the end .'
        # src_i = src_i.split()
        # tgt_indicators_i = 'O O I O I O O I I I O O O O O O O O O O O I O O O O O I I O O O O O O O O O O O O O O O O O O O I I O O O O O O O O O O O O O I O O O O O O O O I I O O O O O O O O O O O O I O O O O O O O O O O I O O O O O O O O O O O O O O O O O O I O O O O O O I I O O O O O O O O O O O O O O O O O O O O I O O O O O O O O I O O O O O O O I O O O O O O O O O O I O O O O O O O O O O O O I O O O O I O O O O O O O O O O O O O O O O I O O I I O O O O I O O O O O O O O O O O'
        # tgt_indicators_i = tgt_indicators_i.split()
        #
        # probs_i = probs[0, 0, :]
        # attn_norescale_i = norescale_align_vectors[0, 0, :]
        # attn_rescaled_i = align_vectors[0, 0, :]
        #
        # sigmoid_e_i = torch.sigmoid(align[0, 0, :])
        # sigmoid_e_rescaled = sigmoid_e_i * probs_i
        # sigmoid_e_rescaled = sigmoid_e_rescaled / sigmoid_e_rescaled.sum()
        #
        # for idx in range(229):
        #     print('{:20}|{}|{:10.4f}|{:10.4f}|{:10.4f}||{:10.4f}|{:10.4f}|'.format(
        #         src_i[idx], tgt_indicators_i[idx], probs_i[idx].item(), attn_norescale_i[idx].item(), attn_rescaled_i[idx].item(),
        #         sigmoid_e_i[idx].item(), sigmoid_e_rescaled[idx].item()))


        # each context vector c_t is the weighted average
        # over all the source hidden states
        if not self.only_rescale_copy:
            c = torch.bmm(align_vectors, memory_bank)
        else:
            c = torch.bmm(norescale_align_vectors, memory_bank)

        rk_c = None
        if self.use_retrieved_keys:
            _, rk_source_l, _ = rk_memory_bank.size()
            rk_align = self.score(source, rk_memory_bank, rk=True)

            rk_mask = None
            if rk_memory_lengths is not None:
                rk_mask = sequence_mask(rk_memory_lengths, max_len=rk_align.size(-1))
                rk_mask = rk_mask.unsqueeze(1)  # Make it broadcastable.
                rk_align.masked_fill_(1 - rk_mask, -float('inf'))

            # Softmax to normalize attention weights
            # add for RK
            rk_align_vectors = self.softmax(rk_align.view(batch * target_l, rk_source_l))
            rk_align_vectors = rk_align_vectors.view(batch, target_l, rk_source_l)

            """
            if not self.no_sftmax_bf_rescale:
                # Softmax to normalize attention weights
                # add for RK
                rk_align_vectors = self.softmax(rk_align.view(batch * target_l, rk_source_l))
                rk_align_vectors = rk_align_vectors.view(batch, target_l, rk_source_l)
            else:
                # Summation to normalize the sigmoided attention weights
                # add for RK
                rk_align = self.sigmoid(rk_align)
                # calculate the norescaled_align_vectors
                if rk_mask is not None:
                    rk_masked_align = rk_align * rk_mask.float()
                rk_normalize_term = torch.sum(rk_masked_align, dim=-1, keepdim=True)  # [batch x tgt_len x 1]
                rk_align_vectors = rk_masked_align / rk_normalize_term
            """

            # each rk context vector rk_c_t is the weighted average
            # over all the rk source hidden states
            rk_c = torch.bmm(rk_align_vectors, rk_memory_bank)

        # concatenate
        if rk_c is None:
            concat_c = torch.cat([c, source], 2).view(batch * target_l, dim * 2)
        else:
            concat_c = torch.cat([rk_c, c, source], 2).view(batch * target_l, dim * 3)

        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
            norescale_align_vectors = norescale_align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            norescale_align_vectors = norescale_align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            target_l_, batch_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        return attn_h, align_vectors, norescale_align_vectors
