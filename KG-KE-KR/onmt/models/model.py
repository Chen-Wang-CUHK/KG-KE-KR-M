""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, rk_encoder, decoder, multigpu=False, rk_to_src_attn=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.rk_encoder = rk_encoder
        self.decoder = decoder

        self.rk_to_src_attn = rk_to_src_attn

    def forward(self, src, tgt, src_lengths, dec_state=None, retrieved_keys=None, rk_lengths=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_final, memory_bank = self.encoder(src, src_lengths)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)

        rk_memory_bank = None
        rk_final_state = None
        if self.rk_encoder is not None:
            rk_final_state, rk_memory_bank = self.rk_encoder(retrieved_keys, rk_lengths)
            if self.rk_to_src_attn:
                rk_final_state = torch.cat([rk_final_state[0], rk_final_state[1]], dim=-1)
                rk_final_state = rk_final_state.squeeze()
            else:
                rk_final_state = None

        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank, rk_memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=src_lengths,
                         rk_memory_lengths=rk_lengths,
                         rk_final_state=rk_final_state)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class SelectorModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface

    Args:
      selector (:obj:``): an selector object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, selector, multigpu=False):
        self.multigpu = multigpu
        super(SelectorModel, self).__init__()
        self.selector = selector

    def forward(self, src, lengths):
        """Forward propagate a `src`
        Args:
            src (:obj:`LongTensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
        Returns:
            (:obj:`FloatTensor`, :obj:`FloatTensor`):
                 * logits output of the selector `[src_len, batch]`
                 * importance scores output of the selector `[src_len, batch]`
        """
        logits, probs, _, _ = self.selector(src, lengths)
        return logits, probs


class E2EModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder, selector + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      selector (:obj:`Selector`): an selector object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, rk_encoder, selector, decoder, multigpu=False, e2e_type="separate_enc_sel",
                 use_gt_sel_probs=False, rk_to_src_attn=False):
        assert e2e_type in ["separate_enc_sel", "share_enc_sel"]

        self.multigpu = multigpu
        super(E2EModel, self).__init__()

        self.e2e_type = e2e_type
        if e2e_type == "separate_enc_sel":
            self.encoder = encoder
        self.selector = selector
        self.rk_encoder = rk_encoder
        self.decoder = decoder

        self.use_gt_sel_probs = use_gt_sel_probs
        self.rk_to_src_attn = rk_to_src_attn

    def forward(self, src, tgt, lengths, dec_state=None, gt_probs=None, retrieved_keys=None, rk_lengths=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[src_len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state.
            gt_probs (:obj:`LongTensor`): the ground-truth importance scores, `[src_len x batch]`
            retrieved_keys (:obj:`LongTensor`): the retrieved keyphrases, `[rk_len x batch]`
            rk_lengths (:obj:`LongTensor`): the retrieved keyphrase lengths, `[rk_len x batch]`
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`, :obj:`FloatTensor`, :obj:`FloatTensor`):
                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
                 * logits output of the selector `[src_len, batch]`
                 * importance scores output of the selector `[src_len, batch]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        if self.e2e_type == 'separate_enc_sel':
            logits, probs, _, _, = self.selector(src, lengths)
            enc_final, memory_bank,  = self.encoder(src, lengths)
        else:
            logits, probs, enc_final, memory_bank = self.selector(src, lengths)

        rk_memory_bank = None
        rk_final_state = None
        if self.rk_encoder is not None:
            rk_final_state, rk_memory_bank = self.rk_encoder(retrieved_keys, rk_lengths)
            if self.rk_to_src_attn:
                rk_final_state = torch.cat([rk_final_state[0], rk_final_state[1]], dim=-1)
                rk_final_state = rk_final_state.squeeze()
            else:
                rk_final_state = None

        # use the gound-truth binary importance score of each source text token
        if gt_probs is not None and self.use_gt_sel_probs:
            gt_probs = (gt_probs == 2)
            gt_probs = gt_probs.float()
            probs = gt_probs

        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank, rk_memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths,
                         rk_memory_lengths=rk_lengths,
                         probs=probs,
                         rk_final_state=rk_final_state)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state, logits, probs