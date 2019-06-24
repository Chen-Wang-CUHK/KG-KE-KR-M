"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn

import onmt
import onmt.inputters as inputters

import random

from onmt.utils.logging import logger


def my_build_loss_compute(model, indicator_vocab, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    if opt.key_model != 'key_selector':
        assert opt.copy_attn
        generator = model.generator
    else:
        generator = None
    compute = E2ELossCompute(opt=opt,
                             generator=generator,
                             indicator_vocab=indicator_vocab,
                             tgt_vocab=tgt_vocab)

    compute.to(device)

    return compute


def build_loss_compute(model, tgt_vocab, opt, train=True):
    """
    This returns user-defined LossCompute object, which is used to
    compute loss in train/validate process. You can implement your
    own *LossCompute class, by subclassing LossComputeBase.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            model.generator, tgt_vocab, opt.copy_attn_force,
            opt.copy_loss_by_seqlength)
    else:
        compute = NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing if train else 0.0)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = onmt.utils.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class InconsistencyLoss(object):
    def __init__(self, top_k, eps=1e-20):
        self.top_k = top_k
        self.eps = eps

    def __call__(self, sel_probs, sel_mask, norescale_attns, dec_mask, normalize_by_length):
        """
        Calculate the inconsistency loss between the selector predicted probs and the norescale_attns.
        Refer to https://aclweb.org/anthology/P18-1013 for more details. Some code is borrowed from the
        released code of this paper.
        Args:
            sel_probs (:obj:'FloatTensor') : the predicted probabilities by the selector [src_len, batch]
            sel_mask (:obj:'FloatTensor') : the mask for sel_probs [src_len, batch]
            norescale_attns: (:obj:'FloatTensor'): the norescaled attention scores [tgt_len, batch, src_len]
            dec_mask (:obj:'FloatTensor') : the mask for the decoder output [tgt_len, batch]
            normalize_by_length (:obj:`bool`) : If true, normalize the loss with the length
        return:
            Inconsistency loss:
                - (1/T) * sum(t=1,...,T)(log((1/top_k) * sum(k=1,...,top_k)(norescale_attns_t_k * sel_probs_t_k)))
        """
        #
        sel_probs = sel_probs * sel_mask
        sel_probs = sel_probs.transpose(0, 1) # [batch, src_len]
        losses = []
        for dec_step, attn_dist in enumerate(norescale_attns.split(1)):
            attn_dist = attn_dist.squeeze(dim=0)
            attn_topk, attn_topk_id = torch.topk(attn_dist, k=self.top_k, dim=1) # [batch, topk]
            sel_topk = torch.gather(sel_probs, dim=1, index=attn_topk_id)   # [batch, topk]
            # mean first than log
            loss_one_step = torch.mean(attn_topk * sel_topk, dim=1) # [batch]
            loss_one_step = -torch.log(loss_one_step + self.eps)    # [batch]
            loss_one_step = loss_one_step * dec_mask[dec_step, :]  # [batch]
            losses.append(loss_one_step)
        if normalize_by_length:
            loss = (sum(losses) / torch.mean(dec_mask, dim=0)).sum()
        else:
            loss = (sum(losses)).sum()
        return loss


class E2ELossCompute(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        indicator_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the ground-truth keyword indicators
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, opt, generator, indicator_vocab, tgt_vocab, eps=1e-20):
        super(E2ELossCompute, self).__init__()

        self.key_model = opt.key_model
        self.tgt_vocab = tgt_vocab
        self.cur_dataset = None
        self.force_copy = opt.copy_attn_force
        self.top_k = opt.top_k
        self.sel_report_topk = opt.top_k
        self.sel_normalize_by_length = opt.sel_normalize_by_length
        self.gen_normalize_by_length = opt.gen_normalize_by_length
        self.incons_normalize_by_length = opt.incons_normalize_by_length
        self.pos_weight = opt.pos_weight    # default 9.0
        self.sel_threshold = opt.sel_threshold  # default 0.9
        self.sel_lambda = opt.sel_lambda    # default 0.5
        self.sel_train_ratio = opt.sel_train_ratio # default 1.0
        self.gen_lambda = opt.gen_lambda    # default 0.5
        self.incons_lambda = opt.incons_lambda  # default 0.5

        self.generator = generator
        if opt.key_model != 'key_selector':
            self.tgt_padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]
        if opt.key_model != 'key_generator':
            assert len(indicator_vocab) == 3
            self.src_unk_idx = inputters.UNK
            self.sel_padding_idx = indicator_vocab.stoi[inputters.PAD_WORD]
            self.pos_idx = indicator_vocab.stoi['I']
            self.neg_idx = indicator_vocab.stoi['O']

        # BCEWithLogits loss for extraction (selector)
        if self.key_model == 'key_selector' or (self.key_model == 'key_end2end' and self.sel_lambda != 0.0):
            self.bcewithlogits_criterion =\
                nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([self.pos_weight]))
        else:
            self.bcewithlogits_criterion = None

        # CopyGenerator loss for generation (generator)
        # (len(tgt_vocab), force_copy, self.padding_idx)
        if self.key_model == 'key_generator' or self.key_model == 'key_end2end':
            self.copynmtloss_criterion =\
                onmt.modules.CopyGeneratorCriterion(len(tgt_vocab), self.force_copy, self.tgt_padding_idx)
        else:
            self.copynmtloss_criterion = 0.0

        # inconsistency loss for extraction attention and generating attention
        if self.key_model == 'key_end2end' and self.incons_lambda != 0.0:
            self.inconsistloss_criterion = InconsistencyLoss(top_k=self.top_k)

        if not self.sel_normalize_by_length:
            logger.warning("These selector losses will not be normalized by length since opt.sel_normalize_by_length=False!")
        if not self.gen_normalize_by_length:
            logger.warning("These generator losses will not be normalized by length since opt.gen_normalize_by_length=False!")
        if not self.incons_normalize_by_length:
            logger.warning("These inconsisitency losses will not be normalized by length since opt.incons_normalize_by_length=False!")

    def _make_shard_state(self, batch, sel_output, sel_probs, dec_output, attns):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            sel_output: the predicted logits output from the selector.
            sel_probs: the predicted importance scores output from the selector.
            dec_output: the predicted output from the decoder.
            attns: the attns dictionary returned from the decoder.
        """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        return {"sel_output": sel_output,
                "sel_probs": sel_probs,
                "dec_output": dec_output,
                "copy_attn": None if attns is None else attns.get("copy"),
                "norescale_attn": None if attns is None else attns.get("norescale_std")}

    def _compute_extraction_loss(self, batch, sel_output):
        """
        Compute the extraction loss.

        Args:
            batch: the current batch.
            sel_output: the predicted logits from the selector.
        """
        sel_target = batch.key_indicators[0]

        # get the sequence mask
        pad_mask = sel_target.ne(self.sel_padding_idx).float()
        # filter out the unk keywords
        src_unk_mask = batch.src[0].ne(self.src_unk_idx).float()
        # final mask
        mask = pad_mask * src_unk_mask

        gt_indicators = sel_target.eq(self.pos_idx).float()

        # get the loss tensor
        if self.sel_lambda != 0.0:
            sel_loss = self.bcewithlogits_criterion(sel_output, gt_indicators)
            sel_loss = sel_loss * mask
            if self.sel_normalize_by_length:
                # Compute Loss as BCE divided by seq length
                # Compute Sequence Lengths
                tgt_lens = mask.sum(0)
                # Compute Total Loss per sequence in batch
                sel_loss = sel_loss.view(-1, batch.batch_size).sum(0)
                # Divide by length of each sequence and sum
                sel_loss = torch.div(sel_loss, tgt_lens).sum()
            else:
                sel_loss = sel_loss.sum()
        else:
            sel_loss = torch.Tensor([0.0])

        return sel_loss, gt_indicators, mask

    def _compute_generation_loss(self, batch, dec_output, copy_attn):
        """
        Compute the generation loss.
        Args:
            batch: the current batch.
            dec_output: the predicted output from the decoder. [tgt_len, batch, hidden_size]
            dec_target: the validate target to compare output with.
            copy_attn: the copy attention value. [tgt_len, batch, src_len]
        """
        dec_target = batch.tgt[1:]  # [tgt_len, batch]
        dec_mask = dec_target.ne(self.tgt_padding_idx).float()  # [tgt_len, batch]
        dec_target = dec_target.view(-1)    # [tgt_len * batch]

        align = batch.alignment[1:]  # [tgt_len, batch]
        align = align.view(-1)   # [tgt_len * batch]

        scores = self.generator(self._bottle(dec_output),
                                self._bottle(copy_attn),
                                batch.src_map)  # [tgt_len * batch, vocab_size + copy_vocab_size]
        loss = self.copynmtloss_criterion(scores, align, dec_target)  # [tgt_len * batch_size]
        scores_data = scores.data.clone()
        scores_data = inputters.TextDataset.collapse_copy_scores(
            self._unbottle(scores_data, batch.batch_size),
            batch, self.tgt_vocab, self.cur_dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = dec_target.data.clone()
        correct_mask = target_data.eq(0) * align.data.ne(0)
        correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
        target_data = target_data + correct_copy

        # Compute sum of perplexities for stats
        loss_data = loss.sum().data.clone()
        # stats = self._stats(loss_data, scores_data, target_data)

        if self.gen_normalize_by_length:
            # Compute Loss as NLL divided by seq length
            # Compute Sequence Lengths
            tgt_lens = dec_mask.sum(0)
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, loss_data, scores_data, target_data, dec_mask

    def _compute_loss(self, batch, sel_output, sel_probs, dec_output, copy_attn, norescale_attn):
        """
        Compute the loss. Subclass must define this method.
        Args:
            batch: the current batch.
            sel_output: the predicted output(logits) from the selector
            sel_probs: the predicted importance scores output from the selector
            dec_output: the predicted output from the decoder
            copy_attn: the rescaled copy attention
            norescale_attn: the original attention
        """
        # calculate the extraction loss
        if self.key_model in ('key_selector', 'key_end2end'):
            sel_loss, gt_indicators, sel_mask = self._compute_extraction_loss(batch, sel_output)

        # calculate the generation loss
        # Note: the gen_loss maybe the length-normalized value of gen_loss_data
        if self.key_model in ('key_generator', 'key_end2end'):
            gen_loss, gen_loss_data, gen_scores_data, gen_target_data, dec_mask =\
                self._compute_generation_loss(batch, dec_output, copy_attn)

        # calculate the inconsistency loss
        # (self, sel_probs, sel_mask, norescale_attns, dec_mask, normalize_by_length)
        if self.key_model == 'key_end2end':
            if self.incons_lambda != 0.0:
                incons_loss \
                    = self.inconsistloss_criterion(sel_probs, sel_mask, norescale_attn, dec_mask, self.incons_normalize_by_length)
            else:
                incons_loss = torch.Tensor([0.0])

        # merge all the loss according to the options
        if self.key_model == 'key_selector':
            assert self.sel_lambda != 0.0
            merged_loss = self.sel_lambda * sel_loss
            stats = self._stats(batch_size=batch.batch_size,
                                sel_loss_data=sel_loss.data.clone(),
                                sel_probs=sel_probs.data.clone(),
                                gt_indicators=gt_indicators.data.clone(),
                                sel_mask=sel_mask.data.clone())

        if self.key_model == 'key_generator':
            assert self.gen_lambda != 0.0
            merged_loss = self.gen_lambda * gen_loss
            stats = self._stats(batch_size=batch.batch_size,
                                gen_loss_data=gen_loss_data.data.clone(),
                                gen_scores_data=gen_scores_data.data.clone(),
                                gen_target_data=gen_target_data.data.clone())

        if self.key_model == 'key_end2end':
            # merged_loss = self.sel_lambda * sel_loss + self.gen_lambda * gen_loss + self.incons_lambda * incons_loss
            merged_loss = self.gen_lambda * gen_loss

            rnd_num = random.random()
            if self.sel_lambda != 0.0 and rnd_num < self.sel_train_ratio:
                merged_loss = self.sel_lambda * sel_loss + merged_loss
            if self.incons_lambda != 0.0:
                merged_loss = self.incons_lambda * incons_loss + merged_loss
            stats = self._stats(batch.batch_size,
                                sel_loss.data.clone(), sel_probs.data.clone(),
                                gt_indicators.data.clone(), sel_mask.data.clone(),
                                gen_loss_data.data.clone(), gen_scores_data.data.clone(),
                                gen_target_data.data.clone(), incons_loss.data.clone(),
                                merged_loss.data.clone())

        return merged_loss, stats

    def monolithic_compute_loss(self, batch, sel_outputs, sel_probs, dec_outputs, attns):
        """
        Compute the forward loss for the batch.
        Args:
          batch (batch): batch of labeled examples
          sel_output (:obj:`FloatTensor`):
              logits output of the selector `[src_len x batch ]`
          sel_probs (:obj:`FloatTensor`):
              importance socres output of the selector `[src_len x batch ]`
          dec_outputs (:obj:`FloatTensor`):
              output of the decoder `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions `[tgt_len x batch x src_len]`

        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        shard_state = self._make_shard_state(batch, sel_outputs, sel_probs, dec_outputs, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, sel_outputs, sel_probs, dec_outputs, attns, normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch): batch of labeled examples
          sel_output (:obj:`FloatTensor`):
              logits output of the selector `[src_len x batch ]`
          sel_probs (:obj:`FloatTensor`):
              importance socres output of the selector `[src_len x batch ]`
          dec_outputs (:obj:`FloatTensor`):
              output of the decoder `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions `[tgt_len x batch x src_len]`
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        # sharding is not convenient to mult-task learning, we avoid it.
        batch_stats = onmt.utils.E2EStatistics()
        shard_state = self._make_shard_state(batch, sel_outputs, sel_probs, dec_outputs, attns)
        loss, stats = self._compute_loss(batch, **shard_state)
        loss.div(float(normalization)).backward()
        batch_stats.update(stats)

        return batch_stats

    def _stats(self, batch_size, sel_loss_data=None, sel_probs=None, gt_indicators=None, sel_mask=None,
               gen_loss_data=None, gen_scores_data=None, gen_target_data=None, incons_loss=None, merged_loss=None):
        """
        Args:
            batch_size (:obj:`int`): the batch size
            sel_loss_data (:obj:`FloatTensor`): the computed extraction loss
            sel_probs (:obj:`FloatTensor`): the predicted importance socres
            gt_indicators (:obj:`LongTensor`): the ground-truth keyword indicators
            sel_mask (:obj:`ByteTensor`): the selection mask
            gen_loss_data (:obj:`FloatTensor`): the computed generation loss
            gen_scores_data (:obj:`FloatTensor`): the predicted probability distribution from the generator
            gen_target_data (:obj:`LongTensor`): the target data
            incons_loss (:obj:`FloatTensor`): the computed inconsistency loss
            merged_loss (:obj:`FloatTensor`): the total loss

        Returns:
            :obj:`onmt.utils.E2EStatistics` : statistics for this batch.
        """
        # ================ calculate selector statistics ====================
        # selector predictions
        if sel_loss_data is not None:
            seq_len, _ = sel_probs.size()
            # predictions
            pred = sel_probs.mul(sel_mask.float())
            # pred.gt(self.sel_threshold)
            if self.sel_report_topk <= seq_len:
                pred_topk, pred_topk_idx = pred.topk(k=self.sel_report_topk, dim=0) # [sel_report_topk, batch_size]
                tgt_at_pred_idx = gt_indicators.gather(dim=0, index=pred_topk_idx)  # [sel_report_topk, batch_size]
                mask_at_pred_idx = sel_mask.gather(dim=0, index=pred_topk_idx)
            else:
                tgt_at_pred_idx = gt_indicators # [seq_len, batch_size]
                mask_at_pred_idx = sel_mask

            sel_pred_select_num = mask_at_pred_idx.sum().item()
            # True Positive
            TP = tgt_at_pred_idx.sum().item()
            # targets
            sel_gt_select_num = gt_indicators.sum().item()
        else:
            sel_loss_data = 0.0
            sel_pred_select_num = 0
            sel_gt_select_num = 0
            TP = 0

        # ================ calculate s2s generator statistics ====================
        # stats = self._stats(loss_data, scores.data, target.view(-1).data)
        if gen_loss_data is not None:
            gen_pred = gen_scores_data.max(1)[1]
            # gen_target_data = gen_target_data.view(-1)
            gen_non_padding = gen_target_data.ne(self.tgt_padding_idx)
            gen_num_correct = gen_pred.eq(gen_target_data)\
                .masked_select(gen_non_padding) \
                .sum() \
                .item()
            gen_num_non_padding = gen_non_padding.sum().item()
        else:
            gen_loss_data = 0.0
            gen_num_correct = 0
            gen_num_non_padding = 0

        # ================ calculate s2s inconsistency loss statistics ====================
        if incons_loss is None:
            incons_loss = 0.0
        if merged_loss is None:
            merged_loss = 0.0

        # (batch_size=0,
        #  sel_loss=0.0, sel_true_positive=0, sel_true_negative=0, sel_pred_select_num=0, sel_gt_select_num=0, sel_total_words_num=0,
        #  gen_loss=0.0, gen_n_words=0, gen_n_correct=0)
        return onmt.utils.E2EStatistics(batch_size,
                                        sel_loss=sel_loss_data,
                                        sel_true_positive=TP,
                                        sel_pred_select_num=sel_pred_select_num,
                                        sel_gt_select_num=sel_gt_select_num,
                                        gen_loss=gen_loss_data, gen_n_words=gen_num_non_padding,
                                        gen_n_correct=gen_num_correct, incons_loss=incons_loss, merged_loss=merged_loss)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert 0.0 <= label_smoothing <= 1.0
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False) # sum all the element-wise loss together
        self.confidence = 1.0 - label_smoothing

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target):
        scores = self.generator(self._bottle(output))

        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze(-1)
            # log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.size(0) > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_
        loss = self.criterion(scores, gtruth)
        if self.confidence < 1:
            # Default: report smoothed ppl.
            # loss_data = -log_likelihood.sum(0)
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
