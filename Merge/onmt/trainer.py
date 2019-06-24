"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

from __future__ import division

import onmt.inputters as inputters
import onmt.utils

import random

from onmt.utils.logging import logger


def build_trainer(opt, model, fields, optim, data_type, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    # changed for KE_KG
    # if opt.key_model == "key_selector":
    #     train_loss = onmt.utils.loss.my_build_loss_compute(
    #         model, fields["key_indicators"].vocab, None, opt)
    #     valid_loss = onmt.utils.loss.my_build_loss_compute(
    #         model, fields["key_indicators"].vocab, None, opt, train=False)
    # elif opt.key_model == "key_end2end":
    #     train_loss = onmt.utils.loss.my_build_loss_compute(
    #         model, fields["key_indicators"].vocab, fields["tgt"].vocab, opt)
    #     valid_loss = onmt.utils.loss.my_build_loss_compute(
    #         model, fields["key_indicators"].vocab, fields["tgt"].vocab, opt, train=False)
    # else:
    #     train_loss = onmt.utils.loss.my_build_loss_compute(
    #         model, None, fields["tgt"].vocab, opt)
    #     valid_loss = onmt.utils.loss.my_build_loss_compute(
    #         model, None, fields["tgt"].vocab, opt, train=False)

    train_loss = onmt.utils.loss.my_build_loss_compute(
        model, fields["posi_neg_indicators"].vocab, None, opt)
    valid_loss = onmt.utils.loss.my_build_loss_compute(
        model, fields["posi_neg_indicators"].vocab, None, opt, train=False)

    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    n_gpu = len(opt.gpuid)
    gpu_rank = opt.gpu_rank
    gpu_verbose_level = opt.gpu_verbose_level

    report_manager = onmt.utils.build_report_manager(opt)
    # add for KE_KG
    key_model = opt.key_model
    trunc_size = 0
    shard_size = None
    trainer = onmt.Trainer(key_model, model, train_loss, valid_loss, optim, trunc_size,
                           shard_size, data_type, norm_method,
                           grad_accum_count, n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, key_model, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=None, data_type='text',
                 norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None, stop_nodecrease_steps=3):
        # changed for KE_KG, sharding is not convenient to mult-task learning, we avoid it.
        # assert not shard_size
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        # add for KE_KG
        self.key_model = key_model
        self.cur_valid_ppl = None
        self.stop_nodecrease_steps = stop_nodecrease_steps
        self.nodecrease_steps = 0

        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter_fct, valid_iter_fct, train_steps, valid_steps):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info(' ')
        logger.info('Start training...')

        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        # if self.key_model == 'key_selector':
        #     total_stats = onmt.utils.SelectorStatistics()
        #     report_stats = onmt.utils.SelectorStatistics()
        # elif self.key_model == "key_end2end":
        #     total_stats = onmt.utils.E2EStatistics()
        #     report_stats = onmt.utils.E2EStatistics()
        # else:
        #     total_stats = onmt.utils.Statistics()
        #     report_stats = onmt.utils.Statistics()

        total_stats = onmt.utils.ReRankerStatistics()
        report_stats = onmt.utils.ReRankerStatistics()

        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    if self.gpu_verbose_level > 1:
                        logger.info("GpuRank %d: index: %d accum: %d"
                                    % (self.gpu_rank, i, accum))
                    cur_dataset = train_iter.get_cur_dataset()
                    self.train_loss.cur_dataset = cur_dataset

                    true_batchs.append(batch)

                    normalization += batch.batch_size

                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.gpu_verbose_level > 0:
                            logger.info("GpuRank %d: reduce_counter: %d \
                                        n_minibatch %d"
                                        % (self.gpu_rank, reduce_counter,
                                           len(true_batchs)))
                        if self.n_gpu > 1:
                            normalization = sum(onmt.utils.distributed
                                                .all_gather_list
                                                (normalization))
                        # if self.key_model == "key_selector":
                        #     self._selector_gradient_accumulation(
                        #         true_batchs, normalization, total_stats,
                        #         report_stats)
                        # elif self.key_model == "key_end2end":
                        #     self._end2end_gradient_accumulation(
                        #         true_batchs, normalization, total_stats,
                        #         report_stats)
                        # else:
                        #     self._generator_gradient_accumulation(
                        #         true_batchs, normalization, total_stats,
                        #         report_stats)
                        self._reranker_gradient_accumulation(
                            true_batchs, normalization, total_stats, report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % valid_steps == 0):
                            if self.gpu_verbose_level > 0:
                                logger.info('GpuRank %d: validate step %d'
                                            % (self.gpu_rank, step))
                            valid_iter = valid_iter_fct()
                            valid_stats = self.validate(valid_iter)
                            if self.gpu_verbose_level > 0:
                                logger.info('GpuRank %d: gather valid stat \
                                            step %d' % (self.gpu_rank, step))
                            valid_stats = self._maybe_gather_stats(valid_stats)
                            if self.gpu_verbose_level > 0:
                                logger.info('GpuRank %d: report stat step %d'
                                            % (self.gpu_rank, step))
                            # changed for KE_KG
                            self._report_step(self.optim.learning_rate,
                                              step, train_stats=total_stats, valid_stats=valid_stats)

                            assert valid_steps == self.model_saver.save_checkpoint_steps
                            self._maybe_save(step, valid_stats)

                            # if self.key_model == 'key_selector':
                            #     new_valid_ppl = valid_stats.sel_ave_loss()
                            #     # total_stats = onmt.utils.SelectorStatistics()
                            #     total_stats = onmt.utils.E2EStatistics()
                            # elif self.key_model == 'key_end2end':
                            #     new_valid_ppl = valid_stats.gen_ppl()
                            #     total_stats = onmt.utils.E2EStatistics()
                            # else:
                            #     new_valid_ppl = valid_stats.gen_ppl()
                            #     # total_stats = onmt.utils.Statistics()
                            #     total_stats = onmt.utils.E2EStatistics()

                            new_valid_ppl = valid_stats.ave_loss()
                            total_stats = onmt.utils.ReRankerStatistics()

                            if self.cur_valid_ppl is not None:
                                if self.cur_valid_ppl < new_valid_ppl:
                                    self.nodecrease_steps += 1
                                else:
                                    self.nodecrease_steps = 0
                            self.cur_valid_ppl = new_valid_ppl
                        # changed for KE_KG
                        # if self.gpu_rank == 0:
                        #     self._maybe_save(step)
                        if self.nodecrease_steps >= self.stop_nodecrease_steps:
                            break
                        step += 1
                        if step > train_steps:
                            break

            if self.nodecrease_steps >= self.stop_nodecrease_steps:
                break

            if self.gpu_verbose_level > 0:
                logger.info('GpuRank %d: we completed an epoch \
                            at step %d' % (self.gpu_rank, step))
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        # changed for KE_KG
        # if self.key_model == "key_selector":
        #     stats = onmt.utils.SelectorStatistics()
        # elif self.key_model == "key_end2end":
        #     stats = onmt.utils.E2EStatistics()
        # else:
        #     stats = onmt.utils.Statistics()

        stats = onmt.utils.ReRankerStatistics()

        print_examples = 3
        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = inputters.make_features(batch, 'src', self.data_type)
            _, src_lengths = batch.src

            tgt = inputters.make_features(batch, 'tgt', self.data_type)
            _, tgt_lengths = batch.tgt

            # F-prop through the model.
            # changed KE_KG
            assert self.key_model == "reranker"
            logits, probs = self.model(src, tgt, src_lengths, tgt_lengths)
            # Compute loss
            batch_stats = self.valid_loss.monolithic_compute_loss(batch, logits, probs)
                # if print_examples != 0:
                #     random.seed(3435)
                #     for ex_idx in random.sample(range(batch.batch_size), print_examples):
                #         _, pred_topk_idx = probs[:, ex_idx].topk(k=self.valid_loss.sel_report_topk, dim=0)
                #         ex_src = batch.src[0][:, ex_idx]
                #         ex_src = [cur_dataset.fields['src'].vocab.itos[word_idx] for word_idx in ex_src]
                #         ex_ind = batch.key_indicators[0][:, ex_idx]
                #         ex_ind = [cur_dataset.fields['key_indicators'].vocab.itos[word_idx] for word_idx in ex_ind]
                #         pred_ind_positions = pred_topk_idx
                #         ex_pred_ind = ['p_1' if posi_idx in pred_ind_positions else 'p_0' for posi_idx in range(len(ex_src))]
                #         ex_pred_scores = probs[:, ex_idx]
                #         ex_pred_scores = ['{:.3f}'.format(prob.data.item()) for prob in ex_pred_scores]
                #         out_ex = ['|'.join([src_word, tgt_ind, pred_ind, pred_score])
                #                   for src_word, tgt_ind, pred_ind, pred_score in zip(ex_src, ex_ind, ex_pred_ind, ex_pred_scores)]
                #         gt_key_words = [word for word in out_ex if 'I' in word]
                #         pred_key_words = [word for word in out_ex if 'p_1' in word]
                #         out_ex = ' '.join(out_ex)
                #         logger.info('')
                #         logger.info('Example {}:'.format(ex_idx) + out_ex)
                #         logger.info('Example gt: ' + ' '.join(gt_key_words))
                #         logger.info('Example pred: ' + ' '.join(pred_key_words))
                # print_examples = 0

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = inputters.make_features(batch, 'src', self.data_type) # [src_len, batch_size, num_features]
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum().item()
            else:
                src_lengths = None

            tgt_outer = inputters.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                outputs, attns, dec_state = \
                    self.model(src, tgt, src_lengths, dec_state)

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                    batch, outputs, attns, j,
                    trunc_size, self.shard_size, normalization)
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        # 3.bis Multi GPU gradient gather
        if self.n_gpu > 1:
            grads = [p.grad.data for p in self.model.parameters()
                     if p.requires_grad
                     and p.grad is not None]
            onmt.utils.distributed.all_reduce_and_rescale_tensors(
                grads, float(1))

        # 4. Update the parameters and statistics.
        # changed for KE_KG
        self.optim.step(self.cur_valid_ppl)
        report_stats.lr_rate = self.optim.learning_rate
        report_stats.total_norm = self.optim.total_norm

    def _generator_gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:

            dec_state = None
            src = inputters.make_features(batch, 'src', self.data_type) # [src_len, batch_size, num_features]
            if self.data_type == 'text':
                _, src_lengths = batch.src
                # report_stats.n_src_words += src_lengths.sum().item()
            else:
                src_lengths = None

            tgt_outer = inputters.make_features(batch, 'tgt')

            # 1. F-prop all but generator.
            if self.grad_accum_count == 1:
                self.model.zero_grad()
            outputs, attns, dec_state = \
                self.model(src, tgt_outer, src_lengths, dec_state)

            # 2. Compute loss.
            # sharded_compute_loss(self, batch, sel_outputs, sel_probs, dec_outputs, attns, normalization)
            batch_stats = self.train_loss.sharded_compute_loss(batch=batch,
                                                               sel_outputs=None,
                                                               sel_probs=None,
                                                               dec_outputs=outputs,
                                                               attns=attns,
                                                               normalization=normalization)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

        # 3. Update the parameters and statistics.
        # changed for KE_KG
        self.optim.step(self.cur_valid_ppl)
        report_stats.lr_rate = self.optim.learning_rate
        report_stats.total_norm = self.optim.total_norm

    def _end2end_gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            # 1. prepare the batch
            src = inputters.make_features(batch, 'src', self.data_type) # [src_len, batch_size, num_features]
            _, src_lengths = batch.src

            tgt = inputters.make_features(batch, 'tgt')

            # 2. F-prop all but generator.
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            dec_outputs, attns, dec_state, sel_outputs, sel_probs =\
                self.model(src, tgt, src_lengths, gt_probs=batch.key_indicators[0])

            # 3. Compute loss
            # (self, batch, sel_outputs, sel_probs, dec_outputs, attns, normalization)
            batch_stats = self.train_loss.sharded_compute_loss(
                batch, sel_outputs, sel_probs, dec_outputs, attns, normalization)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

        # assert self.n_gpu == 1
        # 4. Update the parameters and statistics.
        self.optim.step(self.cur_valid_ppl)
        report_stats.lr_rate = self.optim.learning_rate
        report_stats.total_norm = self.optim.total_norm

    def _selector_gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            src = inputters.make_features(batch, 'src', self.data_type) # [src_len, batch_size, num_features]

            _, src_lengths = batch.src

            # 1. F-prop all.
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            logits, probs = self.model(src, src_lengths)

            # 2. Compute loss in shards for memory efficiency.
            # sharded_compute_loss(self, batch, sel_outputs, sel_probs, dec_outputs, attns, normalization)
            batch_stats = self.train_loss.sharded_compute_loss(
                batch, logits, probs, None, None, normalization)
            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

        assert self.n_gpu == 1
        # 2. Update the parameters and statistics.
        self.optim.step(self.cur_valid_ppl)
        report_stats.lr_rate = self.optim.learning_rate
        report_stats.total_norm = self.optim.total_norm

    def _reranker_gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        assert len(true_batchs) == 1
        # for batch in true_batchs:
        batch = true_batchs[0]
        src = inputters.make_features(batch, 'src', self.data_type) # [src_len, batch_size, num_features]
        _, src_lengths = batch.src

        tgt = inputters.make_features(batch, 'tgt', self.data_type)  # [tgt_len, batch_size, num_features]
        _, tgt_lengths = batch.tgt

        # 1. F-prop all.
        if self.grad_accum_count == 1:
            self.model.zero_grad()

        logits, probs = self.model(src, tgt, src_lengths, tgt_lengths)

        batch_stats = self.train_loss.sharded_compute_loss(
            batch, logits, probs, None, normalization)
        total_stats.update(batch_stats)
        report_stats.update(batch_stats)

        assert self.n_gpu == 1
        # 2. Update the parameters and statistics.
        self.optim.step(self.cur_valid_ppl)
        report_stats.lr_rate = self.optim.learning_rate
        report_stats.total_norm += self.optim.total_norm

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        # # changed for KE_KG
        # if self.key_model == "key_selector":
        #     if stat is not None and self.n_gpu > 1:
        #         return onmt.utils.SelectorStatistics.all_gather_stats(stat)
        # else:
        #     if stat is not None and self.n_gpu > 1:
        #         return onmt.utils.Statistics.all_gather_stats(stat)
        assert self.n_gpu == 1
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step, stats=None):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step, stats)

