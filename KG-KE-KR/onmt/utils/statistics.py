""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys

from onmt.utils.logging import logger


class E2EStatistics(object):
    """
    Accumulator for loss statistics of the whole end2end model.
    """
    def __init__(self, batch_size=0,
                 sel_loss=0.0, sel_true_positive=0, sel_pred_select_num=0, sel_gt_select_num=0,
                 gen_loss=0.0, gen_n_words=0, gen_n_correct=0, incons_loss=0.0, merged_loss=0.0):
        # for the selector
        self.sel_loss = sel_loss
        self.sel_loss_list = [] if sel_loss == 0 else [sel_loss]
        self.sel_true_positive = sel_true_positive
        self.sel_pred_select_num = sel_pred_select_num
        self.sel_gt_select_num = sel_gt_select_num

        # for the s2s generator
        self.gen_loss = gen_loss
        self.gen_loss_list = [] if gen_loss == 0.0 else [gen_loss]
        self.gen_n_words = gen_n_words
        self.gen_n_correct = gen_n_correct

        # for inconsistency loss
        self.incons_loss = incons_loss
        self.incons_loss_list = [] if incons_loss == 0.0 else [incons_loss]

        # for merged loss
        self.merged_loss = merged_loss
        self.merged_loss_list = [] if merged_loss == 0.0 else [merged_loss]

        # for common use
        self.batch_size_list = [] if batch_size == 0 else [batch_size]
        self.lr_rate = 0.0
        self.total_norm = 0.0
        self.start_time = time.time()

    # def sel_pred_select_ratio(self):
    #     """ compute prediction selection ratio """
    #     assert self.sel_total_words_num != 0
    #     return self.sel_pred_select_num * 1.0 / self.sel_total_words_num

    # def sel_gt_select_ratio(self):
    #     """ compute prediction selection ratio """
    #     assert self.sel_total_words_num != 0
    #     return self.sel_gt_select_num * 1.0 / self.sel_total_words_num

    def sel_ave_loss(self):
        """ compute the averaged extraction loss"""
        batch_normalized_losses = [l / b for l, b in zip(self.sel_loss_list, self.batch_size_list)]
        nums = float(len(batch_normalized_losses))
        return sum(batch_normalized_losses) / nums if nums != 0.0 else 0.0

    def sel_precision(self):
        """ compute the precision """
        if self.sel_pred_select_num == 0:
            return 0.0
        else:
            return self.sel_true_positive * 1.0 / self.sel_pred_select_num

    def sel_recall(self):
        """ compute the recall """
        if self.sel_gt_select_num == 0:
            return 0.0
        else:
            return self.sel_true_positive * 1.0 / self.sel_gt_select_num

    def sel_f1_score(self):
        """ compute the F1 score """
        R = self.sel_recall()
        P = self.sel_precision()
        if P == 0.0 or R == 0.0:
            return 0.0
        else:
            return 2 * P * R / (P + R)

    # def sel_accuracy(self):
    #     """ compute the accuracy"""
    #     assert self.sel_total_words_num != 0
    #     return 100 * (self.sel_true_positive + self.sel_true_negative) * 1.0 / self.sel_total_words_num

    def gen_accuracy(self):
        """ compute accuracy """
        if self.gen_n_words == 0:
            return 0.0
        else:
            return 100 * (self.gen_n_correct / self.gen_n_words)

    def gen_xent(self):
        """ compute cross entropy """
        if self.gen_n_words == 0:
            return 0.0
        else:
            return self.gen_loss / self.gen_n_words

    def gen_ppl(self):
        """ compute perplexity """
        if self.gen_n_words == 0:
            return 0.0
        else:
            return math.exp(min(self.gen_loss / self.gen_n_words, 100))

    def gen_ave_loss(self):
        """ compute the averaged generation loss"""
        batch_normalized_losses = [l / b for l, b in zip(self.gen_loss_list, self.batch_size_list)]
        nums = float(len(batch_normalized_losses))
        return sum(batch_normalized_losses) / nums if nums != 0.0 else 0.0

    def incons_ave_loss(self):
        """ compute the averaged inconsistency loss"""
        batch_normalized_losses = [l / b for l, b in zip(self.incons_loss_list, self.batch_size_list)]
        nums = float(len(batch_normalized_losses))
        return sum(batch_normalized_losses) / nums if nums != 0.0 else 0.0

    def merged_ave_loss(self):
        """ compute the averaged merged_loss"""
        batch_normalized_losses = [l / b for l, b in zip(self.merged_loss_list, self.batch_size_list)]
        nums = float(len(batch_normalized_losses))
        return sum(batch_normalized_losses) / nums if nums != 0.0 else 0.0

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def update(self, stat):
        # for the selector
        self.sel_loss += stat.sel_loss
        self.sel_loss_list += stat.sel_loss_list
        self.sel_true_positive += stat.sel_true_positive
        # self.sel_true_negative += stat.sel_true_negative
        self.sel_pred_select_num += stat.sel_pred_select_num
        self.sel_gt_select_num += stat.sel_gt_select_num
        # self.sel_total_words_num += stat.sel_total_words_num

        # for the s2s generator
        self.gen_loss += stat.gen_loss
        self.gen_loss_list += stat.gen_loss_list
        self.gen_n_words += stat.gen_n_words
        self.gen_n_correct += stat.gen_n_correct

        # for the inconsistency loss
        self.incons_loss += stat.incons_loss
        self.incons_loss_list += stat.incons_loss_list

        # for merged loss
        self.merged_loss += stat.merged_loss
        self.merged_loss_list += stat.merged_loss_list

        # for common use
        self.batch_size_list += stat.batch_size_list

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        # t = self.elapsed_time()
        logger.info(
            ("Step %06d; sel_ave_loss: %7.5f; sel_p: %4.2f; sel_r: %4.2f; " +
             "gen_ppl: %5.2f; gen_acc: %6.2f;" +
             "incons_ave_loss: %7.5f; merged_ave_loss: %7.5f; " +
             "lr: %7.5f; tnorm: %4.2f; %6.0f sec")
            % (step, self.sel_ave_loss(),
               self.sel_precision(),
               self.sel_recall(),
               self.gen_ppl(),
               self.gen_accuracy(),
               self.incons_ave_loss(),
               self.merged_ave_loss(),
               learning_rate,
               self.total_norm,
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        # selector
        writer.add_scalar(prefix + "/sel_ave_loss", self.sel_ave_loss(), step)
        writer.add_scalar(prefix + "/sel_precision", self.sel_precision(), step)
        writer.add_scalar(prefix + "/sel_recall", self.sel_recall(), step)
        writer.add_scalar(prefix + "/sel_F1", self.sel_f1_score(), step)
        # generator
        writer.add_scalar(prefix + "/gen_ppl", self.gen_ppl(), step)
        writer.add_scalar(prefix + "/gen_acc", self.gen_accuracy(), step)
        writer.add_scalar(prefix + "/gen_ave_loss", self.gen_ave_loss(), step)
        # inconsistency
        writer.add_scalar(prefix + "/incons_ave_loss", self.incons_ave_loss(), step)
        # merged
        writer.add_scalar(prefix + "/merged_ave_loss", self.merged_ave_loss(), step)
        # public
        writer.add_scalar(prefix + "/lr_rate", learning_rate, step)
        writer.add_scalar(prefix + "/total_norm", self.total_norm, step)


    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        raise NotImplementedError

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        raise NotImplementedError


# class SelectorStatistics(object):
#     """
#     Accumulator for loss statistics of the selector.
#     """
#
#     def __init__(self, loss=0, batch_size=0, true_positive=0, pred_select_num=0, gt_select_num=0, total_words_num=0):
#         self.loss = loss
#         self.loss_list = [] if loss == 0 else [loss]
#         self.batch_size_list = [] if batch_size == 0 else [batch_size]
#         self.true_positive = true_positive
#         self.pred_select_num = pred_select_num
#         self.gt_select_num = gt_select_num
#         self.total_words_num = total_words_num
#
#         # for common use
#         self.lr_rate = 0.0
#         self.total_norm = 0.0
#         self.start_time = time.time()
#
#     # def pred_select_ratio(self):
#     #     """ compute prediction selection ratio """
#     #     assert self.total_words_num != 0
#     #     return self.pred_select_num * 1.0 / self.total_words_num
#
#     # def gt_select_ratio(self):
#     #     """ compute prediction selection ratio """
#     #     assert self.total_words_num != 0
#     #     return self.gt_select_num * 1.0 / self.total_words_num
#
#     def ave_loss(self):
#         """ compute the averaged loss"""
#         batch_normalized_losses = [l / b for l, b in zip(self.loss_list, self.batch_size_list)]
#         return sum(batch_normalized_losses) / float(len(batch_normalized_losses))
#
#     def precision(self):
#         """ compute the precision """
#         if self.pred_select_num == 0:
#             return 0.0
#         else:
#             return self.true_positive * 1.0 / self.pred_select_num
#
#     def recall(self):
#         """ compute the recall """
#         if self.gt_select_num == 0:
#             return 0.0
#         else:
#             return self.true_positive * 1.0 / self.gt_select_num
#
#     def f1_score(self):
#         """ compute the F1 score """
#         R = self.recall()
#         P = self.precision()
#         if P == 0.0 or R == 0.0:
#             return 0.0
#         else:
#             return 2 * P * R / (P + R)
#
#     # def accuracy(self):
#     #     """ compute the accuracy"""
#     #     assert self.total_words_num != 0
#     #     return 100 * (self.true_positive + self.true_negative) * 1.0 / self.total_words_num
#
#     def elapsed_time(self):
#         """ compute elapsed time """
#         return time.time() - self.start_time
#
#     def update(self, stat):
#         self.loss += stat.loss
#         self.loss_list += stat.loss_list
#         self.batch_size_list += stat.batch_size_list
#         self.true_positive += stat.true_positive
#         self.pred_select_num += stat.pred_select_num
#         self.gt_select_num += stat.gt_select_num
#         self.total_words_num += stat.total_words_num
#
#     def output(self, step, num_steps, learning_rate, start):
#         """Write out statistics to stdout.
#
#         Args:
#            step (int): current step
#            n_batch (int): total batches
#            start (int): start time of step.
#         """
#         t = self.elapsed_time()
#         logger.info(
#             ("Step %06d; ave_loss: %7.5f; sel_f1: %4.2f; sel_p: %4.2f; sel_r: %4.2f; " +
#              "lr: %7.5f; tnorm: %4.2f; %06.0f tok/s; %6.0f sec")
#             % (step, self.ave_loss(),
#                self.f1_score(),
#                self.precision(),
#                self.recall(),
#                self.lr_rate,
#                self.total_norm,
#                self.total_words_num / (t + 1e-5),
#                time.time() - start))
#         sys.stdout.flush()
#
#     def log_tensorboard(self, prefix, writer, learning_rate, step):
#         """ display statistics to tensorboard """
#         raise NotImplementedError
#
#     @staticmethod
#     def all_gather_stats(stat, max_size=4096):
#         """
#         Gather a `Statistics` object accross multiple process/nodes
#
#         Args:
#             stat(:obj:Statistics): the statistics object to gather
#                 accross all processes/nodes
#             max_size(int): max buffer size to use
#
#         Returns:
#             `Statistics`, the update stats object
#         """
#         raise NotImplementedError
#
#     @staticmethod
#     def all_gather_stats_list(stat_list, max_size=4096):
#         raise NotImplementedError


# class Statistics(object):
#     """
#     Accumulator for loss statistics.
#     Currently calculates:
#
#     * accuracy
#     * perplexity
#     * elapsed time
#     """
#
#     def __init__(self, loss=0, n_words=0, n_correct=0):
#         self.loss = loss
#         self.n_words = n_words
#         self.n_correct = n_correct
#         self.n_src_words = 0
#         self.start_time = time.time()
#         # changed for KE_KG
#         # for common use
#         self.lr_rate = 0.0
#         self.total_norm = 0.0
#
#     @staticmethod
#     def all_gather_stats(stat, max_size=4096):
#         """
#         Gather a `Statistics` object accross multiple process/nodes
#
#         Args:
#             stat(:obj:Statistics): the statistics object to gather
#                 accross all processes/nodes
#             max_size(int): max buffer size to use
#
#         Returns:
#             `Statistics`, the update stats object
#         """
#         stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
#         return stats[0]
#
#     @staticmethod
#     def all_gather_stats_list(stat_list, max_size=4096):
#         """
#         Gather a `Statistics` list accross all processes/nodes
#
#         Args:
#             stat_list(list([`Statistics`])): list of statistics objects to
#                 gather accross all processes/nodes
#             max_size(int): max buffer size to use
#
#         Returns:
#             our_stats(list([`Statistics`])): list of updated stats
#         """
#         # Get a list of world_size lists with len(stat_list) Statistics objects
#         all_stats = all_gather_list(stat_list, max_size=max_size)
#
#         our_rank = get_rank()
#         our_stats = all_stats[our_rank]
#         for other_rank, stats in enumerate(all_stats):
#             if other_rank == our_rank:
#                 continue
#             for i, stat in enumerate(stats):
#                 our_stats[i].update(stat, update_n_src_words=True)
#         return our_stats
#
#     def update(self, stat, update_n_src_words=False):
#         """
#         Update statistics by suming values with another `Statistics` object
#
#         Args:
#             stat: another statistic object
#             update_n_src_words(bool): whether to update (sum) `n_src_words`
#                 or not
#
#         """
#         self.loss += stat.loss
#         self.n_words += stat.n_words
#         self.n_correct += stat.n_correct
#
#         if update_n_src_words:
#             self.n_src_words += stat.n_src_words
#
#     def accuracy(self):
#         """ compute accuracy """
#         return 100 * (self.n_correct / self.n_words)
#
#     def xent(self):
#         """ compute cross entropy """
#         return self.loss / self.n_words
#
#     def ppl(self):
#         """ compute perplexity """
#         return math.exp(min(self.loss / self.n_words, 100))
#
#     def elapsed_time(self):
#         """ compute elapsed time """
#         return time.time() - self.start_time
#
#     def output(self, step, num_steps, learning_rate, start):
#         """Write out statistics to stdout.
#
#         Args:
#            step (int): current step
#            n_batch (int): total batches
#            start (int): start time of step.
#         """
#         t = self.elapsed_time()
#         logger.info(
#             ("Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
#              "lr: %7.5f; tnorm %4.2f; %3.0f/%3.0f tok/s; %6.0f sec")
#             % (step, num_steps,
#                self.accuracy(),
#                self.ppl(),
#                self.xent(),
#                self.lr_rate,
#                self.total_norm,
#                self.n_src_words / (t + 1e-5),
#                self.n_words / (t + 1e-5),
#                time.time() - start))
#         sys.stdout.flush()
#
#     def log_tensorboard(self, prefix, writer, learning_rate, step):
#         """ display statistics to tensorboard """
#         t = self.elapsed_time()
#         writer.add_scalar(prefix + "/xent", self.xent(), step)
#         writer.add_scalar(prefix + "/ppl", self.ppl(), step)
#         writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
#         writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
#         writer.add_scalar(prefix + "/lr", learning_rate, step)
