""" Report manager utility """
from __future__ import print_function
import time
from datetime import datetime

import onmt

from onmt.utils.logging import logger


def build_report_manager(opt):
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(opt.tensorboard_log_dir + '/' + opt.tensorboard_comment
                               + datetime.now().strftime("/%b-%d_%H-%M-%S"),
                               comment=opt.tensorboard_comment)
    else:
        writer = None
    # if opt.key_model == 'key_selector':
    #     report_mgr = SelectorReportMgr(opt.report_every, start_time=-1,
    #                                    tensorboard_writer=writer)
    # elif opt.key_model == 'key_end2end':
    #     report_mgr = E2EReportMgr(opt.report_every, start_time=-1,
    #                               tensorboard_writer=writer)
    # else:
    #     report_mgr = ReportMgr(opt.report_every, start_time=-1,
    #                            tensorboard_writer=writer)
    report_mgr = MyReportMgr(opt.key_model, opt.report_every, start_time=-1,
                              tensorboard_writer=writer)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if multigpu:
            report_stats = onmt.utils.Statistics.all_gather_stats(report_stats)

        if step % self.report_every == 0:
            self._report_training(
                step, num_steps, learning_rate, report_stats)
            self.progress_step += 1
            return onmt.utils.Statistics()
        else:
            return report_stats

    def _report_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(
            lr, step, train_stats=train_stats, valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()


class MyReportMgr(ReportMgrBase):
    def __init__(self, key_model, report_every, start_time=-1., tensorboard_writer=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(MyReportMgr, self).__init__(report_every, start_time)
        self.key_model = key_model
        self.tensorboard_writer = tensorboard_writer

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(
                prefix, self.tensorboard_writer, learning_rate, step)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(E2EStatistics): old Statistics instance.
        Returns:
            report_stats(E2EStatistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if multigpu:
            raise NotImplementedError
            # report_stats = onmt.utils.SelectorStatistics.all_gather_stats(report_stats)

        if step % self.report_every == 0:
            self._report_training(
                step, num_steps, learning_rate, report_stats)
            self.progress_step += 1
            return onmt.utils.E2EStatistics()
        else:
            return report_stats

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps,
                            learning_rate, self.start_time)

        # Log the progress using the number of batches on the x-axis.
        # self.maybe_log_tensorboard(report_stats,
        #                            "progress",
        #                            learning_rate,
        #                            self.progress_step)
        report_stats = onmt.utils.E2EStatistics()

        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            if self.key_model == 'key_selector' or self.key_model == 'key_end2end':
                self.log(' ')
                self.log('Sel Train sel_ave_loss: %g' % train_stats.sel_ave_loss())
                # self.log('Sel Train sel_acc: %g' % train_stats.sel_accuracy())
                self.log('Sel Train sel_p: %g' % train_stats.sel_precision())
                self.log('Sel Train sel_r: %g' % train_stats.sel_recall())
                self.log('Sel Train sel_F1: %g' % train_stats.sel_f1_score())
                # self.log('Sel Train sel_pred_sel_ratio: %g' % train_stats.sel_pred_select_ratio())
                # self.log('Sel Train sel_gt_sel_ratio: %g' % train_stats.sel_gt_select_ratio())

            if self.key_model == 'key_generator' or self.key_model == 'key_end2end':
                self.log(' ')
                self.log('Gen Train gen_ppl: %g' % train_stats.gen_ppl())
                self.log('Gen Train gen_acc: %g' % train_stats.gen_accuracy())
                self.log('Gen Train gen_xent: %g' % train_stats.gen_xent())
                self.log('Gen Train gen_ave_loss: %g' % train_stats.gen_ave_loss())

            if self.key_model == 'key_end2end':
                self.log(' ')
                self.log('Inc Train incons_ave_loss: %g' % train_stats.incons_ave_loss())
                self.log('Mrg Train merged_ave_loss: %g' % train_stats.merged_ave_loss())
            self.maybe_log_tensorboard(train_stats, "train", lr, step)

            self.log('==========================================================================')

        if valid_stats is not None:
            if self.key_model == 'key_selector' or self.key_model == 'key_end2end':
                # self.log(' ')
                self.log('Sel Validation sel_ave_loss: %g' % valid_stats.sel_ave_loss())
                # self.log('Sel Validation sel_acc: %g' % valid_stats.sel_accuracy())
                self.log('Sel Validation sel_p: %g' % valid_stats.sel_precision())
                self.log('Sel Validation sel_r: %g' % valid_stats.sel_recall())
                self.log('Sel Validation sel_F1: %g' % valid_stats.sel_f1_score())
                # self.log('Sel Validation sel_pred_select_ratio: %g' % valid_stats.sel_pred_select_ratio())
                # self.log('Sel Validation sel_gt_select_ratio: %g' % valid_stats.sel_gt_select_ratio())

            if self.key_model == 'key_generator' or self.key_model == 'key_end2end':
                self.log(' ')
                self.log('Gen Validation gen_ave_loss: %g' % valid_stats.gen_ave_loss())
                self.log('Gen Validation gen_acc: %g' % valid_stats.gen_accuracy())
                self.log('Gen Validation gen_ppl: %g' % valid_stats.gen_ppl())
                self.log('Gen Validation gen_xent: %g' % valid_stats.gen_xent())

            if self.key_model == 'key_end2end':
                self.log(' ')
                self.log('Inc Validation incons_ave_loss: %g' % valid_stats.incons_ave_loss())
                self.log('Mrg Validation merged_ave_loss: %g' % valid_stats.merged_ave_loss())

            self.log(' ')
            self.maybe_log_tensorboard(valid_stats, "valid", lr, step)


# class E2EReportMgr(ReportMgrBase):
#     def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
#         """
#         A report manager that writes statistics on standard output as well as
#         (optionally) TensorBoard
#
#         Args:
#             report_every(int): Report status every this many sentences
#             tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
#                 The TensorBoard Summary writer to use or None
#         """
#         super(E2EReportMgr, self).__init__(report_every, start_time)
#         self.tensorboard_writer = tensorboard_writer
#
#     def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
#         if self.tensorboard_writer is not None:
#             stats.log_tensorboard(
#                 prefix, self.tensorboard_writer, learning_rate, step)
#
#     def report_training(self, step, num_steps, learning_rate,
#                         report_stats, multigpu=False):
#         """
#         This is the user-defined batch-level traing progress
#         report function.
#
#         Args:
#             step(int): current step count.
#             num_steps(int): total number of batches.
#             learning_rate(float): current learning rate.
#             report_stats(E2EStatistics): old Statistics instance.
#         Returns:
#             report_stats(E2EStatistics): updated Statistics instance.
#         """
#         if self.start_time < 0:
#             raise ValueError("""ReportMgr needs to be started
#                                 (set 'start_time' or use 'start()'""")
#
#         if multigpu:
#             report_stats = onmt.utils.SelectorStatistics.all_gather_stats(report_stats)
#
#         if step % self.report_every == 0:
#             self._report_training(
#                 step, num_steps, learning_rate, report_stats)
#             self.progress_step += 1
#             return onmt.utils.E2EStatistics()
#         else:
#             return report_stats
#
#     def _report_training(self, step, num_steps, learning_rate,
#                          report_stats):
#         """
#         See base class method `ReportMgrBase.report_training`.
#         """
#         report_stats.output(step, num_steps,
#                             learning_rate, self.start_time)
#
#         # Log the progress using the number of batches on the x-axis.
#         self.maybe_log_tensorboard(report_stats,
#                                    "progress",
#                                    learning_rate,
#                                    self.progress_step)
#         report_stats = onmt.utils.E2EStatistics()
#
#         return report_stats
#
#     def _report_step(self, lr, step, train_stats=None, valid_stats=None):
#         """
#         See base class method `ReportMgrBase.report_step`.
#         """
#         if train_stats is not None:
#             self.log(' ')
#             self.log('Sel Train sel_ave_loss: %g' % train_stats.sel_ave_loss())
#             self.log('Sel Train sel_acc: %g' % train_stats.sel_accuracy())
#             self.log('Sel Train sel_p: %g' % train_stats.sel_precision())
#             self.log('Sel Train sel_r: %g' % train_stats.sel_recall())
#             self.log('Sel Train sel_F1: %g' % train_stats.sel_f1_score())
#             self.log('Sel Train sel_pred_sel_ratio: %g' % train_stats.sel_pred_select_ratio())
#             self.log('Sel Train sel_gt_sel_ratio: %g' % train_stats.sel_gt_select_ratio())
#
#             self.log('Gen Train gen_ave_loss: %g' % train_stats.gen_ave_loss())
#             self.log('Gen Train gen_acc: %g' % train_stats.gen_accuracy())
#             self.log('Gen Train gen_ppl: %g' % train_stats.gen_ppl())
#             self.log('Gen Train gen_xent: %g' % train_stats.gen_xent())
#
#             self.log('Inc Train incons_ave_loss: %g' % train_stats.incons_ave_loss())
#             self.log('Mrg Train merged_ave_loss: %g' % train_stats.merged_ave_loss())
#             self.maybe_log_tensorboard(train_stats,
#                                        "train",
#                                        lr,
#                                        step)
#
#         if valid_stats is not None:
#             self.log(' ')
#             self.log('Sel Validation sel_ave_loss: %g' % valid_stats.sel_ave_loss())
#             self.log('Sel Validation sel_acc: %g' % valid_stats.sel_accuracy())
#             self.log('Sel Validation sel_p: %g' % valid_stats.sel_precision())
#             self.log('Sel Validation sel_r: %g' % valid_stats.sel_recall())
#             self.log('Sel Validation sel_F1: %g' % valid_stats.sel_f1_score())
#             self.log('Sel Validation sel_pred_select_ratio: %g' % valid_stats.sel_pred_select_ratio())
#             self.log('Sel Validation sel_gt_select_ratio: %g' % valid_stats.sel_gt_select_ratio())
#
#             self.log('Gen Validation gen_ave_loss: %g' % valid_stats.gen_ave_loss())
#             self.log('Gen Validation gen_acc: %g' % valid_stats.gen_accuracy())
#             self.log('Gen Validation gen_ppl: %g' % valid_stats.gen_ppl())
#             self.log('Gen Validation gen_xent: %g' % valid_stats.gen_xent())
#
#             self.log('Inc Validation incons_ave_loss: %g' % valid_stats.incons_ave_loss())
#             self.log('Mrg Validation merged_ave_loss: %g' % valid_stats.merged_ave_loss())
#
#             self.log(' ')
#
#             self.maybe_log_tensorboard(valid_stats,
#                                        "valid",
#                                        lr,
#                                        step)


# class SelectorReportMgr(ReportMgrBase):
#     def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
#         """
#         A report manager that writes statistics on standard output as well as
#         (optionally) TensorBoard
#
#         Args:
#             report_every(int): Report status every this many sentences
#             tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
#                 The TensorBoard Summary writer to use or None
#         """
#         super(SelectorReportMgr, self).__init__(report_every, start_time)
#         self.tensorboard_writer = tensorboard_writer
#
#     def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
#         if self.tensorboard_writer is not None:
#             stats.log_tensorboard(
#                 prefix, self.tensorboard_writer, learning_rate, step)
#
#     def report_training(self, step, num_steps, learning_rate,
#                         report_stats, multigpu=False):
#         """
#         This is the user-defined batch-level traing progress
#         report function.
#
#         Args:
#             step(int): current step count.
#             num_steps(int): total number of batches.
#             learning_rate(float): current learning rate.
#             report_stats(SelectorStatistics): old Statistics instance.
#         Returns:
#             report_stats(SelectorStatistics): updated Statistics instance.
#         """
#         if self.start_time < 0:
#             raise ValueError("""ReportMgr needs to be started
#                                 (set 'start_time' or use 'start()'""")
#
#         if multigpu:
#             report_stats = onmt.utils.SelectorStatistics.all_gather_stats(report_stats)
#
#         if step % self.report_every == 0:
#             self._report_training(
#                 step, num_steps, learning_rate, report_stats)
#             self.progress_step += 1
#             return onmt.utils.SelectorStatistics()
#         else:
#             return report_stats
#
#     def _report_training(self, step, num_steps, learning_rate,
#                          report_stats):
#         """
#         See base class method `ReportMgrBase.report_training`.
#         """
#         report_stats.output(step, num_steps,
#                             learning_rate, self.start_time)
#
#         # Log the progress using the number of batches on the x-axis.
#         self.maybe_log_tensorboard(report_stats,
#                                    "progress",
#                                    learning_rate,
#                                    self.progress_step)
#         report_stats = onmt.utils.SelectorStatistics()
#
#         return report_stats
#
#     def _report_step(self, lr, step, train_stats=None, valid_stats=None):
#         """
#         See base class method `ReportMgrBase.report_step`.
#         """
#         if train_stats is not None:
#             self.log(' ')
#             self.log('Selector Train ave_loss: %g' % train_stats.ave_loss())
#             # self.log('Selector Train accuracy: %g' % train_stats.accuracy())
#             self.log('Selector Train precision: %g' % train_stats.precision())
#             self.log('Selector Train recall: %g' % train_stats.recall())
#             self.log('Selector Train F1 score: %g' % train_stats.f1_score())
#             # self.log('Selector Train pred_select_ratio: %g' % train_stats.pred_select_ratio())
#             # self.log('Selector Train gt_select_ratio: %g' % train_stats.gt_select_ratio())
#
#             self.maybe_log_tensorboard(train_stats,
#                                        "train",
#                                        lr,
#                                        step)
#
#         if valid_stats is not None:
#             self.log(' ')
#             self.log('Selector Validation ave_loss: %g' % valid_stats.ave_loss())
#             # self.log('Selector Validation accuracy: %g' % valid_stats.accuracy())
#             self.log('Selector Validation precision: %g' % valid_stats.precision())
#             self.log('Selector Validation recall: %g' % valid_stats.recall())
#             self.log('Selector Validation F1 score: %g' % valid_stats.f1_score())
#             # self.log('Selector Validation pred_select_ratio: %g' % valid_stats.pred_select_ratio())
#             # self.log('Selector Validation gt_select_ratio: %g' % valid_stats.gt_select_ratio())
#             self.log(' ')
#
#             self.maybe_log_tensorboard(valid_stats,
#                                        "valid",
#                                        lr,
#                                        step)
#
#
# class ReportMgr(ReportMgrBase):
#     def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
#         """
#         A report manager that writes statistics on standard output as well as
#         (optionally) TensorBoard
#
#         Args:
#             report_every(int): Report status every this many sentences
#             tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
#                 The TensorBoard Summary writer to use or None
#         """
#         super(ReportMgr, self).__init__(report_every, start_time)
#         self.tensorboard_writer = tensorboard_writer
#
#     def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
#         if self.tensorboard_writer is not None:
#             stats.log_tensorboard(
#                 prefix, self.tensorboard_writer, learning_rate, step)
#
#     def _report_training(self, step, num_steps, learning_rate,
#                          report_stats):
#         """
#         See base class method `ReportMgrBase.report_training`.
#         """
#         report_stats.output(step, num_steps,
#                             learning_rate, self.start_time)
#
#         # Log the progress using the number of batches on the x-axis.
#         self.maybe_log_tensorboard(report_stats,
#                                    "progress",
#                                    learning_rate,
#                                    self.progress_step)
#         report_stats = onmt.utils.Statistics()
#
#         return report_stats
#
#     def _report_step(self, lr, step, train_stats=None, valid_stats=None):
#         """
#         See base class method `ReportMgrBase.report_step`.
#         """
#         if train_stats is not None:
#             self.log('Train perplexity: %g' % train_stats.ppl())
#             self.log('Train accuracy: %g' % train_stats.accuracy())
#
#             self.maybe_log_tensorboard(train_stats,
#                                        "train",
#                                        lr,
#                                        step)
#
#         if valid_stats is not None:
#             self.log('Validation perplexity: %g' % valid_stats.ppl())
#             self.log('Validation accuracy: %g' % valid_stats.accuracy())
#
#             self.maybe_log_tensorboard(valid_stats,
#                                        "valid",
#                                        lr,
#                                        step)