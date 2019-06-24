"""Module defining various utilities."""
from onmt.utils.misc import aeq, use_gpu
from onmt.utils.report_manager import MyReportMgr, build_report_manager
from onmt.utils.statistics import E2EStatistics
from onmt.utils.optimizers import build_optim, MultipleOptimizer, \
    Optimizer

__all__ = ["aeq", "use_gpu", "MyReportMgr",
           "build_report_manager", "E2EStatistics",
           "build_optim", "MultipleOptimizer", "Optimizer"]
