"""  Attention and normalization modules  """
from onmt.modules.util_class import LayerNorm, Elementwise
from onmt.modules.gate import context_gate_factory, ContextGate
from onmt.modules.global_attention import GlobalAttention, MyGlobalAttention
from onmt.modules.conv_multi_step_attention import ConvMultiStepAttention
from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorLossCompute, CopyGeneratorCriterion
from onmt.modules.multi_headed_attn import MultiHeadedAttention
from onmt.modules.embeddings import Embeddings, PositionalEncoding
from onmt.modules.weight_norm import WeightNormConv2d
from onmt.modules.average_attn import AverageAttention

__all__ = ["LayerNorm", "Elementwise", "context_gate_factory", "ContextGate",
           "GlobalAttention", "MyGlobalAttention", "ConvMultiStepAttention", "CopyGenerator",
           "CopyGeneratorLossCompute", "CopyGeneratorCriterion", "MultiHeadedAttention", "Embeddings",
           "PositionalEncoding", "WeightNormConv2d", "AverageAttention"]
