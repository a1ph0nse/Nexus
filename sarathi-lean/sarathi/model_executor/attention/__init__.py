from enum import Enum
from typing import Union

from sarathi.model_executor.attention.flash_attention_wrapper import (
    FlashAttentionWrapper,
)
from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
    FlashInferAttentionWrapper,
)
from sarathi.model_executor.attention.vattention_flashinfer_wrapper import (
    VAttentionFlashInferWrapper,
)
from sarathi.model_executor.attention.no_op_attention_wrapper import (
    NoOpAttentionWrapper,
)
from sarathi.model_executor.attention.vattention_flashattention_wrapper import (
    VAttentionFlashAttentionWrapper,
)
from sarathi.model_executor.attention.flashinfer_unpaged_attention_wrapper import (
    FlashinferUnpagedAttentionWrapper,
)
from sarathi.model_executor.attention.vattention_flashattention3_wrapper import (
    VAttentionFlashAttention3_Wrapper,
)
from sarathi.model_executor.attention.vattention_flashattention_pod_wrapper import (
    VAttentionFlashAttentionPODWrapper,
)
from sarathi.model_executor.attention.vattention_flashattention_streams_wrapper import (
    VAttentionFlashAttentionStreamsWrapper,
)
from sarathi.model_executor.attention.flashinfer_paged_serial_attention_wrapper import (
    FlashInferSerialAttentionWrapper,
)
from sarathi.model_executor.attention.flashattention_pod_wrapper import (
    FlashAttentionPODWrapper,
)
from sarathi.model_executor.attention.load_attention_wrapper import (
    LoadAttentionWrapper,
)
from sarathi.model_executor.attention.flashattention_streams_wrapper import (
    FlashAttentionStreamsWrapper,
)
from sarathi.model_executor.attention.vattention_flashattention_hfuse_wrapper import (
    VAttentionFlashAttentionHFUSEWrapper,
)
from sarathi.model_executor.attention.vattention_flashattention_hfuse_sparse_wrapper import (
    VAttentionFlashAttentionHFUSESparseWrapper,
)
from sarathi.model_executor.attention.torch_attention_wrapper import (
    TorchAttentionWrapper,
)
from sarathi.model_executor.attention.flash_attention_sparse_wrapper import (
    FlashAttentionSparseWrapper,
)
from sarathi.model_executor.attention.load_attention_sparse_wrapper import (
    LoadAttentionSparseWrapper,
)
from sarathi.model_executor.attention.flashattention_pod_sparse_wrapper import (
    FlashAttentionPODSparseWrapper,
)

# FA: FLASHATTENTION
# FI: FLASHINFER
class AttentionBackend(Enum):
    FA_PAGED = "FA_PAGED"
    FI_PAGED = "FI_PAGED"
    FA_VATTN = "FA_VATTN"
    FI_VATTN = "FI_VATTN"
    FA_VATTN_SYNC = "FA_VATTN_SYNC"
    FI_VATTN_SYNC = "FI_VATTN_SYNC"
    #TODO(ashish): remove the following?
    FI_UNPAGED = "FI_UNPAGED"
    NO_OP = "NO_OP"
    FA3_VATTN = "FA3_VATTN"
    FA3_VATTN_SYNC = "FA3_VATTN_SYNC"
    FA_VATTN_MEGACACHE = "FA_VATTN_MEGACACHE"
    FA_VATTN_MEGACACHE_SYNC = "FA_VATTN_MEGACACHE_SYNC"
    FA_POD = "FA_POD"
    FA_STREAMS = "FA_STREAMS"
    FI_SERIAL_PAGED = "FI_SERIAL_PAGED"
    FA_POD_MEGACACHE = "FA_POD_MEGACACHE"
    FA_STREAMS_MEGACACHE = "FA_STREAMS_MEGACACHE"

    FA_POD_PAGED = "FA_POD_PAGED"
    LA_PAGED = "LA_PAGED"
    FA_STREAMS_PAGED = "FA_STREAMS_PAGED"
    HFUSE_VATTN = "HFUSE_VATTN"
    TORCH_PAGED = "TORCH_PAGED"
    FA_SPARSE = "FA_SPARSE"
    LA_SPARSE = "LA_SPARSE"
    FA_POD_SPARSE = "FA_POD_SPARSE"
    HFUSE_SPARSE = "HFUSE_SPARSE"

    def is_attn_contiguous(attn_cfg):

        return attn_cfg.upper() in [
            AttentionBackend.FA_VATTN.value,
            AttentionBackend.FI_VATTN.value,
            AttentionBackend.FA_VATTN_SYNC.value,
            AttentionBackend.FI_VATTN_SYNC.value,
            AttentionBackend.FA3_VATTN.value,
            AttentionBackend.FA3_VATTN_SYNC.value,
            AttentionBackend.FA_VATTN_MEGACACHE.value,
            AttentionBackend.FA_VATTN_MEGACACHE_SYNC.value,
            AttentionBackend.FA_POD.value,
            AttentionBackend.FA_STREAMS.value,
            AttentionBackend.FA_POD_MEGACACHE.value,
            AttentionBackend.FA_STREAMS_MEGACACHE.value,
            AttentionBackend.HFUSE_VATTN.value,
            AttentionBackend.HFUSE_SPARSE.value,
            AttentionBackend.TORCH_PAGED.value,
        ]

    def is_vATTN(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FA_VATTN.value,
            AttentionBackend.FI_VATTN.value,
            AttentionBackend.FA_VATTN_SYNC.value,
            AttentionBackend.FI_VATTN_SYNC.value,
            AttentionBackend.FA3_VATTN.value,
            AttentionBackend.FA3_VATTN_SYNC.value,
            AttentionBackend.FA_VATTN_MEGACACHE.value,
            AttentionBackend.FA_VATTN_MEGACACHE_SYNC.value,
            AttentionBackend.FA_POD.value,
            AttentionBackend.FA_STREAMS.value,
            AttentionBackend.FA_POD_MEGACACHE.value,
            AttentionBackend.FA_STREAMS_MEGACACHE.value,
            AttentionBackend.HFUSE_VATTN.value,
            AttentionBackend.HFUSE_SPARSE.value,
        ]

    def is_vATTN_SYNC(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FA_VATTN_SYNC.value,
            AttentionBackend.FI_VATTN_SYNC.value,
            AttentionBackend.FA3_VATTN_SYNC.value,
            AttentionBackend.FA_VATTN_MEGACACHE_SYNC.value,
        ]

    def is_vLLM(attn_cfg):
        return attn_cfg.upper() in [
            AttentionBackend.FA_PAGED.value,
            AttentionBackend.FI_PAGED.value,
            AttentionBackend.FI_UNPAGED.value,
            AttentionBackend.FI_SERIAL_PAGED.value,
            AttentionBackend.FA_POD_PAGED.value,
            AttentionBackend.LA_PAGED.value,
            AttentionBackend.FA_STREAMS_PAGED.value,
            AttentionBackend.TORCH_PAGED.value,
            AttentionBackend.FA_SPARSE.value,
            AttentionBackend.LA_SPARSE.value,
            AttentionBackend.FA_POD_SPARSE.value,
        ]

ATTENTION_BACKEND = AttentionBackend.NO_OP

def get_attn_type():
    return ATTENTION_BACKEND.value


def set_attention_backend(backend: Union[str, AttentionBackend]):
    if isinstance(backend, str):
        backend = backend.upper()
        if backend not in AttentionBackend.__members__:
            raise ValueError(f"Unsupported attention backend: {backend}")
        backend = AttentionBackend[backend]
    elif not isinstance(backend, AttentionBackend):
        raise ValueError(f"Unsupported attention backend: {backend}")

    global ATTENTION_BACKEND
    ATTENTION_BACKEND = backend


def get_attention_wrapper():
    if ATTENTION_BACKEND == AttentionBackend.FI_PAGED:
        return FlashInferAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_PAGED:
        return FlashAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.NO_OP:
        return NoOpAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_VATTN:
        return VAttentionFlashAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_VATTN_SYNC:
        return VAttentionFlashAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FI_VATTN:
        return VAttentionFlashInferWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FI_VATTN_SYNC:
        return VAttentionFlashInferWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FI_UNPAGED:
        return FlashinferUnpagedAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA3_VATTN:
        return VAttentionFlashAttention3_Wrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA3_VATTN_SYNC:
        return VAttentionFlashAttention3_Wrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_VATTN_MEGACACHE:
        return VAttentionFlashAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_VATTN_MEGACACHE_SYNC:
        return VAttentionFlashAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FI_SERIAL_PAGED:
        return FlashInferSerialAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_POD:
        return VAttentionFlashAttentionPODWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_STREAMS:
        return VAttentionFlashAttentionStreamsWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_POD_MEGACACHE:
        return VAttentionFlashAttentionPODWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_STREAMS_MEGACACHE:
        return VAttentionFlashAttentionStreamsWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_POD_PAGED:
        return FlashAttentionPODWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.LA_PAGED:
        return LoadAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_STREAMS_PAGED:
        return FlashAttentionStreamsWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.HFUSE_VATTN:
        return VAttentionFlashAttentionHFUSEWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.HFUSE_SPARSE:
        return VAttentionFlashAttentionHFUSESparseWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.TORCH_PAGED:
        return TorchAttentionWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_SPARSE:
        return FlashAttentionSparseWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.LA_SPARSE:
        return LoadAttentionSparseWrapper.get_instance()
    elif ATTENTION_BACKEND == AttentionBackend.FA_POD_SPARSE:
        return FlashAttentionPODSparseWrapper.get_instance()

    raise ValueError(f"Unsupported attention backend: {ATTENTION_BACKEND}")

#TODO(ashish): these functions are also defined above?
def is_vattention_backend():
    return ATTENTION_BACKEND in [
        AttentionBackend.FA_VATTN,
        AttentionBackend.FI_VATTN,
        AttentionBackend.FA_VATTN_SYNC,
        AttentionBackend.FI_VATTN_SYNC,
        AttentionBackend.FA3_VATTN,
        AttentionBackend.FA3_VATTN_SYNC,
        AttentionBackend.FA_VATTN_MEGACACHE,
        AttentionBackend.FA_VATTN_MEGACACHE_SYNC,
        AttentionBackend.FA_POD,
        AttentionBackend.FA_STREAMS,
        AttentionBackend.FA_POD_MEGACACHE,
        AttentionBackend.FA_STREAMS_MEGACACHE,
        AttentionBackend.HFUSE_VATTN,
        AttentionBackend.HFUSE_SPARSE,
    ]

def is_vLLM_backend():
    return ATTENTION_BACKEND in [
        AttentionBackend.FA_PAGED,
        AttentionBackend.FI_PAGED,
        AttentionBackend.FI_UNPAGED,
        AttentionBackend.FI_SERIAL_PAGED,
        AttentionBackend.FA_POD_PAGED,
        AttentionBackend.LA_PAGED,
        AttentionBackend.FA_STREAMS_PAGED,
        AttentionBackend.TORCH_PAGED,
        AttentionBackend.FA_SPARSE,
        AttentionBackend.LA_SPARSE,
        AttentionBackend.FA_POD_SPARSE,
    ]

def is_attn_contiguous():
    return ATTENTION_BACKEND in [
        AttentionBackend.FA_VATTN,
        AttentionBackend.FI_VATTN,
        AttentionBackend.FA_VATTN_SYNC,
        AttentionBackend.FI_VATTN_SYNC,
        AttentionBackend.FA3_VATTN,
        AttentionBackend.FA3_VATTN_SYNC,
        AttentionBackend.FA_VATTN_MEGACACHE,
        AttentionBackend.FA_VATTN_MEGACACHE_SYNC,
        AttentionBackend.FA_POD,
        AttentionBackend.FA_STREAMS,
        AttentionBackend.FA_POD_MEGACACHE,
        AttentionBackend.FA_STREAMS_MEGACACHE,
        AttentionBackend.HFUSE_VATTN,
        AttentionBackend.HFUSE_SPARSE,
    ]