# -*- coding: utf-8 -*-
"""
kws.tts —— 可插拔的 TTS 後端。

    piper        ONNX 本地合成，en_US-libritts_r-medium（904 語者），離線、不需 torch/CUDA。英文預設。
    piper-torch  同一個 Piper 模型，但用 PyTorch + NVIDIA GPU 批次生成（~30x 快）+ SLERP 語者內插。
                 要獨立的 venv-psg（datasets/prepare.py piper-torch）。
    edge         Microsoft edge-tts（線上），~24 英文語者 + zh-tw/zh-cn/zh-hk。中文喚醒詞用。

用法：
    from kws.tts import get_backend
    backend = get_backend("piper")          # 或讀 config.TTS_BACKEND
    jobs = backend.plan(texts, need, rng, is_positive=True, punct_list=[...])
    for job, wav, sr, err in backend.synth_all(jobs, concurrency=8):
        ...
"""

from .base import PUNCT_LATIN, PUNCT_CJK, TTSBackend


def get_backend(name, **kw):
    name = (name or "piper").lower()
    if name == "piper":
        from .piper import PiperBackend
        return PiperBackend(**kw)
    if name in ("piper-torch", "piper_torch"):
        from .piper_torch import PiperTorchBackend
        return PiperTorchBackend(**kw)
    if name == "edge":
        from .edge import EdgeBackend
        return EdgeBackend(**kw)
    raise ValueError(f"未知的 TTS 後端：{name!r}（可用：piper, piper-torch, edge）")


__all__ = ["get_backend", "TTSBackend", "PUNCT_LATIN", "PUNCT_CJK"]
