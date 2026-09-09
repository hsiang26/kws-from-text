# -*- coding: utf-8 -*-
"""
kws/tts/psg_worker.py
=====================
在 venv-psg（裝了 piper-sample-generator + torch）裡跑的工人。

**不要 import kws / tensorflow**（venv-psg 沒有）。由 kws/tts/piper_torch.py 用 subprocess 呼叫：

    venv-psg/Scripts/python.exe kws/tts/psg_worker.py <spec.json>

spec.json:
    {
      "psg_repo": ".../datasets/_models/piper_sample_generator",   # clone 的 repo（含 piper_train/）
      "model": ".../models/en_US-libritts_r-medium.pt",
      "out_dir": "<tmp>",
      "texts": [...], "file_names": [...],
      "batch_size": 32,
      "length_scales": [...], "noise_scales": [...], "noise_scale_ws": [...],
      "slerp_weights": [...], "max_speakers": null
    }
"""

import json
import os
import sys


def main():
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    # 這個檔在 kws/tts/ 底下，跑 script 時 Python 會把 kws/tts/ 塞進 sys.path[0]，
    # 導致 piper-sample-generator 的 `import piper` 撞到我們的 kws/tts/piper.py。先移除它。
    _self = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _self]
    # clone 的 repo root 要在 path 上：piper_sample_generator 和 piper_train 都在那底下
    sys.path.insert(0, spec["psg_repo"])
    from piper_sample_generator.__main__ import generate_samples

    generate_samples(
        text=spec["texts"],
        file_names=spec["file_names"],
        max_samples=len(spec["texts"]),
        output_dir=spec["out_dir"],
        model=spec["model"],
        batch_size=int(spec["batch_size"]),
        length_scales=tuple(spec["length_scales"]),
        noise_scales=tuple(spec["noise_scales"]),
        noise_scale_ws=tuple(spec["noise_scale_ws"]),
        slerp_weights=tuple(spec["slerp_weights"]),
        max_speakers=spec.get("max_speakers"),
    )


if __name__ == "__main__":
    main()
