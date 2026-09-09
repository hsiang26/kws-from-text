# -*- coding: utf-8 -*-
"""
kws/tts/piper_torch.py —— Piper 的 **PyTorch / GPU 批次** 後端（原 colab/AudioGen.ipynb 那條路）。

用 rhasspy/piper-sample-generator 的 `en_US-libritts_r-medium.pt`（VITS）+ torch，
GPU 上批次生成，且做 **SLERP 語者內插**（904 語者兩兩混 → 遠超 904 種音色）。
在 RTX 3050 上 batch 32 約 **150+ 筆/秒**（onnx CPU 約 5 筆/秒）。

⚠️ 它相依 `piper-tts==1.3.0`，跟主 venv 的 1.8.0 衝突 → 跑在**獨立的 `venv-psg/`**，
   由 `datasets/prepare.py piper-torch` 建好（venv + torch(cuda) + clone repo + 下載 .pt）。
   本後端只負責寫 spec.json → subprocess 呼叫 `kws/tts/psg_worker.py` → 讀回 wav。
"""

import json
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from .base import TTSBackend

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_HERE, "psg_worker.py")


class PiperTorchBackend(TTSBackend):
    name = "piper-torch"

    LENGTH_SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)
    NOISE_SCALES = (0.5, 0.6, 0.667, 0.8)
    NOISE_W_SCALES = (0.6, 0.8, 1.0)
    SLERP_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

    def __init__(self, psg_python, psg_repo, model_pt,
                 batch_size=32, max_speakers=None, chunk=3000):
        self.psg_python = psg_python
        self.psg_repo = psg_repo
        self.model_pt = model_pt
        self.batch_size = int(batch_size)
        self.max_speakers = max_speakers
        self.chunk = int(chunk)

    def describe(self):
        return (f"piper-torch  {os.path.basename(self.model_pt)}  "
                f"batch={self.batch_size}  (GPU 批次 + SLERP 語者內插)")

    def _check(self):
        if not os.path.exists(self.psg_python):
            raise SystemExit(f"找不到 venv-psg：{self.psg_python}\n先跑：python datasets/prepare.py piper-torch")
        if not os.path.exists(self.model_pt):
            raise SystemExit(f"找不到 Piper .pt 模型：{self.model_pt}\n先跑：python datasets/prepare.py piper-torch")

    # ---- 規劃 jobs：語者 / 韻律 / SLERP 的多樣性 psg 內部自己做，這裡 job 很單純 ----
    def plan(self, texts, count, rng, is_positive, punct_list):
        jobs = []
        for _ in range(count):
            base_text, group = texts[0] if is_positive else texts[rng.randrange(len(texts))]
            punct = punct_list[rng.randrange(len(punct_list))]
            jobs.append({"text": base_text + punct, "group": group,
                         "voice": "psg", "params": {}, "augment": False})
        rng.shuffle(jobs)
        return jobs

    # ---- 合成：分批 → subprocess → 讀回 ----
    def synth_all(self, jobs, concurrency):
        self._check()
        for c0 in range(0, len(jobs), self.chunk):
            batch_jobs = jobs[c0:c0 + self.chunk]
            with tempfile.TemporaryDirectory() as td:
                names = [f"{i:06d}.wav" for i in range(len(batch_jobs))]
                spec = {
                    "psg_repo": self.psg_repo,
                    "model": self.model_pt,
                    "out_dir": td,
                    "texts": [j["text"] for j in batch_jobs],
                    "file_names": names,
                    "batch_size": self.batch_size,
                    "length_scales": list(self.LENGTH_SCALES),
                    "noise_scales": list(self.NOISE_SCALES),
                    "noise_scale_ws": list(self.NOISE_W_SCALES),
                    "slerp_weights": list(self.SLERP_WEIGHTS),
                    "max_speakers": self.max_speakers,
                }
                spec_path = os.path.join(td, "_spec.json")
                with open(spec_path, "w", encoding="utf-8") as f:
                    json.dump(spec, f, ensure_ascii=False)

                r = subprocess.run([self.psg_python, _WORKER, spec_path],
                                   capture_output=True, text=True, errors="replace")
                if r.returncode != 0:
                    tail = ((r.stderr or "") + (r.stdout or ""))[-500:]
                    for j in batch_jobs:
                        yield j, None, None, f"psg worker exit {r.returncode}: {tail}"
                    continue

                for j, n in zip(batch_jobs, names):
                    p = os.path.join(td, n)
                    if not os.path.exists(p):
                        yield j, None, None, "psg 沒產出此檔"
                        continue
                    try:
                        y, sr = sf.read(p, dtype="float32", always_2d=False)
                        if y.ndim > 1:
                            y = y.mean(axis=1)
                        yield j, y.astype(np.float32), sr, None
                    except Exception as e:  # noqa: BLE001
                        yield j, None, None, str(e)
