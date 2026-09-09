# -*- coding: utf-8 -*-
"""
kws/tts/piper.py —— Piper (piper-tts, ONNX) 後端。**英文預設。**

`en_US-libritts_r-medium`：LibriTTS-R 多語者 VITS，**904 個離散語者**，每個樣本隨機抽一個。
離線、不需 torch / CUDA（只要 onnxruntime）。跟原始 colab/AudioGen.ipynb 的 Piper 路線一致。

多樣性 = 隨機語者(0..903) × 隨機 length_scale(語速) × 隨機 noise_scale / noise_w_scale(韻律)
        × 句尾標點。

模型檔在 datasets/_models/piper/en_US-libritts_r-medium.onnx（+ .onnx.json），
由 `python datasets/prepare.py piper` 下載。

速度：CPU 上每筆約 0.05–0.6 秒。12000+12000 大約 30–75 分鐘（一次性）；smoke 只要幾十秒。
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .base import TTSBackend

_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets", "_models", "piper")


class PiperBackend(TTSBackend):
    name = "piper"

    # 合成參數抽樣範圍（Piper 預設 length_scale≈1.0, noise_scale≈0.667, noise_w_scale≈0.8）
    LENGTH_SCALE = (0.85, 1.25)
    NOISE_SCALE = (0.55, 0.90)
    NOISE_W_SCALE = (0.60, 1.00)

    def __init__(self, model_dir=None, voice="en_US-libritts_r-medium",
                 max_speakers=None, use_cuda=False, workers=4):
        self.model_dir = model_dir or _DEFAULT_MODEL_DIR
        self.voice = voice
        self.onnx = os.path.join(self.model_dir, f"{voice}.onnx")
        self.cfg_path = self.onnx + ".json"
        self.use_cuda = use_cuda
        self.workers = max(1, int(workers))
        self._voice = None

        n = 1
        if os.path.exists(self.cfg_path):
            with open(self.cfg_path, encoding="utf-8") as f:
                n = int(json.load(f).get("num_speakers", 1))
        self.num_speakers = n
        self.max_speakers = min(n, max_speakers) if max_speakers else n

    def describe(self):
        return (f"piper  {self.voice}  語者 {self.num_speakers} 個"
                f"（用前 {self.max_speakers}）  workers={self.workers}")

    def _load(self):
        if self._voice is None:
            if not os.path.exists(self.onnx):
                raise FileNotFoundError(
                    f"找不到 Piper 模型：{self.onnx}\n"
                    f"先跑：python datasets/prepare.py piper")
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.onnx, use_cuda=self.use_cuda)
        return self._voice

    # ---- 規劃 jobs -----------------------------------------------------------
    def plan(self, texts, count, rng, is_positive, punct_list):
        jobs = []
        # 前 max_speakers 個（或全部，取少的）做乾淨的；其餘 50% 加事後增強
        clean_quota = min(count, self.max_speakers)
        for i in range(count):
            base_text, group = texts[0] if is_positive else texts[rng.randrange(len(texts))]
            punct = punct_list[rng.randrange(len(punct_list))]
            sid = rng.randrange(self.max_speakers)
            params = {
                "speaker_id": sid,
                "length_scale": round(rng.uniform(*self.LENGTH_SCALE), 3),
                "noise_scale": round(rng.uniform(*self.NOISE_SCALE), 3),
                "noise_w_scale": round(rng.uniform(*self.NOISE_W_SCALE), 3),
            }
            augment = (i >= clean_quota) and (rng.random() < 0.5)
            jobs.append({"text": base_text + punct, "group": group,
                         "voice": f"spk{sid}", "params": params, "augment": augment})
        rng.shuffle(jobs)
        return jobs

    # ---- 合成 --------------------------------------------------------------
    def synth_all(self, jobs, concurrency):
        from piper import SynthesisConfig
        voice = self._load()
        sr = voice.config.sample_rate

        def one(job):
            try:
                p = job["params"]
                cfg = SynthesisConfig(
                    speaker_id=p["speaker_id"], length_scale=p["length_scale"],
                    noise_scale=p["noise_scale"], noise_w_scale=p["noise_w_scale"],
                    normalize_audio=False)
                chunks = list(voice.synthesize(job["text"], syn_config=cfg))
                if not chunks:
                    return job, None, None, "piper 回傳空音訊"
                wav = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
                return job, wav, sr, None
            except Exception as e:  # noqa: BLE001
                return job, None, None, str(e)

        workers = max(1, min(self.workers, concurrency or self.workers))
        if workers == 1:
            for job in jobs:
                yield one(job)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for r in ex.map(one, jobs):
                    yield r
