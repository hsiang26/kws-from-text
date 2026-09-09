# -*- coding: utf-8 -*-
"""
kws/tts/edge.py —— Microsoft edge-tts 後端（線上、免費、免金鑰）。

英文 ~24 個多國口音語者；中文有 zh-tw / zh-cn / zh-hk voice pack。
多樣性 = 語者(加權) × 語速網格 × 音高網格 × 句尾標點。

需要 `pip install -r requirements-edge.txt`（edge-tts）+ 網路。
HTTP 403 → `pip install -U edge-tts`（微軟會輪替 Sec-MS-GEC token）。
"""

import asyncio
import io

import numpy as np
import soundfile as sf

from .base import TTSBackend

VOICE_PACKS = {
    "zh-tw": {
        "zh-TW-HsiaoChenNeural": 3.0, "zh-TW-HsiaoYuNeural": 3.0, "zh-TW-YunJheNeural": 3.0,
        "zh-CN-XiaoxiaoNeural": 1.0, "zh-CN-XiaoyiNeural": 1.0, "zh-CN-YunxiNeural": 1.0,
        "zh-CN-YunyangNeural": 1.0, "zh-CN-YunjianNeural": 1.0, "zh-CN-YunxiaNeural": 1.0,
        "zh-CN-liaoning-XiaobeiNeural": 0.7, "zh-CN-shaanxi-XiaoniNeural": 0.7,
    },
    "zh-cn": {
        "zh-CN-XiaoxiaoNeural": 1.0, "zh-CN-XiaoyiNeural": 1.0,
        "zh-CN-YunxiNeural": 1.0, "zh-CN-YunyangNeural": 1.0,
        "zh-CN-YunjianNeural": 1.0, "zh-CN-YunxiaNeural": 1.0,
        "zh-CN-liaoning-XiaobeiNeural": 1.0, "zh-CN-shaanxi-XiaoniNeural": 1.0,
    },
    "zh-hk": {
        "zh-HK-HiuGaaiNeural": 1.0, "zh-HK-HiuMaanNeural": 1.0, "zh-HK-WanLungNeural": 1.0,
    },
    "en": {
        "en-US-AriaNeural": 2.0, "en-US-JennyNeural": 2.0, "en-US-GuyNeural": 2.0,
        "en-US-MichelleNeural": 2.0, "en-US-ChristopherNeural": 2.0, "en-US-EricNeural": 2.0,
        "en-US-RogerNeural": 1.5, "en-US-SteffanNeural": 1.5, "en-US-AnaNeural": 1.0,
        "en-GB-SoniaNeural": 1.5, "en-GB-RyanNeural": 1.5, "en-GB-LibbyNeural": 1.5,
        "en-GB-ThomasNeural": 1.0, "en-GB-MaisieNeural": 0.7,
        "en-AU-NatashaNeural": 1.0, "en-AU-WilliamNeural": 1.0,
        "en-CA-ClaraNeural": 1.0, "en-CA-LiamNeural": 1.0,
        "en-IE-EmilyNeural": 0.8, "en-IE-ConnorNeural": 0.8,
        "en-IN-NeerjaNeural": 0.8, "en-IN-PrabhatNeural": 0.8,
        "en-NZ-MollyNeural": 0.6, "en-ZA-LeahNeural": 0.6,
    },
    "en-us": {
        "en-US-AriaNeural": 1.0, "en-US-JennyNeural": 1.0, "en-US-GuyNeural": 1.0,
        "en-US-MichelleNeural": 1.0, "en-US-ChristopherNeural": 1.0, "en-US-EricNeural": 1.0,
        "en-US-RogerNeural": 1.0, "en-US-SteffanNeural": 1.0, "en-US-AnaNeural": 1.0,
    },
}

RATES_PCT = [-15, -8, 0, 8, 18, 28]
PITCHES_HZ = [-8, -4, 0, 4, 8]


class EdgeBackend(TTSBackend):
    name = "edge"

    def __init__(self, voice_pack="en", voices=""):
        if voices:
            self.weights = {v.strip(): 1.0 for v in voices.split(",") if v.strip()}
            self.pack_name = "custom"
        else:
            self.pack_name = voice_pack or "en"
            self.weights = dict(VOICE_PACKS[self.pack_name])

    def describe(self):
        return f"edge-tts  voice-pack={self.pack_name}  語者 {len(self.weights)} 個"

    # ---- 規劃 jobs -------------------------------------------------------------
    def plan(self, texts, count, rng, is_positive, punct_list):
        voices = list(self.weights)
        if is_positive and len(texts) == 1:
            return self._plan_single(texts[0], voices, count, rng, punct_list)
        return self._plan_multi(texts, voices, count, rng, punct_list)

    def _job(self, base_text, group, voice, rate, pitch, punct, augment):
        return {"text": base_text + punct, "group": group, "voice": voice,
                "params": {"rate_pct": rate, "pitch_hz": pitch}, "augment": augment}

    def _plan_single(self, text_group, voices, count, rng, punct_list):
        base_text, group = text_group
        grid = [(v, r, p, punct)
                for v in voices for r in RATES_PCT for p in PITCHES_HZ for punct in punct_list]
        rng.shuffle(grid)
        w = [self.weights.get(g[0], 1.0) for g in grid]
        jobs, i = [], 0
        while len(jobs) < count:
            v, r, p, punct = grid[i % len(grid)]
            augment = i >= len(grid)
            if augment and w[i % len(grid)] < 1.0 and rng.random() > w[i % len(grid)]:
                i += 1
                continue
            jobs.append(self._job(base_text, group, v, r, p, punct, augment))
            i += 1
        rng.shuffle(jobs)
        return jobs

    def _plan_multi(self, texts, voices, count, rng, punct_list):
        vw = np.array([self.weights[v] for v in voices], dtype=float)
        vw = np.cumsum(vw / vw.sum())
        distinct_cap = len(texts) * len(voices) * len(RATES_PCT) * len(PITCHES_HZ) * len(punct_list)
        seen, jobs, attempts = set(), [], 0
        while len(jobs) < count and attempts < count * 20:
            attempts += 1
            base_text, group = texts[rng.randrange(len(texts))]
            v = voices[int(np.searchsorted(vw, rng.random()))]
            r = RATES_PCT[rng.randrange(len(RATES_PCT))]
            p = PITCHES_HZ[rng.randrange(len(PITCHES_HZ))]
            punct = punct_list[rng.randrange(len(punct_list))]
            key = (base_text, v, r, p, punct)
            augment = False
            if key in seen:
                if len(seen) >= distinct_cap:
                    augment = True
                else:
                    continue
            seen.add(key)
            jobs.append(self._job(base_text, group, v, r, p, punct, augment))
        rng.shuffle(jobs)
        return jobs

    # ---- 合成（分批跑 asyncio，逐批 yield 出來，不把上千個波形全塞記憶體）--------
    def synth_all(self, jobs, concurrency):
        CHUNK = 200
        for c0 in range(0, len(jobs), CHUNK):
            batch = jobs[c0:c0 + CHUNK]
            yield from asyncio.run(_gather_batch(batch, concurrency))


async def _synth_one(text, voice, rate_pct, pitch_hz, retries=4):
    import edge_tts
    rate, pitch = f"{rate_pct:+d}%", f"{pitch_hz:+d}Hz"
    last_err = None
    for attempt in range(retries):
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            buf = io.BytesIO()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            if buf.getbuffer().nbytes == 0:
                raise RuntimeError("edge-tts 回傳空音訊")
            buf.seek(0)
            y, sr = sf.read(buf, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            return y.astype(np.float32), sr
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(1.5 * (attempt + 1))
    raise last_err


async def _gather_batch(batch, concurrency):
    sem = asyncio.Semaphore(concurrency)

    async def fetch(job):
        async with sem:
            try:
                y, sr = await _synth_one(job["text"], job["voice"],
                                         job["params"]["rate_pct"], job["params"]["pitch_hz"])
                return job, y, sr, None
            except Exception as e:  # noqa: BLE001
                return job, None, None, str(e)

    return list(await asyncio.gather(*(fetch(j) for j in batch)))
