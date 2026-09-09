# -*- coding: utf-8 -*-
"""
inference/mic_demo.py
=====================
麥克風即時喚醒詞偵測。滑動視窗（1.0 秒，每 ~0.2 秒推論一次），連續兩次超過門檻就觸發。

    pip install -r requirements-mic.txt      # 需要 sounddevice
    python inference/mic_demo.py
    python inference/mic_demo.py --model models/hey_assistant_20260908_231511 --threshold 0.75

Ctrl+C 結束。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kws import paths
from kws import features as F

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

HOP_SEC = 0.20
CONSEC_HITS = 2          # 連續幾次超過門檻才觸發
COOLDOWN_SEC = 1.5       # 觸發後冷卻


def main():
    ap = argparse.ArgumentParser(description="麥克風即時喚醒詞偵測")
    ap.add_argument("--model", default=None, help="run 資料夾 / .tflite（預設 models/ 最新）")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", default=None, help="sounddevice 輸入裝置（名稱或 index）")
    args = ap.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("需要 sounddevice：pip install -r requirements-mic.txt")
    import tensorflow as tf

    # ── 載入模型 + metadata ──────────────────────────────────────────
    if args.model:
        run_dir = args.model if os.path.isdir(args.model) else os.path.dirname(args.model)
        tflite = args.model if args.model.endswith(".tflite") else os.path.join(run_dir, "model_int8.tflite")
    else:
        latest = paths.latest_model_dir()
        if latest is None:
            sys.exit("找不到任何模型。先訓練，或用 --model 指定。")
        run_dir, tflite = str(latest), str(latest / "model_int8.tflite")
    meta = {}
    mp = os.path.join(run_dir, "metadata.json")
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding="utf-8"))

    audio = meta.get("audio", {})
    sr = audio.get("sample_rate", F.SR)
    samples = audio.get("samples", F.SAMPLES)
    norm = meta.get("input_norm", {"mode": "peak", "target": 0.9, "floor": 1e-3})
    pos_idx = meta.get("positive_class_index", 1)
    thr = args.threshold if args.threshold is not None else meta.get("suggested_threshold", 0.5)
    wake_word = meta.get("wake_word", "?")

    interp = F.load_tflite(tflite)
    interp.allocate_tensors()
    ind, outd = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = ind["quantization"]
    out_scale, out_zp = outd["quantization"]

    def score(buf):
        mfcc = F.wav_to_mfcc(buf, input_norm=norm["mode"],
                             input_norm_target=norm["target"],
                             input_norm_floor=norm["floor"]).numpy()[np.newaxis, ...]
        q = F.quantize_input(mfcc, in_scale, in_zp)
        interp.set_tensor(ind["index"], q)
        interp.invoke()
        probs = F.dequantize_output(interp.get_tensor(outd["index"])[0], out_scale, out_zp)
        return float(probs[pos_idx])

    ring = np.zeros(samples, dtype=np.float32)
    hop = int(HOP_SEC * sr)
    hits = 0
    last_fire = 0.0

    def cb(indata, frames, time_info, status):
        nonlocal ring, hits, last_fire
        x = indata[:, 0] if indata.ndim > 1 else indata
        ring = np.roll(ring, -len(x))
        ring[-len(x):] = x
        s = score(ring)
        bar = "#" * int(s * 40)
        print(f"\r{s:5.3f} |{bar:<40}|", end="", flush=True)
        if s >= thr:
            hits += 1
            if hits >= CONSEC_HITS and time.time() - last_fire > COOLDOWN_SEC:
                last_fire = time.time()
                hits = 0
                print(f"\n🔔  WAKE  ({wake_word!r})  score={s:.3f}")
        else:
            hits = 0

    print(f"模型: {tflite}")
    print(f"喚醒詞: {wake_word!r}   門檻: {thr}")
    print("開始聽麥克風… 講喚醒詞試試，Ctrl+C 結束\n")
    with sd.InputStream(channels=1, samplerate=sr, blocksize=hop,
                        dtype="float32", device=args.device, callback=cb):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n結束。")


if __name__ == "__main__":
    main()
