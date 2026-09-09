# -*- coding: utf-8 -*-
"""
datasets/make_noise.py
======================
用 numpy 合成環境噪音，取代舊流程裡外部來的 noise_44/。輸出：

    datasets/noise/white/gen_white_0000.wav
    datasets/noise/pink/ ...
    datasets/noise/brown/ ...
    datasets/noise/fan/ ...
    datasets/noise/wind/ ...

各類預設 1000 段、每段 3 秒、44100 Hz mono PCM_16。訓練時 (kws/train.py) 會把這些
以 SNR 4~20dB 隨機混進正負樣本，也直接當「純噪音」負樣本用。

    python datasets/make_noise.py                 # 各類 1000 段
    python datasets/make_noise.py --per-type 20   # 煙霧測試用小量
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

TYPES = ("white", "pink", "brown", "fan", "wind")


def _colored(n, beta, rng):
    """beta: 0=white, 1=pink, 2=brown。頻域整形白噪音。"""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0)
    f[0] = f[1]
    X = X / (f ** (beta / 2.0))
    y = np.fft.irfft(X, n=n)
    return y / (np.max(np.abs(y)) + 1e-9)


def _fan(n, sr, rng):
    """風扇/馬達：pink 噪音 + 低頻嗡嗡聲（基頻 + 諧波）+ 緩慢振幅起伏。"""
    base = _colored(n, 1.0, rng) * 0.6
    t = np.arange(n) / sr
    f0 = rng.uniform(45, 120)
    hum = sum((0.5 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
              for k in (1, 2, 3, 4))
    am = 1.0 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.2, 1.5) * t)
    y = (base + 0.4 * hum) * am
    return y / (np.max(np.abs(y)) + 1e-9)


def _wind(n, sr, rng):
    """風聲：brown 噪音 + 緩慢的陣風振幅包絡 + 輕微低通。"""
    y = _colored(n, 2.2, rng)
    t = np.arange(n) / sr
    gust = 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t
                                     + rng.uniform(0, 6.28)))
    gust *= 1.0 + 0.3 * _colored(n, 2.0, rng)
    y = y * np.clip(gust, 0, None)
    # 一階低通
    a = 0.02
    out = np.zeros_like(y)
    acc = 0.0
    for i in range(n):
        acc += a * (y[i] - acc)
        out[i] = acc
    return out / (np.max(np.abs(out)) + 1e-9)


def gen_one(kind, n, sr, rng):
    if kind == "white":
        y = _colored(n, 0.0, rng)
    elif kind == "pink":
        y = _colored(n, 1.0, rng)
    elif kind == "brown":
        y = _colored(n, 2.0, rng)
    elif kind == "fan":
        y = _fan(n, sr, rng)
    elif kind == "wind":
        y = _wind(n, sr, rng)
    else:
        raise ValueError(kind)
    y = y * rng.uniform(0.5, 0.95)          # 隨機整體音量
    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="numpy 合成環境噪音")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "noise"))
    ap.add_argument("--per-type", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--types", default=",".join(TYPES))
    args = ap.parse_args()

    n = int(args.seconds * args.sr)
    for kind in [t.strip() for t in args.types.split(",") if t.strip()]:
        d = os.path.join(args.out, f"{kind}_noise")
        os.makedirs(d, exist_ok=True)
        have = len([f for f in os.listdir(d) if f.endswith(".wav")])
        if have >= args.per_type:
            print(f"{kind}: 已有 {have} 段，跳過")
            continue
        rng = np.random.default_rng(args.seed + hash(kind) % 100000)
        for i in range(have, args.per_type):
            y = gen_one(kind, n, args.sr, rng)
            sf.write(os.path.join(d, f"gen_{kind}_noise_{i:04d}.wav"), y, args.sr, subtype="PCM_16")
        print(f"{kind}: 生成到 {args.per_type} 段 → {d}")


if __name__ == "__main__":
    main()
