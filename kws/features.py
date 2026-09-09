# -*- coding: utf-8 -*-
"""
kws/features.py
===============
特徵提取 —— 波形 → MFCC-10（49 frames × 10），加上 INT8 量化 / 反量化。

**訓練和推論共用這一份**，避免 front-end 漂移（訓練用一套公式、推論用另一套，
分數就對不起來）。架構來自 Hello Edge（Zhang et al. 2017, arXiv:1711.07128）的
DS-CNN "Small" + MFCC，per-utterance CMVN 正規化。

音訊管線固定：44100 Hz mono、1.0 秒、20ms hop、1024-pt FFT、40 mel、取前 10 個 MFCC。
"""

import numpy as np
import tensorflow as tf

# ── 音訊 / 特徵參數（必須與部署端硬體 front-end 一致）──────────────────────────
SR = 44100
DURATION = 1.0
SAMPLES = int(SR * DURATION)                 # 44100
FRAME_STEP = int(round(0.020 * SR))          # 882 = 20ms
FFT_LENGTH = 1024
NUM_MELS = 40
NUM_MFCC = 10
MEL_LOW_HZ = 20.0
MEL_HIGH_HZ = 8000.0
NUM_FRAMES = 49                              # (SAMPLES - FFT_LENGTH) // FRAME_STEP + 1

# 預算 mel 濾波器組矩陣（(FFT_LENGTH//2+1) × NUM_MELS），跟 trainer 一致
MEL_MAT = tf.signal.linear_to_mel_weight_matrix(
    NUM_MELS, FFT_LENGTH // 2 + 1, SR, MEL_LOW_HZ, MEL_HIGH_HZ
).numpy().astype(np.float32)


def wav_to_mfcc(w, input_norm="peak", input_norm_target=0.9, input_norm_floor=1e-3):
    """
    (SAMPLES,) float32 波形 → (49, 10, 1) float32 MFCC 特徵。

    純 TensorFlow ops，eager（推論）和 graph（`tf.data.Dataset.map`）都能用。

    input_norm：特徵抽取前把波形拉到固定位準，解掉「小聲/遠場的字被 log(mel+1e-6)
    破壞」的問題。必須跟部署端硬體正規化 IP 一致。
      peak = 峰值拉到 target；rms = RMS 拉到 target；none = 不做。
    """
    if input_norm == "peak":
        lvl = tf.reduce_max(tf.abs(w))
        w = w * (input_norm_target / tf.maximum(lvl, input_norm_floor))
    elif input_norm == "rms":
        lvl = tf.sqrt(tf.reduce_mean(w * w) + 1e-12)
        w = w * (input_norm_target / tf.maximum(lvl, input_norm_floor))

    stft = tf.signal.stft(w, FFT_LENGTH, FRAME_STEP, FFT_LENGTH,
                          window_fn=tf.signal.hann_window)
    power_spec = tf.abs(stft) ** 2.0
    mel = tf.matmul(power_spec, MEL_MAT)
    log_mel = tf.math.log(mel + 1e-6)
    mfcc = tf.signal.dct(log_mel, type=2)[:, :NUM_MFCC]
    mfcc_norm = (mfcc - tf.reduce_mean(mfcc)) / (tf.math.reduce_std(mfcc) + 1e-6)
    return tf.expand_dims(tf.clip_by_value(mfcc_norm, -3.0, 3.0), -1)


def load_wav_fixed(path):
    """讀 wav → mono → pad/crop 到 SAMPLES。與 trainer 的 _read() 邏輯一致。"""
    b = tf.io.read_file(path)
    w, _ = tf.audio.decode_wav(b, desired_channels=1)
    w = tf.squeeze(w, -1)
    cur = tf.shape(w)[0]
    w = tf.cond(cur < SAMPLES,
                lambda: tf.pad(w, [[0, SAMPLES - cur]]),
                lambda: w[:SAMPLES])
    return tf.reshape(w, (SAMPLES,))


def load_tflite(path):
    """建立 tf.lite.Interpreter。用 model_content（bytes）而非 model_path —— 後者的 C++
    file-open 在 Windows 上開不了含中文的路徑（models/小幫手_.../...）。"""
    with open(path, "rb") as f:
        return tf.lite.Interpreter(model_content=f.read())


def quantize_input(mfcc, scale, zero_point):
    """float MFCC → int8，用 TFLite 輸入張量的 (scale, zero_point)。"""
    q = np.round(np.asarray(mfcc) / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_output(q, scale, zero_point):
    """int8 輸出 → float 機率。"""
    return (np.asarray(q).astype(np.float32) - zero_point) * scale
