# -*- coding: utf-8 -*-
"""
kws/paths.py
============
集中管理 repo 內所有路徑，全部從 repo root 相對算出。
資料集 / 模型 / 產物的實體位置只在這裡定義一次，其餘程式一律 import 這裡。

    datasets/
      speech_commands/        Google Speech Commands @ 44.1kHz（30 個詞資料夾）
      music/                  FMA-small 片段 @ 44.1kHz
      noise/                  生成的環境噪音 @ 44.1kHz（white/pink/brown/fan/wind）
      tts/<label>/
        positive/{train,test}/    喚醒詞正樣本（edge-tts）
        negative/{train,test}/    hard negatives（近音詞 / 部分片語 / 通用語音）
    models/<label>_<timestamp>/   每次訓練一個資料夾
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = REPO_ROOT / "datasets"
SPEECH_COMMANDS = DATASETS / "speech_commands"
NOISE = DATASETS / "noise"
MUSIC = DATASETS / "music"
TTS = DATASETS / "tts"
GENERIC_SPEECH = DATASETS / "generic_speech"   # 通用英文語音負樣本，全部喚醒詞共用

MODELS = REPO_ROOT / "models"


def tts_dir(label: str) -> Path:
    return TTS / label


def positive_dir(label: str) -> Path:
    return TTS / label / "positive"


def negative_dir(label: str) -> Path:
    return TTS / label / "negative"


def manifest_path(label: str, kind: str) -> Path:
    """kind ∈ {"positive", "negative"}"""
    return TTS / label / f"manifest_{kind}.csv"


def latest_model_dir() -> Path | None:
    if not MODELS.is_dir():
        return None
    runs = [p for p in MODELS.iterdir() if p.is_dir() and (p / "model_int8.tflite").exists()]
    return max(runs, key=lambda p: (p / "model_int8.tflite").stat().st_mtime) if runs else None
