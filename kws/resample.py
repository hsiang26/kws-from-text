# -*- coding: utf-8 -*-
"""
kws/resample.py
===============
把一個資料夾樹狀結構裡的音檔（.wav / .mp3 / .flac / .ogg）批次重採樣到固定取樣率、
mono、PCM_16，輸出到另一個資料夾並保留相對路徑。datasets/prepare.py 用它把
16kHz 的 Google Speech Commands / FMA 轉成 pipeline 需要的 44.1kHz。
"""

import glob
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import librosa
import soundfile as sf

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def _one(job):
    src, dst, sr = job
    if os.path.exists(dst):
        return False
    try:
        y, _ = librosa.load(src, sr=sr, mono=True)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sf.write(dst, y, sr, subtype="PCM_16")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] {src}: {e}")
        return False


def resample_tree(src_dir, dst_dir, target_sr=44100, workers=None, out_ext=".wav"):
    """src_dir 下所有音檔 → dst_dir，同樣的相對路徑、副檔名換成 out_ext。已存在的跳過。"""
    src_dir, dst_dir = os.path.abspath(src_dir), os.path.abspath(dst_dir)
    jobs = []
    for src in glob.glob(os.path.join(src_dir, "**", "*"), recursive=True):
        if not src.lower().endswith(AUDIO_EXTS):
            continue
        rel = os.path.relpath(src, src_dir)
        dst = os.path.join(dst_dir, os.path.splitext(rel)[0] + out_ext)
        jobs.append((src, dst, target_sr))

    if not jobs:
        print(f"  {src_dir} 裡沒有音檔")
        return 0

    workers = workers or max(1, multiprocessing.cpu_count() - 1)
    print(f"  重採樣 {len(jobs)} 檔 → {dst_dir}（{target_sr} Hz, {workers} workers）")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        done = sum(1 for r in ex.map(_one, jobs) if r)
    print(f"  完成，新增 {done} 檔")
    return done
