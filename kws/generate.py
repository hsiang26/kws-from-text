# -*- coding: utf-8 -*-
"""
kws/generate.py
===============
批次合成訓練樣本，輸出到 kws/paths.py 定義的資料夾（全程 44100 Hz mono PCM_16）：

    --kind positive  → datasets/tts/<label>/positive/{train,test}/   喚醒詞本身
    --kind negative  → datasets/tts/<label>/negative/{train,test}/   近音 / 部分詞（每個喚醒詞自己生）
    --kind both      = positive + negative
    --kind generic   → datasets/generic_speech/{train,test}/         通用英文語音（不需喚醒詞，
                                                                     生一次全部喚醒詞共用）

TTS 後端可插拔（見 kws/tts/）：
    piper  ONNX 本地合成，en_US-libritts_r-medium（904 語者），離線 —— 英文正樣本 + 所有負樣本
    edge   Microsoft edge-tts（線上），zh-tw/zh-cn 等 —— 中文正樣本

負樣本文字：near-miss 來自 kws/negative_phrases.near_miss_phrases，generic 來自 .generic_phrases。
訓練腳本 (kws/train.py) 訓練時會混噪音 / FMA 音樂，所以這裡輸出「乾淨」樣本即可。

用法
----
    python -m kws.generate --kind both --wake-word "hey assistant"     # 正 + 近音負
    python -m kws.generate --kind generic --generic-count 10000        # 共用通用語音池
    python -m kws.generate --kind both --wake-word "小幫手"            # 中文（自動偵測）

    # resume：重跑同指令會從既有數量接續；--overwrite 先清空
"""

import argparse
import csv
import json
import os
import random
import sys
import uuid

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kws import paths, negative_phrases
from kws import tts

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import config as _cfg
except ImportError:
    _cfg = None

WORK_SR = 48000
SR_44 = 44100
CROP_SEC = 1.0
MAX_WORD_END = 0.95


# --------------------------------------------------------------------------------------
# 音訊後處理（跟 TTS 後端無關）
# --------------------------------------------------------------------------------------
def _trim(y, sr, top_db=30):
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if yt.size > int(0.1 * sr) else y


def make_clip(y, sr, rng, augment=True, fit_word=True):
    """
    fit_word=True （正樣本）：詞一定要在 0.95s 前講完，太長就時間壓縮塞進安全視窗。
    fit_word=False（負樣本）：不壓縮，放進 ~1.0-1.35s 的 clip，讓 trainer 的 1.0s crop 自然截斷。
    """
    if sr != WORK_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=WORK_SR)
    sr = WORK_SR
    y = _trim(y, sr)

    if augment:
        y = librosa.effects.time_stretch(y, rate=rng.uniform(0.90, 1.12))
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=rng.uniform(-2.5, 2.5))
        y = _trim(y, sr)

    crop = int(CROP_SEC * sr)
    if fit_word:
        max_word = int(MAX_WORD_END * sr)
        if y.size > max_word:
            y = librosa.effects.time_stretch(y, rate=y.size / max_word)
            y = _trim(y, sr)
            y = y[:max_word]
        pre_max = max(0, max_word - y.size)
        pre = int(min(rng.uniform(0.03, 0.30) * sr, pre_max))
        total = int(rng.uniform(1.05, 1.40) * sr)
        total = max(total, pre + y.size + int(0.05 * sr))
    else:
        pre = int(rng.uniform(0.0, 0.12) * sr)
        y = y[: int(1.5 * sr)]
        total = int(rng.uniform(1.00, 1.35) * sr)
        total = max(total, pre + min(y.size, crop) + int(0.02 * sr))

    clip = np.zeros(total, dtype=np.float32)
    n = min(y.size, total - pre)
    clip[pre:pre + n] = y[:n]
    peak = float(np.max(np.abs(clip))) + 1e-9
    clip *= 0.94 / peak
    return clip


def _write_clip(clip48, base, split, name):
    y44 = librosa.resample(clip48, orig_sr=WORK_SR, target_sr=SR_44)
    sf.write(os.path.join(base, split, name), np.clip(y44, -1, 1), SR_44, subtype="PCM_16")


# --------------------------------------------------------------------------------------
# 目錄 / 計數（路徑集中在 kws/paths.py；全程 44.1kHz；train/test 子夾由 trainer 遞迴 glob）
# --------------------------------------------------------------------------------------
SPLITS = ("train", "test")


def _base_dir(kind, label):
    if kind == "positive":
        return str(paths.positive_dir(label))
    if kind == "negative":
        return str(paths.negative_dir(label))
    if kind == "generic":
        return str(paths.GENERIC_SPEECH)
    raise ValueError(kind)


def _ensure_dirs(base):
    for s in SPLITS:
        os.makedirs(os.path.join(base, s), exist_ok=True)


def _count_existing(base):
    n = 0
    for s in SPLITS:
        p = os.path.join(base, s)
        if os.path.isdir(p):
            n += len([f for f in os.listdir(p) if f.lower().endswith(".wav")])
    return n


def _clear(base):
    removed = 0
    for s in SPLITS:
        p = os.path.join(base, s)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f.lower().endswith(".wav"):
                os.remove(os.path.join(p, f))
                removed += 1
    return removed


# --------------------------------------------------------------------------------------
# 後端建構
# --------------------------------------------------------------------------------------
def _cfg_get(name, default):
    return getattr(_cfg, name, default) if _cfg is not None else default


def is_cjk(s):
    return any("⺀" <= c <= "鿿" or "豈" <= c <= "﫿" for c in s)


def _piper_backend(args):
    voice = args.piper_voice or _cfg_get("PIPER_VOICE", "en_US-libritts_r-medium")
    max_spk = args.piper_max_speakers or _cfg_get("PIPER_MAX_SPEAKERS", None)
    engine = (args.piper_engine or _cfg_get("PIPER_ENGINE", "onnx")).lower()

    if engine == "torch":
        psg = paths.DATASETS / "_models" / "piper_sample_generator"
        return tts.get_backend(
            "piper-torch",
            psg_python=str(paths.REPO_ROOT / "venv-psg" / "Scripts" / "python.exe"),
            psg_repo=str(psg),
            model_pt=str(psg / "models" / f"{voice}.pt"),
            batch_size=_cfg_get("PIPER_TORCH_BATCH", 32),
            max_speakers=max_spk,
        )
    return tts.get_backend(
        "piper", voice=voice, max_speakers=max_spk,
        workers=args.concurrency or _cfg_get("PIPER_WORKERS", 4),
    )


def pick_backends(args):
    """
    回傳 {"positive": backend, "negative": backend}。

    中文喚醒詞：正樣本固定用 edge-tts（真中文，zh voice pack），負樣本固定用 piper
    （負樣本文字都是英文 / 拼音，見 kws/negative_phrases._build_cjk）。
    英文喚醒詞：正負都用 --tts-backend（預設 piper）。
    """
    if is_cjk(args.wake_word):
        pack = args.voice_pack or _cfg_get("ZH_VOICE_PACK", "zh-tw")
        print(f"[語言] 偵測到中文喚醒詞 → 正樣本 edge-tts（{pack}），負樣本 piper")
        return {"positive": tts.get_backend("edge", voice_pack=pack, voices=args.voices),
                "negative": _piper_backend(args)}
    if args.tts_backend == "piper":
        b = _piper_backend(args)
    else:
        b = tts.get_backend("edge", voice_pack=args.voice_pack or "en", voices=args.voices)
    return {"positive": b, "negative": b}


# --------------------------------------------------------------------------------------
# 執行一種 kind
# --------------------------------------------------------------------------------------
MANIFEST_COLS = ["filename", "split", "group", "backend", "voice", "text", "params", "augment"]


def _plan_texts(kind, args, rng):
    """回傳 [(base_text, group), ...] —— 這個 kind 要合成的文字。"""
    if kind == "positive":
        return [(args.wake_word, "positive")]
    if kind == "negative":
        near = negative_phrases.near_miss_phrases(args.wake_word, rng)
        if not near:
            print("[negative] 警告：產生不出近音 / 部分詞片語")
        return [(p, "near_miss") for p in near]
    if kind == "generic":
        return [(p, "generic") for p in negative_phrases.generic_phrases(rng)]
    raise ValueError(kind)


def run_kind(kind, args, rng, backend):
    is_positive = kind == "positive"
    fit_word = is_positive
    positive_cjk = is_positive and is_cjk(args.wake_word)
    punct_list = tts.PUNCT_CJK if positive_cjk else tts.PUNCT_LATIN

    base = _base_dir(kind, args.label)
    target = {"positive": args.pos_count, "negative": args.neg_count,
              "generic": args.generic_count}[kind]
    if kind == "generic":
        manifest_path = str(paths.GENERIC_SPEECH / "manifest.csv")
    else:
        manifest_path = str(paths.manifest_path(args.label, kind))

    _ensure_dirs(base)
    if args.overwrite:
        print(f"[{kind}] overwrite：刪掉舊的 {_clear(base)} 個 wav")
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

    existing = _count_existing(base)
    need = max(0, target - existing)
    print(f"\n=== [{kind}] 目標 {target}，已存在 {existing}，本次要產生 {need} ===")
    print(f"後端：{backend.describe()}")
    if need == 0:
        print(f"[{kind}] 已達標，跳過。")
        return

    texts = _plan_texts(kind, args, rng)
    if not texts:
        print(f"[{kind}] 沒有可合成的文字，跳過。")
        return
    jobs = backend.plan(texts, need, rng, is_positive, punct_list)
    print(f"[{kind}] {len(texts)} 條文字 → 規劃 {len(jobs)} 個 job")

    new_file = not os.path.exists(manifest_path)
    mf = open(manifest_path, "a", newline="", encoding="utf-8-sig")
    mw = csv.writer(mf)
    if new_file:
        mw.writerow(MANIFEST_COLS)

    done = fail = 0
    for job, wav, sr, err in backend.synth_all(jobs, args.concurrency):
        if err is not None:
            fail += 1
            if fail <= 10:
                print(f"  [skip] {job['voice']} {job['text']!r}: {err}")
            continue
        try:
            clip = make_clip(wav, sr, rng, augment=job["augment"] and args.augment, fit_word=fit_word)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  [skip-dsp] {e}")
            continue
        split = SPLITS[1] if rng.random() < args.test_frac else SPLITS[0]
        name = uuid.uuid4().hex + ".wav"
        _write_clip(clip, base, split, name)
        mw.writerow([name, split, job["group"], backend.name, job["voice"],
                     job["text"], json.dumps(job["params"], ensure_ascii=False),
                     int(job["augment"])])
        done += 1
        if done % 200 == 0:
            mf.flush()
            print(f"  進度 {done}/{len(jobs)}  (失敗 {fail})")

    mf.close()
    print(f"[{kind}] 完成：本次成功 {done}，失敗 {fail}。現有 {_count_existing(base)} 筆。manifest: {manifest_path}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="產生客製化喚醒詞的正 / 負樣本（可插拔 TTS 後端）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["positive", "negative", "both", "generic"], default="both",
                    help="both = positive + negative(近音)；generic = 共用通用語音池（不需喚醒詞）")
    ap.add_argument("--wake-word", default=None,
                    help='喚醒詞文字（--kind generic 以外都要）；含空白請加引號')
    ap.add_argument("--label", default=None, help="資料夾標籤（預設 = wake-word 空白換底線）")
    ap.add_argument("--pos-count", type=int, default=12000)
    ap.add_argument("--neg-count", type=int, default=6000)
    ap.add_argument("--generic-count", type=int, default=10000)
    ap.add_argument("--tts-backend", choices=["piper", "edge"],
                    default=_cfg_get("TTS_BACKEND", "piper"))
    # piper 專屬
    ap.add_argument("--piper-voice", default=None)
    ap.add_argument("--piper-max-speakers", type=int, default=None)
    ap.add_argument("--piper-engine", choices=["onnx", "torch"], default=None,
                    help="onnx（CPU，預設）/ torch（NVIDIA GPU 批次，要 venv-psg）")
    # edge 專屬
    ap.add_argument("--voice-pack", default=None,
                    help="edge 後端的語者組合：en / en-us / zh-tw / zh-cn / zh-hk")
    ap.add_argument("--voices", default="", help="edge 自訂語者清單，逗號分隔，覆蓋 --voice-pack")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--overwrite", action="store_true", help="先清空既有樣本再產生")
    args = ap.parse_args()
    if args.kind != "generic" and not args.wake_word:
        ap.error("--kind positive/negative/both 需要 --wake-word")
    args.label = (args.label or args.wake_word or "").strip().replace(" ", "_")
    return args


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    if args.kind == "generic":
        run_kind("generic", args, rng, _piper_backend(args))
        return
    backends = pick_backends(args)
    kinds = ["positive", "negative"] if args.kind == "both" else [args.kind]
    for k in kinds:
        run_kind(k, args, rng, backends[k])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中斷。重跑同指令會從既有數量接續。")
        sys.exit(1)
