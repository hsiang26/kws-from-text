# -*- coding: utf-8 -*-
"""
inference/infer.py
==================
離線評分器 —— 拿訓練好的 model_int8.tflite 對 .wav 檔 / 資料夾 / labels.csv 測試集跑推論，
輸出每個檔案是不是喚醒詞 + 信心分數。特徵提取跟訓練共用 kws/features.py。

    python inference/infer.py --input clip.wav
    python inference/infer.py --input some_folder/            # 遞迴掃所有 wav
    python inference/infer.py --model models/hey_assistant_xxx --input x.wav --threshold 0.7

    # 對正式測試集算 precision/recall/FAR/F1 + 類別分解 + 列出 FN/FP：
    python inference/infer.py --labels-csv test_database/labels.csv --input test_database

給資料夾（無 labels.csv）時，會從路徑（含 "positive"/"negative"）猜正確答案順便算指標。
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kws import paths
from kws import features as F

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_model(model_arg):
    """model_arg 可以是 run 資料夾、.tflite 檔、或 None（取 models/ 最新一個）。"""
    if model_arg:
        p = model_arg
        run_dir = p if os.path.isdir(p) else os.path.dirname(p)
        tflite = p if p.endswith(".tflite") else os.path.join(p, "model_int8.tflite")
    else:
        latest = paths.latest_model_dir()
        if latest is None:
            sys.exit("找不到任何模型。先訓練，或用 --model 指定。")
        run_dir, tflite = str(latest), str(latest / "model_int8.tflite")

    if not os.path.exists(tflite):
        sys.exit(f"找不到 {tflite}")

    meta_path = os.path.join(run_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
    return tflite, meta


def _load_items(args):
    """回傳 [(abs_path, truth|None, category), ...]。"""
    if args.labels_csv:
        base = args.input or os.path.dirname(args.labels_csv)
        items = []
        with open(args.labels_csv, "r", encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                rel = r["relative_path"].replace("/", os.sep)
                items.append((os.path.join(base, rel),
                              r["label"].strip().lower(), r.get("category", "")))
        return items
    if os.path.isdir(args.input):
        wavs = sorted(glob.glob(os.path.join(args.input, "**", "*.wav"), recursive=True))
    else:
        wavs = [args.input]
    return [(w, _guess_truth(w), "") for w in wavs]


def main():
    ap = argparse.ArgumentParser(description="離線喚醒詞評分器")
    ap.add_argument("--model", default=None, help="run 資料夾 / .tflite（預設 models/ 最新）")
    ap.add_argument("--input", default=None, help=".wav 檔或資料夾")
    ap.add_argument("--labels-csv", default=None,
                    help="測試集 labels.csv（欄位 relative_path,label,category）；"
                         "路徑相對 --input（或 csv 所在資料夾）")
    ap.add_argument("--threshold", type=float, default=None, help="判定門檻（預設讀 metadata）")
    ap.add_argument("--quiet", action="store_true", help="不印每個檔案，只印總結")
    args = ap.parse_args()
    if not args.input and not args.labels_csv:
        ap.error("要給 --input 或 --labels-csv")

    import tensorflow as tf

    tflite, meta = load_model(args.model)
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

    print(f"模型: {tflite}")
    print(f"喚醒詞: {wake_word!r}   門檻: {thr}   正類 index: {pos_idx}\n")

    items = _load_items(args)
    if not items:
        sys.exit("找不到任何 wav")

    rows = []
    for path, truth, cat in items:
        if not os.path.exists(path):
            print(f"  [skip] 找不到 {path}")
            continue
        try:
            y, file_sr = sf.read(path, always_2d=False)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {path}: {e}")
            continue
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        y = np.pad(y, (0, samples - len(y)))[:samples] if len(y) < samples else y[:samples]

        mfcc = F.wav_to_mfcc(y, input_norm=norm["mode"],
                             input_norm_target=norm["target"],
                             input_norm_floor=norm["floor"]).numpy()[np.newaxis, ...]
        q = F.quantize_input(mfcc, in_scale, in_zp)
        interp.set_tensor(ind["index"], q)
        interp.invoke()
        probs = F.dequantize_output(interp.get_tensor(outd["index"])[0], out_scale, out_zp)
        score = float(probs[pos_idx])
        rows.append({"path": path, "score": score, "pred": score >= thr,
                     "truth": truth, "cat": cat})

    if not args.quiet:
        for r in rows:
            mark = "WAKE " if r["pred"] else "  -  "
            tag = f"  (truth={r['truth']})" if r["truth"] else ""
            print(f"  [{mark}] {r['score']:5.3f}  {os.path.relpath(r['path'])}{tag}")

    _summarize(rows, thr)


def _prf(rows):
    tp = sum(1 for r in rows if r["truth"] == "positive" and r["pred"])
    fn = sum(1 for r in rows if r["truth"] == "positive" and not r["pred"])
    fp = sum(1 for r in rows if r["truth"] == "negative" and r["pred"])
    tn = sum(1 for r in rows if r["truth"] == "negative" and not r["pred"])
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, prec=prec, rec=rec, far=far, f1=f1)


def _summarize(rows, thr):
    print(f"\n共 {len(rows)} 檔，判為 wake {sum(1 for r in rows if r['pred'])} 個（門檻 {thr}）")
    labelled = [r for r in rows if r["truth"] in ("positive", "negative")]
    if not labelled:
        return
    s = _prf(labelled)
    npos = sum(1 for r in labelled if r["truth"] == "positive")
    nneg = len(labelled) - npos
    print(f"ground truth：{npos} 正 / {nneg} 負")
    print(f"  TP={s['tp']} FN={s['fn']} FP={s['fp']} TN={s['tn']}")
    print(f"  Precision={s['prec']:.3f}  Recall(TPR)={s['rec']:.3f}  "
          f"FAR={s['far']:.3f}  F1={s['f1']:.3f}  "
          f"Accuracy={(s['tp']+s['tn'])/len(labelled):.3f}")

    cats = sorted({r["cat"] for r in labelled if r["cat"]})
    if cats:
        print("  類別分解：")
        for c in cats:
            sub = [r for r in labelled if r["cat"] == c]
            acc = sum(1 for r in sub if (r["truth"] == "positive") == r["pred"]) / len(sub)
            print(f"    {c:8s} n={len(sub):4d}  acc={acc:.3f}")

    fn_rows = sorted((r for r in labelled if r["truth"] == "positive" and not r["pred"]),
                     key=lambda r: r["score"])
    fp_rows = sorted((r for r in labelled if r["truth"] == "negative" and r["pred"]),
                     key=lambda r: -r["score"])
    if fn_rows:
        print(f"  漏抓的喚醒詞 (FN={len(fn_rows)})：")
        for r in fn_rows:
            print(f"    {r['score']:.3f}  {os.path.relpath(r['path'])}")
    if fp_rows:
        print(f"  誤觸發 (FP={len(fp_rows)})：")
        for r in fp_rows:
            print(f"    {r['score']:.3f}  {os.path.relpath(r['path'])}  [{r['cat']}]")


def _guess_truth(path):
    p = path.replace("\\", "/").lower()
    if "/positive/" in p or "/pos/" in p or os.path.basename(p).startswith("pos"):
        return "positive"
    if "/negative/" in p or "/neg/" in p or os.path.basename(p).startswith("neg"):
        return "negative"
    return None


if __name__ == "__main__":
    main()
