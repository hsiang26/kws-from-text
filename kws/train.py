# -*- coding: utf-8 -*-
"""
kws/train.py
Hello Edge Small (S) DS-CNN + MFCC 10
架構來自：Zhang et al. 2017, arXiv:1711.07128, Table 7
特徵：MFCC 10 係數（49 frames × 10），per-utterance CMVN 正規化（見 kws/features.py）

資料來源（路徑集中在 kws/paths.py）：
  datasets/tts/<label>/positive        喚醒詞正樣本（edge-tts）
  datasets/tts/<label>/negative        hard negatives（近音 / 部分片語 / 通用語音）
  datasets/speech_commands/            Google Speech Commands（負樣本）
  datasets/music/                      FMA 音樂（負樣本 + 訓練時混入）
  datasets/noise/                      環境噪音（負樣本 + 訓練時混入）

執行：`python -m kws.train --command_labels <label> ...`（run_all.py 會代勞）
"""

import os, sys, glob, random, argparse, csv, time, json, datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, BackupAndRestore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kws import paths
from kws.features import (
    wav_to_mfcc as _features_wav_to_mfcc,
    load_wav_fixed as _load_wav_fixed,
    load_tflite as _features_load_tflite,
    SR, DURATION, SAMPLES, FRAME_STEP, FFT_LENGTH, NUM_MELS, NUM_MFCC,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_mode", type=str, default="lite")
    parser.add_argument("--run_dir",       type=str,   default="")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size",    type=int,   default=100)
    parser.add_argument("--max_epochs",    type=int,   default=100)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--target_unknown_count", type=int, default=4000)
    parser.add_argument("--ratio_gsc", type=float, default=0.2)
    parser.add_argument("--ratio_near_miss", type=float, default=0.25)
    parser.add_argument("--ratio_generic", type=float, default=0.25)
    parser.add_argument("--ratio_fma", type=float, default=0.1)
    parser.add_argument("--ratio_pure_noise", type=float, default=0.2)
    parser.add_argument("--mix_prob_noise", type=float, default=0.8)
    parser.add_argument("--mix_prob_fma", type=float, default=0.6)
    parser.add_argument("--ratio_command_to_unknown", type=float, default=1.5)
    parser.add_argument("--command_labels", type=str, default="hey_assistant")
    parser.add_argument("--wake_word", type=str, default="",
                         help="喚醒詞原文（只用來寫進 metadata.json；預設 = command_labels 底線換空白）")
    parser.add_argument("--num_test_samples", type=int, default=50)
    parser.add_argument("--suggested_threshold", type=float, default=0.9,
                         help="寫進 metadata.json 的預設判定門檻（infer/mic_demo 會讀）")
    parser.add_argument("--snr_min", type=float, default=4.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--test_database_dir", type=str, default="",
                         help="Root dir of the real-world test set, default <project_root>/test_database")
    parser.add_argument("--test_labels_csv", type=str, default="",
                         help="labels.csv for the real-world test set, default <test_database_dir>/labels.csv")
    parser.add_argument("--skip_external_eval", action="store_true",
                         help="Skip the real-world test set evaluation after training")

    # --- input level normalisation (see wav_to_mfcc) -----------------------
    # MUST mirror the deployed hardware front-end normalisation IP, or this
    # just relocates the train/deploy mismatch. peak = scale so max|x| ==
    # target; rms = scale so RMS == target; none = original behaviour.
    parser.add_argument("--input_norm", type=str, default="peak",
                         choices=["peak", "rms", "none"])
    parser.add_argument("--input_norm_target", type=float, default=0.9)
    parser.add_argument("--input_norm_floor", type=float, default=1e-3,
                         help="caps the max gain (target/floor); below this level the clip is left alone")

    # --- band-limiting augmentation (see load_and_mix) --------------------
    # The model trained on edge-tts positives (~41% of energy above 4 kHz);
    # it drops to ~0 score once a clip is low-passed below ~6 kHz (phone /
    # Bluetooth-HFP / cheap-mic captures). Randomly band-limiting a fraction
    # of training clips teaches it to tolerate that. Set --aug_lowpass_prob 0
    # to disable if the deployed mic is always full-band.
    parser.add_argument("--aug_lowpass_prob", type=float, default=0.35)
    parser.add_argument("--aug_lowpass_min_hz", type=float, default=3200.0)
    parser.add_argument("--aug_lowpass_max_hz", type=float, default=8000.0)

    # --- reverberation augmentation (see load_and_mix / _reverb) ----------
    # edge-tts positives are anechoic (close-mic, no room). A word spoken
    # ~0.5 m from a mic in a real cabin/room arrives with a reverberant tail
    # + a raised noise floor once level-normalised, which the model scores
    # low (the pos_real "girl"/"pinhong" clips: quiet + 300-425 ms decay).
    # Convolving a fraction of training clips with a synthetic room impulse
    # response teaches it to tolerate that. t60 = -60 dB reverb time (car
    # cabin ~0.05-0.15 s, small room ~0.3-0.6 s); mix = dry/wet ratio.
    # For higher fidelity, swap the synthetic RIR for real ones (MIT IR
    # Survey / openSLR RIR) -- see _reverb's docstring.
    parser.add_argument("--aug_reverb_prob", type=float, default=0.4)
    parser.add_argument("--aug_reverb_t60_min", type=float, default=0.15)
    parser.add_argument("--aug_reverb_t60_max", type=float, default=0.7)
    parser.add_argument("--aug_reverb_mix_min", type=float, default=0.2)
    parser.add_argument("--aug_reverb_mix_max", type=float, default=0.7)
    return parser.parse_args()

args = parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# Audio Parameters（SR / SAMPLES / FRAME_STEP / NUM_MELS / NUM_MFCC / FFT_LENGTH / mel_mat
# 全部從 kws/features.py import，訓練與推論共用同一份定義）
SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}

# Path Configuration（集中在 kws/paths.py）
ROOT_DIR = str(paths.REPO_ROOT)
DATA_GSC = str(paths.SPEECH_COMMANDS)
DATA_POS = str(paths.positive_dir(args.command_labels))
DATA_NEG_NEAR_MISS = str(paths.negative_dir(args.command_labels))  # 每個喚醒詞自己的近音負樣本
DATA_NEG_GENERIC = str(paths.GENERIC_SPEECH)                       # 共用的通用英文語音負樣本
DATA_NEG_FMA = str(paths.MUSIC)        # FMA 是 label 無關的共用池
DATA_NOISE = str(paths.NOISE)

if args.run_dir:
    OUTPUT_DIR = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(ROOT_DIR, args.run_dir)
else:
    OUTPUT_DIR = str(paths.MODELS / f"kws_ds_cnn_{args.model_mode}")
BACKUP_DIR = os.path.join(OUTPUT_DIR, "training_backup")
os.makedirs(BACKUP_DIR, exist_ok=True)

TEST_DB_DIR = args.test_database_dir or ""
TEST_LABELS_CSV = args.test_labels_csv or (os.path.join(TEST_DB_DIR, "labels.csv") if TEST_DB_DIR else "")

# Data Splitting
def get_file_splits():
    def _split(lst):
        random.shuffle(lst)
        n = len(lst)
        i1, i2 = int(n*0.8), int(n*0.9)
        return {"train": lst[:i1], "val": lst[i1:i2], "test": lst[i2:]}
    return {
        "pos": _split(glob.glob(os.path.join(DATA_POS, "**", "*.wav"), recursive=True)),
        "gsc": _split(glob.glob(os.path.join(DATA_GSC, "**", "*.wav"), recursive=True)),
        "near_miss": _split(glob.glob(os.path.join(DATA_NEG_NEAR_MISS, "**", "*.wav"), recursive=True)),
        "generic": _split(glob.glob(os.path.join(DATA_NEG_GENERIC, "**", "*.wav"), recursive=True)),
        "fma": _split(glob.glob(os.path.join(DATA_NEG_FMA, "**", "*.wav"), recursive=True)),
        "noise": _split(glob.glob(os.path.join(DATA_NOISE, "**", "*.wav"), recursive=True))
    }

ALL_SPLITS = get_file_splits()


@tf.function
def wav_to_mfcc(w):
    """(SAMPLES,) 波形 → (49,10,1) MFCC。實作在 kws/features.py，這裡只把 CLI 的
    --input_norm* 參數帶進去（訓練與推論共用同一份公式，避免 front-end 漂移）。"""
    return _features_wav_to_mfcc(
        w,
        input_norm=args.input_norm,
        input_norm_target=args.input_norm_target,
        input_norm_floor=args.input_norm_floor,
    )

def _band_limit(v, cutoff_hz):
    """Soft low-pass in the FFT domain (raised-cosine transition ~500 Hz),
    to simulate phone / Bluetooth-HFP / cheap-mic band-limiting. cutoff_hz
    >= SR/2 is a no-op."""
    V = tf.signal.rfft(v)
    nb = tf.shape(V)[0]
    freqs = tf.linspace(0.0, SR / 2.0, nb)
    trans = 500.0
    mask = tf.clip_by_value((cutoff_hz + trans - freqs) / trans, 0.0, 1.0)
    V = V * tf.complex(mask, tf.zeros_like(mask))
    return tf.signal.irfft(V, fft_length=[SAMPLES])


_RIR_LEN = SAMPLES // 2  # 0.5 s synthetic impulse response is plenty for t60 <= ~0.7 s


def _reverb(v, t60, wet_mix):
    """Convolve v with a synthetic room impulse response and blend dry/wet.

    The RIR is exponentially-decaying decorrelated noise -- a standard cheap
    stand-in for a real recorded RIR that captures what matters here (energy
    smears forward in time and decorrelates). The dry term IS the direct
    path (time-aligned), so wet_mix alone sets the direct-to-reverberant
    ratio. For higher fidelity, replace this body with a gather from a pool
    of real RIR wavs (MIT IR Survey / openSLR RIR), same as noise_pool.
    """
    t = tf.range(_RIR_LEN, dtype=tf.float32) / float(SR)
    decay = tf.exp(-6.9077553 * t / t60)                 # -60 dB at t60
    rir = tf.random.normal([_RIR_LEN]) * decay
    rir = rir / (tf.sqrt(tf.reduce_mean(rir * rir)) + 1e-9)   # unit-RMS tail

    L = SAMPLES + _RIR_LEN
    wet = tf.signal.irfft(
        tf.signal.rfft(v, fft_length=[L]) * tf.signal.rfft(rir, fft_length=[L]),
        fft_length=[L],
    )[:SAMPLES]
    # match wet energy to dry so wet_mix is a true ratio, not a level change
    wet = wet * (tf.sqrt(tf.reduce_mean(v * v) + 1e-12)
                 / (tf.sqrt(tf.reduce_mean(wet * wet) + 1e-12)))
    return (1.0 - wet_mix) * v + wet_mix * wet


def load_and_mix(path, label, stype, noise_pool, fma_pool, training=True):
    def _read(p):
        try:
            b = tf.io.read_file(p)
            w, _ = tf.audio.decode_wav(b, desired_channels=1)
            w = tf.squeeze(w, -1)
            cur = tf.shape(w)[0]
            w = tf.cond(cur < SAMPLES, lambda: tf.pad(w, [[0, SAMPLES-cur]]), lambda: w[:SAMPLES])
            return tf.reshape(w, (SAMPLES,))
        except: return tf.zeros((SAMPLES,))
    
    v = _read(path)

    if training:
        # room reverb BEFORE additive noise: the noise_44 clips are already
        # real (roomy) recordings, so the clean speech is what needs a room.
        if tf.random.uniform(()) < args.aug_reverb_prob:
            t60 = tf.random.uniform((), args.aug_reverb_t60_min, args.aug_reverb_t60_max)
            wet_mix = tf.random.uniform((), args.aug_reverb_mix_min, args.aug_reverb_mix_max)
            v = _reverb(v, t60, wet_mix)

        if stype != "pure_noise" and tf.random.uniform(()) < args.mix_prob_noise:
            n_path = tf.gather(noise_pool, tf.random.uniform((), 0, tf.shape(noise_pool)[0], tf.int32))
            n = _read(n_path)
            snr = tf.random.uniform((), args.snr_min, args.snr_max)
            scale = tf.pow(10.0, -snr/20.0) * (tf.math.reduce_std(v)/(tf.math.reduce_std(n)+1e-6))
            v += n * scale

        if stype != "fma" and tf.random.uniform(()) < args.mix_prob_fma:
            m_path = tf.gather(fma_pool, tf.random.uniform((), 0, tf.shape(fma_pool)[0], tf.int32))
            m = _read(m_path)
            snr = tf.random.uniform((), args.snr_min, args.snr_max)
            scale = tf.pow(10.0, -snr/20.0) * (tf.math.reduce_std(v)/(tf.math.reduce_std(m)+1e-6))
            v += m * scale

        # band-limit last: a phone/BT codec bandlimits speech + its background
        # together. Applied to positives AND negatives so the model doesn't
        # learn "narrowband => not the wake word".
        if tf.random.uniform(()) < args.aug_lowpass_prob:
            cutoff = tf.random.uniform((), args.aug_lowpass_min_hz, args.aug_lowpass_max_hz)
            v = _band_limit(v, cutoff)

    return wav_to_mfcc(v), label

SAMPLE_STATS = {}

def _sample(pool, k, tag, split):
    """Sample without replacement. If k exceeds the pool size, just take everything
    available (no replacement means no duplicates within one epoch's draw)."""
    used = random.sample(pool, min(k, len(pool))) if pool else []
    SAMPLE_STATS.setdefault(split, {})[tag] = {
        "requested": k, "available": len(pool), "used": len(used)}
    if k > len(pool):
        print(f"[Sampling] {split}/{tag}: requested {k} but pool only has {len(pool)}; "
              f"using all {len(pool)} available (no replacement).")
    return used

def create_ds(name):
    m = SPLIT_RATIO[name]
    nu = int(args.target_unknown_count * m)
    final = []

    for f in _sample(ALL_SPLITS["pos"][name], int(nu*args.ratio_command_to_unknown), "pos", name): final.append((f, 1, "pos"))
    for f in _sample(ALL_SPLITS["gsc"][name], int(nu*args.ratio_gsc), "gsc", name): final.append((f, 0, "gsc"))
    for f in _sample(ALL_SPLITS["near_miss"][name], int(nu*args.ratio_near_miss), "near_miss", name): final.append((f, 0, "near_miss"))
    for f in _sample(ALL_SPLITS["generic"][name], int(nu*args.ratio_generic), "generic", name): final.append((f, 0, "generic"))
    for f in _sample(ALL_SPLITS["fma"][name], int(nu*args.ratio_fma), "fma", name): final.append((f, 0, "fma"))
    for f in _sample(ALL_SPLITS["noise"][name], int(nu*args.ratio_pure_noise), "noise", name): final.append((f, 0, "pure_noise"))

    random.shuffle(final)
    p, l, t = zip(*final)
    
    npool = tf.constant(ALL_SPLITS["noise"][name] if ALL_SPLITS["noise"][name] else [""])
    fpool = tf.constant(ALL_SPLITS["fma"][name] if ALL_SPLITS["fma"][name] else [""])
    
    ds = tf.data.Dataset.from_tensor_slices((list(p), list(l), list(t)))
    return ds.map(lambda p,l,t: load_and_mix(p,l,t,npool,fpool,name=="train"), num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

def build_model():
    """
    Hello Edge Small (S) DS-CNN
    Zhang et al. 2018, arXiv:1711.07128, Table 7
    C(64,10,4,2,2) - DSC(64,3,1)×4 - AvgPool(25,5) - Dense(2) - Softmax
    Input: (49, 10, 1)
    """
    inputs = keras.Input(shape=(49, 10, 1))

    # Conv1: (49,10,1) → (25,5,64)
    x = layers.Conv2D(64, kernel_size=(10, 4), strides=(2, 2),
                      padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    def ds_conv(inp, filters):
        x = layers.DepthwiseConv2D(3, padding='same', use_bias=False)(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # 4 × DS-Conv: (25,5,64) → (25,5,64)
    x = ds_conv(x, 64)
    x = ds_conv(x, 64)
    x = ds_conv(x, 64)
    x = ds_conv(x, 64)

    # AvgPool: (25,5,64) → (1,1,64)
    x = layers.AveragePooling2D(pool_size=(25, 5))(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(2, activation='softmax')(x)

    return keras.Model(inputs, outputs)


# _load_wav_fixed 從 kws/features.py import（見檔案開頭）—— 與 _read() 邏輯一致


def evaluate_external_test_set(tflite_path, labels_csv, database_dir, output_dir):
    """After training, run the final int8-quantized tflite model over the real-world
    test set (225 files) and write per-file results plus an accuracy summary."""
    if not os.path.exists(tflite_path):
        print(f"[External Eval] tflite model not found: {tflite_path}, skipping.")
        return None
    if not os.path.exists(labels_csv):
        print(f"[External Eval] labels.csv not found: {labels_csv}, skipping.")
        return None

    print(f"\n=== [External Eval] Evaluating model on real-world test set: {labels_csv} ===")

    interpreter = _features_load_tflite(tflite_path)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    in_scale, in_zp = in_detail["quantization"]
    out_scale, out_zp = out_detail["quantization"]

    with open(labels_csv, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    results = []
    for row in rows:
        rel_path = row["relative_path"].replace("/", os.sep)
        audio_path = os.path.join(database_dir, rel_path)
        true_label = row["label"].strip().lower()
        category = row.get("category", "")

        if not os.path.exists(audio_path):
            print(f"  [skip] file not found: {audio_path}")
            continue

        wav = _load_wav_fixed(audio_path)
        mfcc = wav_to_mfcc(wav).numpy()[np.newaxis, ...]  # (1, 49, 10, 1)

        q_in = np.round(mfcc / in_scale + in_zp)
        q_in = np.clip(q_in, -128, 127).astype(np.int8)

        interpreter.set_tensor(in_detail["index"], q_in)
        interpreter.invoke()
        q_out = interpreter.get_tensor(out_detail["index"])[0]
        probs = (q_out.astype(np.float32) - out_zp) * out_scale

        pred_label = "positive" if int(np.argmax(probs)) == 1 else "negative"
        results.append({
            "filename": row["filename"],
            "relative_path": row["relative_path"],
            "true_label": true_label,
            "category": category,
            "predicted_label": pred_label,
            "negative_prob": float(probs[0]),
            "positive_prob": float(probs[1]),
            "correct": pred_label == true_label,
        })

    results_csv = os.path.join(output_dir, "external_test_results.csv")
    with open(results_csv, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "filename", "relative_path", "true_label", "category",
            "predicted_label", "negative_prob", "positive_prob", "correct"])
        writer.writeheader()
        writer.writerows(results)

    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    overall_acc = n_correct / n if n else 0.0

    tp = sum(1 for r in results if r["true_label"] == "positive" and r["predicted_label"] == "positive")
    fn = sum(1 for r in results if r["true_label"] == "positive" and r["predicted_label"] == "negative")
    tn = sum(1 for r in results if r["true_label"] == "negative" and r["predicted_label"] == "negative")
    fp = sum(1 for r in results if r["true_label"] == "negative" and r["predicted_label"] == "positive")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    lines = [
        f"Total evaluated: {n}",
        f"Overall accuracy: {overall_acc:.4f} ({n_correct}/{n})",
        f"Confusion matrix: TP={tp} FN={fn} TN={tn} FP={fp}",
        f"Precision={precision:.4f}  Recall(TPR)={recall:.4f}  FAR={far:.4f}  F1={f1:.4f}",
        "",
        "Per-category breakdown:",
    ]
    per_category = {}
    for cat in sorted(set(r["category"] for r in results)):
        sub = [r for r in results if r["category"] == cat]
        sub_n = len(sub)
        sub_correct = sum(1 for r in sub if r["correct"])
        sub_acc = sub_correct / sub_n if sub_n else 0.0
        per_category[cat] = {"n": sub_n, "correct": sub_correct, "acc": sub_acc}
        lines.append(f"  {cat:10s}: n={sub_n:4d}  acc={sub_acc:.4f} ({sub_correct}/{sub_n})")

    # --- misclassified files (so you can diff across runs / spot the clips
    #     that fail no matter what you change -- see analyze_persistent_failures.py)
    fn_rows = sorted((r for r in results if r["true_label"] == "positive"
                      and r["predicted_label"] == "negative"),
                     key=lambda r: r["positive_prob"])
    fp_rows = sorted((r for r in results if r["true_label"] == "negative"
                      and r["predicted_label"] == "positive"),
                     key=lambda r: -r["positive_prob"])
    fn_files = [r["relative_path"] for r in fn_rows]
    fp_files = [r["relative_path"] for r in fp_rows]

    lines.append("")
    lines.append(f"Missed wake words (FN={len(fn_rows)}) -- positive clip scored as negative:")
    for r in fn_rows:
        lines.append(f"  {r['positive_prob']:.3f}  {r['relative_path']}")
    lines.append("")
    lines.append(f"False triggers (FP={len(fp_rows)}) -- negative clip scored as positive:")
    for r in fp_rows:
        lines.append(f"  {r['positive_prob']:.3f}  {r['relative_path']}  [{r['category']}]")

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = os.path.join(output_dir, "external_test_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary_text + "\n")

    # plain list of just the wrong files, one per line, for quick cross-run diffing
    mis_path = os.path.join(output_dir, "external_test_misclassified.txt")
    with open(mis_path, "w", encoding="utf-8") as fh:
        for p in fn_files:
            fh.write(f"FN\t{p}\n")
        for p in fp_files:
            fh.write(f"FP\t{p}\n")

    print(f"\n[External Eval] Per-file results: {results_csv}")
    print(f"[External Eval] Summary report: {summary_path}")
    print(f"[External Eval] Misclassified list: {mis_path}")

    return {
        "n": n, "n_correct": n_correct, "overall_acc": overall_acc,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "precision": precision, "recall": recall, "far": far, "f1": f1,
        "per_category": per_category,
        "results_csv": results_csv, "summary_path": summary_path,
        "fn_files": fn_files, "fp_files": fp_files,
    }


def write_training_report(output_dir, history, tflite_path, sample_stats,
                           elapsed_seconds, eval_summary):
    """Consolidate this run's config, dataset composition, training curve and
    (if available) real-world test accuracy into a single training_report.txt."""
    lines = [
        "=== KWS Training Report ===",
        f"Run dir: {output_dir}",
        f"Command label: {args.command_labels}",
        f"Model mode: {args.model_mode}",
        f"Seed: {args.seed}",
        "",
        "--- Hyperparameters ---",
        f"learning_rate={args.learning_rate}  batch_size={args.batch_size}  "
        f"max_epochs={args.max_epochs}  patience={args.patience}",
        f"target_unknown_count={args.target_unknown_count}  "
        f"ratio_command_to_unknown={args.ratio_command_to_unknown}",
        f"ratio_gsc={args.ratio_gsc}  ratio_near_miss={args.ratio_near_miss}  "
        f"ratio_generic={args.ratio_generic}  "
        f"ratio_fma={args.ratio_fma}  ratio_pure_noise={args.ratio_pure_noise}",
        f"mix_prob_noise={args.mix_prob_noise}  mix_prob_fma={args.mix_prob_fma}",
        f"snr_min={args.snr_min}  snr_max={args.snr_max}",
        f"input_norm={args.input_norm}  input_norm_target={args.input_norm_target}  "
        f"input_norm_floor={args.input_norm_floor}",
        f"aug_lowpass_prob={args.aug_lowpass_prob}  "
        f"aug_lowpass_hz=[{args.aug_lowpass_min_hz},{args.aug_lowpass_max_hz}]",
        f"aug_reverb_prob={args.aug_reverb_prob}  "
        f"aug_reverb_t60=[{args.aug_reverb_t60_min},{args.aug_reverb_t60_max}]  "
        f"aug_reverb_mix=[{args.aug_reverb_mix_min},{args.aug_reverb_mix_max}]",
        "",
        "--- Dataset sampling (per split: requested vs. actually used, no replacement) ---",
    ]
    for split_name, cats in sample_stats.items():
        lines.append(f"[{split_name}]")
        for cat, stat in cats.items():
            flag = "  (undersampled: pool exhausted)" if stat["used"] < stat["requested"] else ""
            lines.append(f"  {cat:10s}: requested={stat['requested']:5d}  "
                          f"pool={stat['available']:5d}  used={stat['used']:5d}{flag}")

    lines.append("")
    lines.append("--- Training result ---")
    h = history.history
    epochs_run = len(h.get("loss", []))
    stopped_early = epochs_run < args.max_epochs
    lines.append(f"Epochs run: {epochs_run} / {args.max_epochs} configured"
                  + ("  (early stopped)" if stopped_early else ""))
    if "val_loss" in h and h["val_loss"]:
        best_epoch = int(np.argmin(h["val_loss"]))
        lines.append(f"Best epoch (min val_loss): {best_epoch + 1}")
        lines.append(f"  val_loss={h['val_loss'][best_epoch]:.4f}  "
                      f"val_accuracy={h['val_accuracy'][best_epoch]:.4f}")
    if h.get("loss") and h.get("accuracy"):
        lines.append(f"Final train_loss={h['loss'][-1]:.4f}  "
                      f"train_accuracy={h['accuracy'][-1]:.4f}")
    lines.append(f"Training wall time: {elapsed_seconds/60:.1f} min")

    lines.append("")
    lines.append("--- Exported model ---")
    if os.path.exists(tflite_path):
        size_kb = os.path.getsize(tflite_path) / 1024
        lines.append(f"tflite: {tflite_path}  ({size_kb:.1f} KB)")
    else:
        lines.append("tflite: (not found)")

    lines.append("")
    lines.append("--- Real-world test set (test_database) ---")
    if eval_summary is None:
        lines.append("Not run (skipped or test_database/labels.csv unavailable).")
    else:
        lines.append(f"Overall accuracy: {eval_summary['overall_acc']:.4f} "
                      f"({eval_summary['n_correct']}/{eval_summary['n']})")
        lines.append(f"Confusion matrix: TP={eval_summary['tp']} FN={eval_summary['fn']} "
                      f"TN={eval_summary['tn']} FP={eval_summary['fp']}")
        lines.append(f"Precision={eval_summary['precision']:.4f}  "
                      f"Recall(TPR)={eval_summary['recall']:.4f}  "
                      f"FAR={eval_summary['far']:.4f}  F1={eval_summary['f1']:.4f}")
        for cat, stat in eval_summary["per_category"].items():
            lines.append(f"  {cat:10s}: n={stat['n']:4d}  acc={stat['acc']:.4f} "
                          f"({stat['correct']}/{stat['n']})")
        if eval_summary.get("fn_files"):
            lines.append(f"Missed wake words (FN): {', '.join(eval_summary['fn_files'])}")
        if eval_summary.get("fp_files"):
            lines.append(f"False triggers (FP): {', '.join(eval_summary['fp_files'])}")
        lines.append(f"Full per-file results: {eval_summary['results_csv']}")

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text + "\n")
    print(f"\n[Training Report] saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    print(f"Checking dataset status...")
    print(f"Positive samples: {len(ALL_SPLITS['pos']['train'])}")
    print(f"Negative pools (train): GSC:{len(ALL_SPLITS['gsc']['train'])}, "
          f"near_miss:{len(ALL_SPLITS['near_miss']['train'])}, "
          f"generic:{len(ALL_SPLITS['generic']['train'])}, "
          f"FMA:{len(ALL_SPLITS['fma']['train'])}, noise:{len(ALL_SPLITS['noise']['train'])}")

    if args.num_test_samples > 0:
        test_mix_dir = os.path.join(OUTPUT_DIR, "test_mix_audio")
        os.makedirs(test_mix_dir, exist_ok=True)
        test_candidates = ALL_SPLITS["pos"]["train"]
        num_to_gen = min(len(test_candidates), args.num_test_samples)
        sample_paths = random.sample(test_candidates, num_to_gen)
        print(f"Generating {num_to_gen} test audio samples in: {test_mix_dir}")
        for i, path in enumerate(sample_paths):
            try:
                audio_binary = tf.io.read_file(path)
                wav, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
                wav = tf.squeeze(wav, axis=-1)
                curr_len = tf.shape(wav)[0]
                wav = tf.cond(curr_len < SAMPLES, lambda: tf.pad(wav, [[0, SAMPLES - curr_len]]), lambda: wav[:SAMPLES])
                mixed_wav = wav
                mix_info = "clean"
                
                # [Bugfix] 將 args.mix_prob_positive 修正為 args.mix_prob_noise
                if random.random() < args.mix_prob_noise and len(ALL_SPLITS["noise"]["train"]) > 0:
                    n_path = random.choice(ALL_SPLITS["noise"]["train"])
                    n_binary = tf.io.read_file(n_path)
                    noise, _ = tf.audio.decode_wav(n_binary, desired_channels=1)
                    noise = tf.squeeze(noise, -1)
                    n_len = tf.shape(noise)[0]
                    noise = tf.cond(n_len < SAMPLES, lambda: tf.pad(noise, [[0, SAMPLES - n_len]]), lambda: noise[:SAMPLES])
                    snr_db = random.uniform(args.snr_min, args.snr_max)
                    scale = tf.pow(10.0, -snr_db / 20.0) * (tf.math.reduce_std(wav) / (tf.math.reduce_std(noise) + 1e-6))
                    mixed_wav = wav + (noise * scale)
                    mix_info = f"snr_{snr_db:.1f}"
                    
                out_name = f"test_{i}_{args.command_labels}_{mix_info}.wav"
                out_path = os.path.join(test_mix_dir, out_name)
                final_audio = tf.clip_by_value(mixed_wav, -1.0, 1.0)
                encoded_wav = tf.audio.encode_wav(tf.expand_dims(final_audio, -1), SR)
                tf.io.write_file(out_path, encoded_wav)
            except Exception as e:
                print(f"Skip generating test sample {i}: {e}")

    # Dataset 建立
    train_ds = create_ds("train").map(lambda x,y: (x, tf.one_hot(y, 2)))
    val_ds = create_ds("val").map(lambda x,y: (x, tf.one_hot(y, 2)))
    
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.AdamW(args.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(os.path.join(OUTPUT_DIR, "best.keras"), save_best_only=True),
        EarlyStopping(patience=args.patience, restore_best_weights=True),
        BackupAndRestore(backup_dir=BACKUP_DIR)
    ]

    print(f"Starting/Resuming training for command: {args.command_labels}")
    train_start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.max_epochs, callbacks=callbacks)
    train_elapsed = time.time() - train_start

    print("Exporting INT8 quantized model...")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    # 量化校準資料生成
    def rep_data_gen():
        for x, _ in train_ds.unbatch().batch(1).take(100):
            yield [x]
            
    conv.representative_dataset = rep_data_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type, conv.inference_output_type = tf.int8, tf.int8
    
    tflite_path = os.path.join(OUTPUT_DIR, "model_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(conv.convert())
    print(f"Done! Quantized model saved to {tflite_path}")

    # --- metadata.json：inference/ 要靠這個知道特徵參數 / 正類 index / 建議 threshold ---
    _interp = _features_load_tflite(tflite_path)
    _interp.allocate_tensors()
    _in, _out = _interp.get_input_details()[0], _interp.get_output_details()[0]
    metadata = {
        "wake_word": args.wake_word or args.command_labels.replace("_", " "),
        "label": args.command_labels,
        "model_mode": args.model_mode,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "audio": {"sample_rate": SR, "duration_sec": DURATION, "samples": SAMPLES,
                  "frame_step": FRAME_STEP, "fft_length": FFT_LENGTH,
                  "num_mels": NUM_MELS, "num_mfcc": NUM_MFCC,
                  "mel_low_hz": 20.0, "mel_high_hz": 8000.0},
        "input_norm": {"mode": args.input_norm, "target": args.input_norm_target,
                       "floor": args.input_norm_floor},
        "tflite_io": {
            "input_shape": [int(x) for x in _in["shape"]],
            "input_scale": float(_in["quantization"][0]),
            "input_zero_point": int(_in["quantization"][1]),
            "output_scale": float(_out["quantization"][0]),
            "output_zero_point": int(_out["quantization"][1]),
        },
        "positive_class_index": 1,      # 訓練時 label=1 = 喚醒詞（見 create_ds）
        "suggested_threshold": args.suggested_threshold,
        "dataset_sampling": SAMPLE_STATS,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"metadata.json saved to {OUTPUT_DIR}/metadata.json")

    eval_summary = None
    if not args.skip_external_eval:
        eval_summary = evaluate_external_test_set(
            tflite_path=os.path.join(OUTPUT_DIR, "model_int8.tflite"),
            labels_csv=TEST_LABELS_CSV,
            database_dir=TEST_DB_DIR,
            output_dir=OUTPUT_DIR,
        )

    write_training_report(
        output_dir=OUTPUT_DIR,
        history=history,
        tflite_path=os.path.join(OUTPUT_DIR, "model_int8.tflite"),
        sample_stats=SAMPLE_STATS,
        elapsed_seconds=train_elapsed,
        eval_summary=eval_summary,
    )