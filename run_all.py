# -*- coding: utf-8 -*-
"""
run_all.py
==========
全自動化地端關鍵詞偵測（KWS / keyword spotting）訓練 pipeline —— 純 Python、純地端。

使用者唯一要手動給的就是喚醒詞：

    venv\\Scripts\\python.exe run_all.py --wake-word "hey assistant"

流程：
    [1] 檢查 / 建立 venv 與套件
    [2] 準備資料集（GSC / FMA / noise / piper 模型 / 共用 generic_speech 池，缺就自動補）
    [3] 用 TTS 生成這個喚醒詞的正樣本 + 近音負樣本（kws.generate）
    [4] 建立時間戳記 run 目錄（models/<label>_<timestamp>/）
    [5] 訓練 DS-CNN（kws.train）→ model_int8.tflite + metadata.json + training_report.txt

跑完後用 inference/infer.py 或 inference/mic_demo.py 測試模型。

參數：
    --wake-word "..."      （必填）喚醒詞文字
    --label xxx            資料夾標籤（預設 = wake-word 空白換底線）
    --smoke               小樣本煙霧測試，驗證整條路徑
    --pos-count / --near-miss-count / --tts-backend / --voice-pack
    --test-database <dir>  訓練後對此測試集算 precision/recall（需含 labels.csv）
    --skip-gen            跳過 Step 3（樣本已生好時）
    --skip-train          只生樣本，不訓練
"""

import argparse
import datetime as dt
import os
import subprocess
import sys

# 讓中文輸出在 Windows 主控台 / 被導向到檔案時都用 UTF-8
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from kws import paths

ROOT = str(paths.REPO_ROOT)
VENV_DIR = os.path.join(ROOT, "venv")
PYTHON_EXE = os.path.join(VENV_DIR, "Scripts", "python.exe")
if not os.path.exists(PYTHON_EXE):                       # 非 Windows fallback
    PYTHON_EXE = os.path.join(VENV_DIR, "bin", "python")
REQUIRE_PKGS = ["numpy", "tensorflow", "librosa", "soundfile", "piper-tts", "pronouncing"]


def _is_cjk(s):
    return any("⺀" <= c <= "鿿" or "豈" <= c <= "﫿" for c in s)


def sh(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, env=env, **kw)
    if r.returncode != 0:
        raise SystemExit(f"[run_all] 指令失敗（exit {r.returncode}）：{cmd}")
    return r


# ======================================================================================
# Step 1 — venv + 套件
# ======================================================================================
def step1_env(cfg):
    print("\n=== [Step 1] 檢查虛擬環境與套件 ===")
    global PYTHON_EXE
    if not os.path.exists(PYTHON_EXE):
        print(f"建立 venv：{VENV_DIR}")
        sh([sys.executable, "-m", "venv", VENV_DIR])
        cand = [os.path.join(VENV_DIR, "Scripts", "python.exe"),
                os.path.join(VENV_DIR, "bin", "python")]
        PYTHON_EXE = next((c for c in cand if os.path.exists(c)), PYTHON_EXE)

    have = subprocess.run(
        [PYTHON_EXE, "-c", "import tensorflow, librosa, piper, pronouncing, pypinyin"],
        capture_output=True).returncode == 0
    if not have:
        print("安裝核心套件：", ", ".join(REQUIRE_PKGS))
        sh([PYTHON_EXE, "-m", "pip", "install", "--upgrade", "pip"])
        sh([PYTHON_EXE, "-m", "pip", "install", *REQUIRE_PKGS])
        sh([PYTHON_EXE, "-m", "pip", "install", "-r",
            os.path.join(ROOT, "datasets", "requirements.txt")])
    else:
        print("核心套件已安裝，跳過。")

    # edge-tts：--tts-backend edge，或中文喚醒詞（正樣本用 edge）時需要
    if cfg["TTS_BACKEND"] == "edge" or cfg["_ZH"]:
        edge_ok = subprocess.run([PYTHON_EXE, "-c", "import edge_tts"],
                                 capture_output=True).returncode == 0
        if not edge_ok:
            print("安裝 edge 後端套件：edge-tts")
            sh([PYTHON_EXE, "-m", "pip", "install", "-r",
                os.path.join(ROOT, "requirements-edge.txt")])


# ======================================================================================
# Step 2 — 資料集
# ======================================================================================
def step2_datasets(cfg):
    print("\n=== [Step 2] 準備資料集 ===")
    need = []
    checks = {
        "speech_commands": paths.SPEECH_COMMANDS,
        "music": paths.MUSIC,
        "noise": paths.NOISE,
    }
    for name, p in checks.items():
        n = sum(len(fs) for _, _, fs in os.walk(p)) if p.is_dir() else 0
        if n == 0:
            need.append(name)
            print(f"  {name}: 缺")
        else:
            print(f"  {name}: {n} 檔  OK")

    # 負樣本一律用 piper（連中文近音負樣本也是）→ 依 PIPER_ENGINE 檢查 onnx 或 torch 那份
    if cfg["PIPER_ENGINE"] == "torch":
        pt = paths.DATASETS / "_models" / "piper_sample_generator" / "models" / f"{cfg['PIPER_VOICE']}.pt"
        vpsg_py = paths.REPO_ROOT / "venv-psg" / "Scripts" / "python.exe"
        if pt.exists() and vpsg_py.exists():
            print(f"  piper 模型 (torch): OK ({pt.name})")
        else:
            need.append("piper-torch")
            print(f"  piper 模型 (torch): 缺 → 會建 venv-psg + 下載 .pt")
    else:
        onnx = paths.DATASETS / "_models" / "piper" / f"{cfg['PIPER_VOICE']}.onnx"
        if onnx.exists():
            print(f"  piper 模型 (onnx): OK ({onnx.name})")
        else:
            need.append("piper")
            print(f"  piper 模型 (onnx): 缺 ({onnx.name})")

    if need:
        print(f"  → 呼叫 datasets/prepare.py 補齊：{', '.join(need)}")
        sh([PYTHON_EXE, os.path.join(ROOT, "datasets", "prepare.py"), *need], cwd=ROOT)

    # 共用的通用英文語音負樣本池（datasets/generic_speech/）—— 只生一次，全部喚醒詞共用
    n_gen = _count_wavs(paths.GENERIC_SPEECH)
    if n_gen >= cfg["GENERIC_COUNT"]:
        print(f"  generic_speech: {n_gen} 檔  OK")
    else:
        print(f"  generic_speech: {n_gen}/{cfg['GENERIC_COUNT']} → 生成中（一次性）")
        sh([PYTHON_EXE, "-m", "kws.generate", "--kind", "generic",
            "--generic-count", str(cfg["GENERIC_COUNT"]),
            "--piper-engine", cfg["PIPER_ENGINE"],
            "--test-frac", str(cfg["TEST_FRAC"]), "--seed", str(cfg["SEED"]),
            "--concurrency", str(cfg["PIPER_WORKERS"])], cwd=ROOT)


def _count_wavs(p):
    return sum(len([f for f in fs if f.lower().endswith(".wav")])
               for _, _, fs in os.walk(p)) if p.is_dir() else 0


# ======================================================================================
# Step 3 — 生成客製化 TTS 樣本（正樣本 + 近音負樣本）
# ======================================================================================
def step3_generate(args, cfg):
    zh = _is_cjk(args.wake_word)
    backend = cfg["TTS_BACKEND"]
    if zh:
        print(f"\n=== [Step 3] 生成樣本（中文：正=edge-tts {cfg['ZH_VOICE_PACK']}，近音負=piper）===")
    else:
        print(f"\n=== [Step 3] 生成樣本（後端：{backend}）===")
    cmd = [
        PYTHON_EXE, "-m", "kws.generate",
        "--kind", "both",
        "--wake-word", args.wake_word,
        "--label", args.label,
        "--pos-count", str(cfg["POS_COUNT"]),
        "--neg-count", str(cfg["NEAR_MISS_COUNT"]),
        "--tts-backend", backend,
        "--piper-engine", cfg["PIPER_ENGINE"],
        "--test-frac", str(cfg["TEST_FRAC"]),
        "--seed", str(cfg["SEED"]),
        "--concurrency", str(cfg["PIPER_WORKERS"] if (backend == "piper" or zh) else cfg["TTS_CONCURRENCY"]),
    ]
    if zh:
        cmd += ["--voice-pack", cfg["ZH_VOICE_PACK"]]
    elif backend == "edge":
        cmd += ["--voice-pack", cfg["VOICE_PACK"]]
    if cfg.get("PIPER_MAX_SPEAKERS"):
        cmd += ["--piper-max-speakers", str(cfg["PIPER_MAX_SPEAKERS"])]
    sh(cmd, cwd=ROOT)


# ======================================================================================
# Step 5 — 訓練
# ======================================================================================
def step5_train(args, cfg, run_dir):
    print("\n=== [Step 5] 訓練 DS-CNN ===")
    cmd = [
        PYTHON_EXE, "-m", "kws.train",
        "--model_mode", cfg["MODEL_MODE"],
        "--run_dir", run_dir,
        "--wake_word", args.wake_word,
        "--seed", str(cfg["SEED"]),
        "--learning_rate", str(cfg["LEARNING_RATE"]),
        "--batch_size", str(cfg["BATCH_SIZE"]),
        "--max_epochs", str(cfg["MAX_EPOCHS"]),
        "--patience", str(cfg["PATIENCE"]),
        "--target_unknown_count", str(cfg["TARGET_UNKNOWN_COUNT"]),
        "--ratio_gsc", str(cfg["RATIO_GSC"]),
        "--ratio_near_miss", str(cfg["RATIO_NEAR_MISS"]),
        "--ratio_generic", str(cfg["RATIO_GENERIC"]),
        "--ratio_fma", str(cfg["RATIO_FMA"]),
        "--ratio_pure_noise", str(cfg["RATIO_PURE_NOISE"]),
        "--mix_prob_noise", str(cfg["MIX_PROB_NOISE"]),
        "--mix_prob_fma", str(cfg["MIX_PROB_FMA"]),
        "--ratio_command_to_unknown", str(cfg["RATIO_COMMAND_TO_UNKNOWN"]),
        "--command_labels", args.label,
        "--num_test_samples", str(cfg["NUM_TEST_SAMPLES"]),
        "--suggested_threshold", str(cfg["SUGGESTED_THRESHOLD"]),
        "--snr_min", str(cfg["SNR_MIN"]),
        "--snr_max", str(cfg["SNR_MAX"]),
        "--input_norm", cfg["INPUT_NORM"],
        "--input_norm_target", str(cfg["INPUT_NORM_TARGET"]),
        "--input_norm_floor", str(cfg["INPUT_NORM_FLOOR"]),
        "--aug_lowpass_prob", str(cfg["AUG_LOWPASS_PROB"]),
        "--aug_lowpass_min_hz", str(cfg["AUG_LOWPASS_MIN_HZ"]),
        "--aug_lowpass_max_hz", str(cfg["AUG_LOWPASS_MAX_HZ"]),
        "--aug_reverb_prob", str(cfg["AUG_REVERB_PROB"]),
        "--aug_reverb_t60_min", str(cfg["AUG_REVERB_T60_MIN"]),
        "--aug_reverb_t60_max", str(cfg["AUG_REVERB_T60_MAX"]),
        "--aug_reverb_mix_min", str(cfg["AUG_REVERB_MIX_MIN"]),
        "--aug_reverb_mix_max", str(cfg["AUG_REVERB_MIX_MAX"]),
    ]
    if args.test_database:
        cmd += ["--test_database_dir", args.test_database,
                "--test_labels_csv", os.path.join(args.test_database, "labels.csv")]
    elif cfg["SKIP_EXTERNAL_EVAL"]:
        cmd += ["--skip_external_eval"]
    sh(cmd, cwd=ROOT)


# ======================================================================================
def build_cfg(args):
    cfg = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    cfg["_ZH"] = _is_cjk(args.wake_word)
    if args.smoke:
        cfg.update(config.SMOKE_OVERRIDES)
        print("[smoke] 套用小樣本覆蓋：", config.SMOKE_OVERRIDES)
    if args.pos_count is not None:
        cfg["POS_COUNT"] = args.pos_count
    if args.near_miss_count is not None:
        cfg["NEAR_MISS_COUNT"] = args.near_miss_count
    if args.voice_pack:
        cfg["VOICE_PACK"] = args.voice_pack
    if args.tts_backend:
        cfg["TTS_BACKEND"] = args.tts_backend
    if args.piper_engine:
        cfg["PIPER_ENGINE"] = args.piper_engine
    return cfg


def main():
    ap = argparse.ArgumentParser(description="全自動化地端喚醒詞訓練 pipeline")
    ap.add_argument("--wake-word", required=True, help='喚醒詞，例：--wake-word "hey assistant"')
    ap.add_argument("--label", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pos-count", type=int, default=None)
    ap.add_argument("--near-miss-count", type=int, default=None, help="近音負樣本生成池大小")
    ap.add_argument("--piper-engine", choices=["onnx", "torch"], default=None,
                    help="onnx（CPU，預設）/ torch（NVIDIA GPU，快很多，要先 prepare.py piper-torch）")
    ap.add_argument("--tts-backend", choices=["piper", "edge"], default=None,
                    help="TTS 後端（預設讀 config.TTS_BACKEND = piper）")
    ap.add_argument("--voice-pack", default=None, help="edge 後端用：zh-tw / zh-cn / en …")
    ap.add_argument("--test-database", default=None)
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    args.label = (args.label or args.wake_word).strip().replace(" ", "_")

    cfg = build_cfg(args)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.label}_{ts}"
    run_dir = str(paths.MODELS / run_name)

    print(f"=== Run: {run_name} ===")
    print(f"喚醒詞: {args.wake_word!r}   label: {args.label}   smoke: {args.smoke}")

    step1_env(cfg)
    step2_datasets(cfg)
    if not args.skip_gen:
        step3_generate(args, cfg)
    else:
        print("\n=== [Step 3] 跳過（--skip-gen）===")

    if args.skip_train:
        print("\n=== [Step 4-5] 跳過（--skip-train）===")
        return

    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== [Step 4] Run 目錄：{run_dir} ===")
    step5_train(args, cfg, run_dir)

    print("\n=== 全部完成 ===")
    print(f"輸出資料夾：{run_dir}")
    for f in sorted(os.listdir(run_dir)):
        print("  -", f)
    print(f"\n測試模型：")
    print(f'  {PYTHON_EXE} inference\\infer.py --input datasets\\tts\\{args.label}\\positive\\test')
    print(f'  {PYTHON_EXE} inference\\mic_demo.py')


if __name__ == "__main__":
    main()
