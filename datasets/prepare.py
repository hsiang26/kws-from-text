# -*- coding: utf-8 -*-
"""
datasets/prepare.py
===================
把 pipeline 需要、但因為體積/授權不放進 git 的第三方資料集抓下來、轉成 44.1kHz：

    speech_commands   Google Speech Commands v0.02（CC BY 4.0）—— 負樣本用的一般語音
    music             FMA-small 片段（逐曲 CC 授權）—— 音樂負樣本 + 訓練時混入
    noise             datasets/make_noise.py 合成的環境噪音
    piper             Piper ONNX 模型 en_US-libritts_r-medium（預設 TTS 引擎用）
    generic           共用的通用英文語音負樣本（piper 合成，一次生給所有喚醒詞用）
    piper-torch       （選配）venv-psg + torch(CUDA) + piper-sample-generator + .pt 模型，
                      給 PIPER_ENGINE='torch' 的 GPU 批次生成用。不在預設清單裡。

    python datasets/prepare.py                    # 上面前 5 個
    python datasets/prepare.py speech_commands    # 只準備某幾個
    python datasets/prepare.py noise --per-type 20
    python datasets/prepare.py generic --count 10000
    python datasets/prepare.py piper-torch        # GPU 生成的環境（~3GB 下載）

授權 / 出處見 README.md 的「致謝與授權」與 datasets/README.md。
"""

import argparse
import os
import subprocess
import sys
import tarfile
import urllib.request

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kws import paths
from kws.resample import resample_tree

GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
FMA_HF = ("rudraml/fma", "small")
FMA_CLIPS = 120
TARGET_SR = 44100

PIPER_VOICES = {
    "en_US-libritts_r-medium":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/",
}

PSG_REPO = "https://github.com/rhasspy/piper-sample-generator"
PSG_PT = {  # PyTorch .pt 生成器模型（GitHub release）
    "en_US-libritts_r-medium":
        "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
}
PSG_TORCH_INDEX = "https://download.pytorch.org/whl/cu124"  # NVIDIA CUDA 12.4 wheel


# ------------------------------------------------------------------ speech commands
def prepare_speech_commands():
    dst = paths.SPEECH_COMMANDS
    if dst.is_dir() and sum(len(f) for _, _, f in os.walk(dst)) > 1000:
        print(f"[speech_commands] 已存在，跳過（{dst}）")
        return
    raw = paths.DATASETS / "_gsc_raw"
    raw.mkdir(parents=True, exist_ok=True)
    if not any(raw.iterdir()):
        print(f"[speech_commands] 下載 {GSC_URL} (~2.3GB)…")
        tmp = raw / "speech_commands_v0.02.tar.gz"
        _download(GSC_URL, tmp)
        print("[speech_commands] 解壓…")
        with tarfile.open(tmp) as tf:
            tf.extractall(raw, filter="data")
        tmp.unlink()
    # 只留「詞」資料夾，跳過 _background_noise_ 和說明檔
    for sub in sorted(p for p in raw.iterdir() if p.is_dir()):
        if sub.name.startswith("_"):
            continue
        resample_tree(sub, dst / sub.name, TARGET_SR)
    print(f"[speech_commands] 完成 → {dst}")
    print("   （原始下載留在 datasets/_gsc_raw/，可自行刪掉省空間）")


# ------------------------------------------------------------------ music (FMA)
def prepare_music():
    dst = paths.MUSIC
    have = len([f for f in dst.glob("*.wav")]) if dst.is_dir() else 0
    if have >= FMA_CLIPS:
        print(f"[music] 已有 {have} 段，跳過")
        return
    try:
        import numpy as np
        import scipy.io.wavfile
        import datasets as hfds
    except ImportError:
        sys.exit("[music] 需要 `pip install datasets scipy`（見 datasets/requirements.txt）")

    dst.mkdir(parents=True, exist_ok=True)
    print(f"[music] 從 Hugging Face 串流 {FMA_HF[0]}（{FMA_HF[1]}），取 {FMA_CLIPS} 段…")
    ds = hfds.load_dataset(FMA_HF[0], name=FMA_HF[1], split="train", streaming=True)
    ds = iter(ds.cast_column("audio", hfds.Audio(sampling_rate=TARGET_SR)))
    saved, errs = have, 0
    while saved < FMA_CLIPS:
        try:
            row = next(ds)
        except StopIteration:
            break
        try:
            name = os.path.basename(row["audio"]["path"]).rsplit(".", 1)[0] + ".wav"
            out = dst / name
            if out.exists():
                continue
            scipy.io.wavfile.write(str(out), TARGET_SR,
                                   (row["audio"]["array"] * 32767).astype(np.int16))
            saved += 1
        except Exception as e:  # noqa: BLE001
            errs += 1
            if errs <= 5:
                print(f"  [skip] {e}")
    print(f"[music] 完成，共 {saved} 段 → {dst}")


# ------------------------------------------------------------------ noise
def prepare_noise(extra_args):
    print("[noise] 呼叫 make_noise.py…")
    subprocess.run([sys.executable, os.path.join(HERE, "make_noise.py"),
                    "--out", str(paths.NOISE), *extra_args], check=True)


# ------------------------------------------------------------------ 共用通用語音負樣本
def prepare_generic(count):
    n = sum(len([f for f in fs if f.endswith(".wav")])
            for _, _, fs in os.walk(paths.GENERIC_SPEECH)) if paths.GENERIC_SPEECH.is_dir() else 0
    if n >= count:
        print(f"[generic] 已有 {n} 檔，跳過")
        return
    print(f"[generic] {n}/{count} → 用 piper 生成（一次性）…")
    subprocess.run([sys.executable, "-m", "kws.generate", "--kind", "generic",
                    "--generic-count", str(count)], cwd=os.path.dirname(HERE), check=True)


# ------------------------------------------------------------------ piper-torch（GPU 生成）
def prepare_piper_torch(voice="en_US-libritts_r-medium"):
    """建 venv-psg（torch+CUDA + piper-sample-generator）+ clone repo + 下載 .pt。"""
    repo_root = os.path.dirname(HERE)
    venv = os.path.join(repo_root, "venv-psg")
    py = os.path.join(venv, "Scripts", "python.exe")
    if not os.path.exists(py):
        py = os.path.join(venv, "bin", "python")

    if not os.path.exists(py):
        print(f"[piper-torch] 建立 venv-psg：{venv}")
        subprocess.run([sys.executable, "-m", "venv", venv], check=True)
        py = os.path.join(venv, "Scripts", "python.exe")
        if not os.path.exists(py):
            py = os.path.join(venv, "bin", "python")
        subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
        print(f"[piper-torch] 裝 torch (CUDA) —— 從 {PSG_TORCH_INDEX}（~2.5GB）…")
        subprocess.run([py, "-m", "pip", "install", "torch", "torchaudio",
                        "--index-url", PSG_TORCH_INDEX], check=True)
        print("[piper-torch] 裝 piper-sample-generator + 相依…")
        subprocess.run([py, "-m", "pip", "install", "piper-sample-generator"], check=True)
    else:
        print("[piper-torch] venv-psg 已存在，跳過安裝")

    clone = paths.DATASETS / "_models" / "piper_sample_generator"
    if not (clone / "piper_train").is_dir():
        clone.parent.mkdir(parents=True, exist_ok=True)
        print(f"[piper-torch] git clone {PSG_REPO} …（要 piper_train/ 那份原始碼）")
        subprocess.run(["git", "clone", "--depth", "1", PSG_REPO, str(clone)], check=True)

    pt = clone / "models" / f"{voice}.pt"
    if pt.exists() and pt.stat().st_size > 1_000_000:
        print(f"[piper-torch] {pt.name} 已存在，跳過")
    else:
        url = PSG_PT.get(voice)
        if not url:
            sys.exit(f"[piper-torch] 沒有 {voice!r} 的 .pt 下載點")
        print(f"[piper-torch] 下載 {voice}.pt（~204MB）…")
        _download(url, pt)          # .pt.json 已在 clone 的 models/ 裡
    print(f"[piper-torch] 完成。config 設 PIPER_ENGINE='torch' 或跑 --piper-engine torch")

    r = subprocess.run([py, "-c", "import torch; print(torch.cuda.is_available())"],
                       capture_output=True, text=True)
    if "True" not in r.stdout:
        print("[piper-torch] ⚠️ torch 看不到 CUDA GPU —— 會 fallback 到 CPU（就沒意義了）。"
              "確認有 NVIDIA 顯卡 + 驅動。")


# ------------------------------------------------------------------ piper 語音模型
def prepare_piper(voice="en_US-libritts_r-medium"):
    dst = paths.DATASETS / "_models" / "piper"
    dst.mkdir(parents=True, exist_ok=True)
    base = PIPER_VOICES.get(voice)
    if base is None:
        sys.exit(f"[piper] 不認得的語音 {voice!r}；可用：{', '.join(PIPER_VOICES)}")
    for ext in (".onnx", ".onnx.json"):
        out = dst / f"{voice}{ext}"
        if out.exists() and out.stat().st_size > 1000:
            print(f"[piper] {out.name} 已存在，跳過")
            continue
        print(f"[piper] 下載 {voice}{ext} …")
        _download(base + f"{voice}{ext}", out)
    print(f"[piper] 完成 → {dst}")


# ------------------------------------------------------------------ helpers
def _download(url, dst):
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dst, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:.0f} / {total/1e6:.0f} MB", end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser(description="準備第三方資料集 / 共用負樣本池")
    ALL = ["speech_commands", "music", "noise", "piper", "generic"]
    ap.add_argument("which", nargs="*", default=ALL,
                    choices=ALL + ["piper-torch"])
    ap.add_argument("--per-type", type=int, help="傳給 make_noise.py（每類噪音段數）")
    ap.add_argument("--count", type=int, default=10000, help="generic 通用語音負樣本數")
    args = ap.parse_args()

    noise_extra = []
    if args.per_type is not None:
        noise_extra = ["--per-type", str(args.per_type)]

    for w in args.which:
        if w == "speech_commands":
            prepare_speech_commands()
        elif w == "music":
            prepare_music()
        elif w == "noise":
            prepare_noise(noise_extra)
        elif w == "piper":
            prepare_piper()
        elif w == "piper-torch":
            prepare_piper_torch()
        elif w == "generic":
            prepare_generic(args.count)


if __name__ == "__main__":
    main()
