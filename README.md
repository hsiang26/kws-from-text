# KWS —— 地端關鍵詞偵測 / 自訂喚醒詞訓練

**從一句喚醒詞，在本地訓練出你自己的語音喚醒模型。** 用 TTS 合成訓練資料 → 訓練
Hello-Edge DS-CNN → 匯出 INT8 `.tflite`（~47 KB）→ 直接跑推論。

只需要輸入喚醒詞，其餘全自動（建環境、下載資料集、生樣本、訓練都由 `run_all.py` 包辦）。
**英文 / 中文喚醒詞都支援。**

---

## ▶ 線上試玩

`docs/index.html` 是一個**單檔、純前端**的 demo：用 JavaScript 跑訓練好的同一個 47 KB 模型
（權重直接內嵌），開麥克風即時偵測「hey assistant」和「小幫手」—— 不上傳、不連線、不用後端。

- **部署**：push 上 GitHub → Settings → Pages → Source 選 `Deploy from a branch`、資料夾 `/docs`
  → 幾分鐘後在 `https://<你的帳號>.github.io/<repo>/` 就有了。
- **本機先看**：`cd docs && python -m http.server 8000`，開 `http://localhost:8000`。

> 麥克風需要 **HTTPS 或 localhost**（瀏覽器的 secure-context 規定）。GitHub Pages 是 HTTPS，
> 直接雙擊 `file://` 開則拿不到麥克風。

---

## 快速開始

**需要**：Python 3.9–3.13、網路、約 10 GB 磁碟。第一次跑約 **1–2 小時**（下載 GSC 2.3 GB +
生成 ~28000 筆 TTS 樣本 + 訓練）。GPU 非必要（有 NVIDIA 卡可大幅加速，見〈GPU 加速生成〉）。

```bash
git clone <this-repo> kws && cd kws
python run_all.py --wake-word "hey assistant"
```

`run_all.py` 會依序：建 `venv/` 並裝套件 → 下載 / 生成資料集 → 用 Piper 合成正樣本 + 近音負樣本
→ 訓練 → 匯出。產出在 `models/hey_assistant_<時間戳>/`：

| 檔案 | 用途 |
|---|---|
| `model_int8.tflite` | INT8 量化模型（~47 KB） |
| `metadata.json` | 推論需要的一切：取樣率、特徵參數、量化 scale/zero-point、正類 index、建議門檻、喚醒詞 |
| `training_report.txt` | 這次 run 的設定、資料組成、訓練曲線、評估 |
| `best.keras` | 訓練中最佳 checkpoint（float） |
| `test_mix_audio/` | 訓練前預生的測試混音樣本 |

中文一樣：`python run_all.py --wake-word "小幫手"`（偵測到中文字自動切換路線）。

---

## 用訓練好的模型

repo 附了兩個**現成的範例模型**（`models/examples/hey_assistant/`、`models/examples/xiao_bang_shou/`），
不用訓練就能測：

```bash
# 麥克風即時偵測（需 pip install -r requirements-mic.txt）
python inference/mic_demo.py --model models/examples/hey_assistant

# 對 .wav 檔或整個資料夾離線評分
python inference/infer.py --model models/examples/hey_assistant --input some_clip.wav

# 對測試集算 Precision / Recall / FAR / F1 + 類別分解
python inference/infer.py --model models/examples/hey_assistant --labels-csv test_set/labels.csv
```

**部署到你自己的專案**：載入 `model_int8.tflite`，照 `metadata.json` 的參數做特徵抽取
（44100 Hz、1.0 秒、1024-pt FFT、40 mel、10 MFCC、per-utterance CMVN），量化後丟進去。
`kws/features.py` 是唯一的一份參考實作，訓練和推論都走它。

### 門檻（`--threshold`）

模型參數基本**不用調**，唯一要依需求調的是判定門檻。輸出是 2 類 softmax，`>0.5` 就是
「喚醒詞分數贏過 unknown」。分數分布是雙峰的（真喚醒詞 ≈ 0.99、其他 ≈ 0），所以：

| 需求 | 門檻 | 效果 |
|---|---|---|
| **少誤觸發**（車用、always-on） | 0.8 – 0.9 | 誤喚醒極少，但講得不標準時可能要重講 |
| **平衡**（預設） | **0.5** | argmax，哪一類分數高就算哪類 |
| **少漏抓** | 0.5 – 0.7 | 儘量不錯過，可容忍偶爾誤觸發 |

`inference/*.py` 預設讀 `metadata.json` 的 `suggested_threshold`（0.5），`--threshold 0.85` 可蓋掉。

---

## 運作原理

**5 步**：`run_all.py` → ① 建 venv / 裝套件 ② 備資料集 ③ TTS 生正樣本 + 近音負樣本
④ 建 run 目錄 ⑤ 訓練 + 匯出。

**樣本生成**（`kws/generate.py` + `kws/tts/`）——

- **英文喚醒詞** → **Piper** `en_US-libritts_r-medium`（LibriTTS-R VITS，**904 個語者**），每個樣本
  隨機抽語者 + 隨機語速 / 韻律 + 句尾標點，再 librosa 事後 time-stretch / pitch-shift。
  預設 ONNX、CPU、離線。
- **中文喚醒詞** → 正樣本用 **edge-tts**（真中文，`zh-tw` 語者）；近音負樣本用 Piper（喚醒詞的
  部分片語轉拼音，`小幫手` → `xiao` / `bang shou` …）。

訓練時 `kws/train.py` 自己會混噪音 / FMA 音樂（SNR 4–20 dB）、殘響、低通，所以這裡輸出乾淨樣本。

**負樣本在 trainer 裡有 5 個池**，比例由 `config.py` 的 `RATIO_*`（總和 1.0）決定：

| 池 | 內容 | 誰生 | 預設比例 |
|---|---|---|---|
| `near_miss` | 喚醒詞近音 / 部分詞（`say assistant` / `hey resistant` / `insistent` …） | `kws/generate.py`，每個喚醒詞自己生 | 0.25 |
| `generic` | 通用英文詞 / 短句（數字、星期、語音助理指令、日常對話） | `kws.generate --kind generic`，**生一次**共用 | 0.25 |
| `gsc` | Google Speech Commands 單字 | `datasets/prepare.py` | 0.20 |
| `fma` | FMA 音樂 | `datasets/prepare.py` | 0.10 |
| `noise` | 合成環境噪音（white/pink/brown/fan/wind） | `datasets/make_noise.py` | 0.20 |

**模型** —— Hello Edge "Small" DS-CNN + MFCC-10（Zhang et al. 2017, arXiv:1711.07128），
輸入 (49, 10, 1)，2 類。**特徵** —— 44100 Hz、1.0 秒、20 ms hop、1024-pt FFT、40 mel、
取前 10 個 MFCC、per-utterance CMVN。訓練與推論走 `kws/features.py` 同一份（含 web demo，逐位對齊）。

---

## GPU 加速生成（選配）

預設 Piper 用 ONNX 在 CPU 跑，12000 + 6000 樣本約 **30–60 分鐘**。有 **NVIDIA GPU** 的話，
可切到 PyTorch 批次生成（原 `piper-sample-generator` 那條路，還多做 SLERP 語者內插）：

```bash
python datasets/prepare.py piper-torch      # 建 venv-psg + 裝 torch(CUDA) + 下載 .pt（~3 GB，一次性）
python run_all.py --wake-word "hey assistant" --piper-engine torch
```

實測 RTX 3050 Laptop（4 GB）：**~150 筆 / 秒**，整批生成 **~2–3 分鐘**。
或在 `config.py` 設 `PIPER_ENGINE = "torch"`。

---

## 實測與分析

`hey_assistant` 範例模型對一組 225 筆測試集（25 正 / 200 負，門檻 **0.9**）：

| 分組 | 樣本數 | 正確 | 準確率 |
|---|---:|---:|---:|
| 正樣本 · 真人錄音（原始） | 5 | 2 | **40 %** |
| 正樣本 · 真人錄音（變速 / 變調 / 增益） | 7 | 2 | **29 %** |
| 正樣本 · TTS 合成 | 13 | 13 | **100 %** |
| 負樣本 · 真人近音（"hey sister" …） | 50 | 48 | **96 %** |
| 負樣本 · TTS 合成 | 50 | 50 | **100 %** |
| 負樣本 · 一般語音 | 100 | 100 | **100 %** |
| **合計** | 225 | 215 | **95.6 %** |
| — Recall（正樣本抓到率） | 25 | 17 | 68 % |
| — 1 − FAR（負樣本擋掉率） | 200 | 198 | 99 % |

**看得出來的重點**：

1. **合成 / 一般語音近乎完美**（100 %）—— 模型分辨「這是不是喚醒詞的音」沒問題。
2. **真人講的喚醒詞是弱點** —— 真人正樣本只有 ~35 % recall，其中 5 筆分數幾乎是 0，**任何門檻都救不回來**。
3. **原因**：模型 100 % 用 TTS 訓練，從沒聽過真人的音色、口氣、遠場、環境。合成語音的分布跟真人有落差。
4. **解法**：把真人錄音（哪怕幾十筆）加進 `datasets/tts/<label>/positive/` 一起訓練，真人 recall 會明顯拉高。
5. 誤觸發只有 2 筆，都是分數 0.98 的真人近音錄音（"hey sister" 類）——
   `near_miss` 負樣本已經擋掉大部分，剩這種極接近的靠調高門檻。

---

## 調參

`config.py` 有註解，**預設就能用**。真的要調，常見的是：

- `RATIO_NEAR_MISS` / `RATIO_GENERIC` / `RATIO_GSC` / `RATIO_FMA` / `RATIO_PURE_NOISE` —— 5 個負樣本池比例（總和 1.0）
- `POS_COUNT` / `NEAR_MISS_COUNT` / `GENERIC_COUNT` —— 各池生成大小
- `MAX_EPOCHS` / `LEARNING_RATE` / `BATCH_SIZE` / `PATIENCE`
- `INPUT_NORM` / `AUG_LOWPASS_*` / `AUG_REVERB_*` —— robustness 增強
- `PIPER_ENGINE`（onnx / torch）、`ZH_VOICE_PACK`（zh-tw / zh-cn / zh-hk）

`run_all.py` 旗標：`--smoke`（小樣本 2 epoch，驗證整條路徑）、`--piper-engine {onnx,torch}`、
`--tts-backend {piper,edge}`、`--pos-count` / `--near-miss-count`、
`--test-database <dir>`（訓練後自動評估）、`--skip-gen` / `--skip-train`。

---

## 專案結構

```
config.py              所有可調參數
run_all.py             一鍵：env → datasets → 生成 → 訓練

kws/                   pipeline 套件
  paths.py               所有路徑集中
  features.py            wav → MFCC-10 + INT8 量化（訓練 / 推論 / web demo 共用）
  generate.py            生成正樣本 + 近音負樣本（呼叫 kws/tts/ 後端）
  negative_phrases.py    近音 / 部分詞語料 + 通用英文語料
  train.py               Hello-Edge DS-CNN 訓練 + 匯出 tflite + metadata
  tts/
    piper.py               Piper ONNX 後端（CPU，預設）
    piper_torch.py          Piper PyTorch 後端（NVIDIA GPU 批次）
    edge.py                 edge-tts 後端（中文正樣本）

datasets/              (內容 .gitignore；prepare.py 取得)
  prepare.py             下載 GSC + FMA + Piper 模型、生成 noise / generic 池
  make_noise.py          numpy 合成環境噪音
  speech_commands/  music/  noise/  generic_speech/     負樣本池
  tts/<label>/           每個喚醒詞的 positive/ + negative/（近音）
  _models/piper/         Piper 模型

models/
  <label>_<時間戳>/      (.gitignore) 訓練產物
  examples/               (進 git) 兩個現成範例模型

inference/
  infer.py               離線評分器（.wav / 資料夾 / labels.csv）
  mic_demo.py            麥克風即時偵測

docs/index.html         純瀏覽器 web demo（GitHub Pages）
```

---

## 目前限制

- **只用 TTS 訓練 → 真人辨識率有落差**（見〈實測與分析〉）。要拉高就補真人錄音進訓練資料。
- **中文負樣本還很陽春**：只有「喚醒詞部分片語的拼音」+ 通用英文句，沒有聲母 / 韻母 / 聲調近音
  （`小幫手` vs `小胖手` / `小幫忙`）。`kws/negative_phrases._build_cjk` 的 TODO。
- **中文正負樣本 TTS 引擎不同**（正 = edge-tts 真中文、負 = Piper 英文拼音），模型有機會靠音色而非
  語音內容分類，實際效果要看回測。
- **中文正樣本音色少**：edge-tts 中文只有 ~11 個語者。
- **外部下載會失效**：GSC / FMA / Piper 模型都是從第三方抓，來源掛掉 repo 就半殘。

---

## 致謝與授權

### 授權

本專案以 **GPL-3.0-or-later** 釋出（`LICENSE` 為全文）。之所以是 GPL：預設的 TTS 引擎
`piper-tts`（及可選的 `edge-tts`、Piper 內含的 espeak-ng）都是 GPL-3.0。

```
KWS — 地端關鍵詞偵測 / 自訂喚醒詞訓練
Copyright (C) 2026 Jerome Hsiang

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. This program is distributed WITHOUT ANY WARRANTY. See the GNU General
Public License for more details <https://www.gnu.org/licenses/>.
```

### TTS 引擎

| | 授權 | 說明 |
|---|---|---|
| [Piper / piper-tts](https://github.com/OHF-Voice/piper1-gpl) | GPL-3.0-or-later | 預設樣本生成引擎（含 espeak-ng 音素化） |
| `en_US-libritts_r-medium` 語音模型 | — | Piper 官方發布，904 語者；訓練資料 [LibriTTS-R](https://www.openslr.org/141/)（Koizumi et al. 2023），衍生自 LibriTTS（Zen et al. 2019），**CC BY 4.0** |
| [edge-tts](https://github.com/rany2/edge-tts) | GPL-3.0 | 可選後端，中文喚醒詞正樣本用 |
| [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) | MIT | GPU 批次生成路線（`--piper-engine torch`） |

### 資料集

| | 授權 | 引用 / 來源 |
|---|---|---|
| Google Speech Commands v0.02 | **CC BY 4.0** | Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209 · `download.tensorflow.org/data/speech_commands_v0.02.tar.gz` |
| FMA: Free Music Archive（FMA-small） | 逐曲 Creative Commons（部分為 CC BY-NC） | Defferrard et al. (2017). *FMA: A Dataset for Music Analysis.* ISMIR. arXiv:1612.01840 · Hugging Face `rudraml/fma` |

repo 本身**不含**這些資料集的音檔；`datasets/prepare.py` 才會從各自的官方來源取得。

### 方法 / 架構

- **openWakeWord**（Apache-2.0）—— 「合成 TTS 正樣本 + 對抗性近音負樣本」的整體思路參考自此；
  `kws/negative_phrases.py` 的近音負樣本是它 `generate_adversarial_texts` 的簡化版
  （改用 `pronouncing` + CMUdict，另加一份通用英文語料）。
- **Hello Edge: Keyword Spotting on Microcontrollers**（Zhang, Suda, Lai, Chandra, 2017, arXiv:1711.07128）
  + ARM [ML-KWS-for-MCU](https://github.com/ARM-software/ML-KWS-for-MCU)（Apache-2.0）—— DS-CNN 架構。
- `pronouncing`（BSD-2）+ CMU Pronouncing Dictionary；`pypinyin`（MIT）。

其餘 Python 相依（TensorFlow、librosa …）各自的授權見 `requirements*.txt`。
