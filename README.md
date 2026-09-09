<a id="chinese"></a>

# kws-from-text —— 自訂中英文關鍵詞偵測（keyword spotting）模型

**打一句關鍵詞，就能訓練出你專屬的語音喚醒模型。不用錄音、不用標註，中英文都支援。**

- **只給一句話**：輸入「小幫手」或「hey assistant」，不用自己錄音、不用找人標註
- **中英文都行**：中文用真人級 Mandarin 語音、英文用 900+ 語者的語音自動合成訓練資料
- **一個指令到底**：建環境、抓資料、生樣本、訓練、匯出，`run_all.py` 全包
- **極度輕量**：~2.2 萬參數、~47&nbsp;KB 的全 INT8 `.tflite`，可直接上微控制器 / NPU，完全離線常駐
  （詳細規格見〈模型規格〉）

*(English version below — [jump to English](#english))*

---

## 線上試玩

`docs/index.html` 是一個**單檔、純前端**的 demo：用 JavaScript 跑訓練好的同一個 47&nbsp;KB 模型
（權重直接內嵌），開麥克風即時偵測「hey assistant」和「小幫手」—— 不上傳、不連線、不用後端。

<!-- DEMO 影片：把螢幕錄影拖進任一個 GitHub issue／PR 留言框，GitHub 會生成一個
     https://github.com/user-attachments/assets/... 連結，把下面這行換成該連結即可（GitHub 會自動內嵌播放器）。 -->
> **DEMO 影片**：_（待補）_

- **部署**：push 上 GitHub → Settings → Pages → Source 選 `Deploy from a branch`、資料夾 `/docs`
  → 幾分鐘後在 `https://<你的帳號>.github.io/<repo>/` 就有了。
- **本機先看**：`cd docs && python -m http.server 8000`，開 `http://localhost:8000`。

> 麥克風需要 **HTTPS 或 localhost**（瀏覽器的 secure-context 規定）。GitHub Pages 是 HTTPS，
> 直接雙擊 `file://` 開則拿不到麥克風。建議用 Chrome / Edge。按「開始聆聽」後若沒反應，
> 頁面會在 2 秒後顯示診斷（收到幾個音框、音量峰值、選到哪個輸入裝置）—— 峰值是 0 通常是
> Windows 隱私權設定擋住桌面應用程式、或麥克風被其他程式（Discord 等）占用。

---

## 快速開始

**需要**：Python 3.9–3.13、網路、約 10 GB 磁碟。第一次跑約 **1–2 小時**（下載 GSC 2.3 GB +
生成 ~28000 筆 TTS 樣本 + 訓練）。GPU 非必要（有 NVIDIA 卡可大幅加速，見〈GPU 加速生成〉）。

```bash
git clone https://github.com/hsiang26/kws-from-text.git && cd kws-from-text
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

repo 附了兩個**現成的範例模型**（`models/examples/hey_assistant/`、`models/examples/小幫手/`），
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

模型的參數（特徵、網路、負樣本比例、增強）**我們都已經調好了，而且有實測佐證**（見〈實測與分析〉），
可以直接用。**唯一需要你依場景調的是判定門檻 `--threshold`，預設 `0.5`。**

輸出是 2 類 softmax，門檻就是「喚醒詞分數要多高才算數」。分數分布是雙峰的（真喚醒詞 ≈ 0.99、
其他 ≈ 0），大部分情況預設 0.5 就夠：

| 門檻 | 適合場景 | 取捨 |
|---|---|---|
| **0.5**（預設） | 一般消費級應用：智慧家電、桌面 / App 助理、互動裝置。使用者用自然、隨口的語氣講就會醒，體驗最順 | 極少數情況會被發音非常接近的詞觸發 |
| **0.9** | 誤觸發代價高的場景：車載 / 免持工業設備、always-on 電池裝置，或喚醒詞跟日常用語很像、需要硬性區隔時 | 使用者得清楚、完整地把整個喚醒詞唸出來才會醒 |

`inference/*.py` 預設讀 `metadata.json` 的 `suggested_threshold`（0.5），`--threshold 0.9` 可蓋掉。

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

**特徵與模型** —— 44100 Hz、1.0 秒音訊 → MFCC-10（20 ms hop、1024-pt FFT、40 mel、
per-utterance CMVN）→ tensor (49, 10, 1) → Hello Edge DS-CNN "Small" → 2 類。訓練與推論走
`kws/features.py` 同一份（含 web demo，逐位對齊）。完整規格見〈模型規格〉。

---

## 模型規格

### 架構 —— Hello Edge DS-CNN "Small"

深度可分離卷積網路（depthwise-separable CNN），出自 *Hello Edge: Keyword Spotting on
Microcontrollers*（Zhang et al. 2017, arXiv:1711.07128）：

```
輸入 MFCC (49, 10, 1)
  Conv2D  64 @ (10x4), stride 2, SAME    + BN + ReLU        -> (25, 5, 64)
  4x |  DepthwiseConv2D (3x3)            + BN + ReLU
     |  Conv2D  64 @ (1x1)               + BN + ReLU         -> (25, 5, 64)
  GlobalAveragePooling                                       -> (64,)
  Dense 2 + softmax                                          -> [非喚醒詞, 喚醒詞]
```

深度可分離卷積把每個區塊的運算量壓到標準 3×3 卷積的約 1/8。整體每次推論約 **2.7M 次乘加（MAC）**
—— 這是網路結構決定的固定值，不隨處理器 / 加速器變動（實際執行時間才依硬體而異）。

### 規格表

| 項目 | 值 |
|---|---|
| 可訓練參數 | **22,530**（約 2.2 萬） |
| 模型檔 | 全 INT8 量化 `.tflite`，**~47 KB**（BN 折進卷積後，權重本體約 21 KB） |
| 執行期記憶體 | 常駐權重 ~21 KB ＋ 一塊 (25×5×64) INT8 activation buffer（~8 KB）；無動態配置 |
| 輸入 | 單聲道 PCM，**44100 Hz**，**固定 1.0 秒**（44100 sample，不足補零、超過截斷） |
| 特徵 | MFCC-10 → tensor **(49, 10, 1)**（20 ms hop、1024-pt FFT、40 mel、per-utterance CMVN） |
| 輸出 | 2 類 softmax `[非喚醒詞, 喚醒詞]` |
| 判定 | `喚醒詞分數 ≥ threshold`（預設 0.5，見〈門檻〉） |
| 串流用法 | 1 秒滑動視窗，每 ~100–120 ms 推論一次（見 `inference/mic_demo.py`） |
| 量化目標 | 權重 + activation 全 INT8，可直接上 [TFLite Micro](https://github.com/tensorflow/tflite-micro) / CMSIS-NN / NPU，不需 FPU |

### 使用限制與特性

- **一個模型只認一個喚醒詞**。要偵測多個詞就訓練多個模型並行跑。
- **喚醒詞長度：整句要能在約 1 秒內自然講完**。模型只吃 1.0 秒的音訊視窗，講太長尾巴會被截掉。
  最穩的是 **2–4 個中文字 / 1–3 個英文單字**；只有單一個音節則太短、線索不足、容易誤觸發。
- **非語者相關**：不做聲紋 / 語者註冊，任何人講都會觸發（喚醒詞本來就該這樣，但也擋不掉刻意模仿）。
- **不是語音辨識**：只回答「這 1 秒內有沒有出現這個詞」，不轉寫、不定位詞出現的時間點。
- 輸入必須是 44100 Hz；收音裝置若是其他取樣率要先 resample（`inference/` 的工具會自動處理）。
- 語言：`--wake-word` 含中日韓字元 → 中文路線，否則英文路線；其他語言未測試。

### 實證

`docs/index.html` 用**手寫的 JavaScript**（自寫 radix-2 FFT + MFCC + DS-CNN 前向）在瀏覽器分頁裡
即時跑同一個模型，每 ~120 ms 推論一次 —— 純前端、無 WebAssembly、無 GPU。作為對照，同任務常見的
CNN / RNN 喚醒詞模型多在數百 KB ~ 數 MB、參數十萬起跳。

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

`hey_assistant` 範例模型，對一組 **225 筆**測試集（25 正 / 200 負）在兩個門檻下的表現。
正樣本分「真人錄音」與「TTS 合成」；負樣本分「真人念的近音詞」「TTS 合成的近音詞」
「一般語音（Google Speech Commands）」。

**門檻 0.5（預設）**

| 分組 | 樣本數 | 正確 | 準確率 |
|---|---:|---:|---:|
| 正樣本 · 真人錄音（原始） | 5 | 3 | **60 %** |
| 正樣本 · 真人錄音（變速 / 變調 / 增益） | 7 | 4 | **57 %** |
| 正樣本 · TTS 合成 | 13 | 13 | **100 %** |
| 負樣本 · 真人近音（"hey sister" …） | 50 | 44 | **88 %** |
| 負樣本 · TTS 合成近音 | 50 | 49 | **98 %** |
| 負樣本 · 一般語音 | 100 | 100 | **100 %** |
| **合計** | 225 | 213 | **94.7 %** |
| — Recall（正樣本抓到率） | 25 | 20 | 80 % |
| — 1 − FAR（負樣本擋掉率） | 200 | 193 | 96.5 % |

**門檻 0.9**

| 分組 | 樣本數 | 正確 | 準確率 |
|---|---:|---:|---:|
| 正樣本 · 真人錄音（原始） | 5 | 2 | **40 %** |
| 正樣本 · 真人錄音（變速 / 變調 / 增益） | 7 | 2 | **29 %** |
| 正樣本 · TTS 合成 | 13 | 13 | **100 %** |
| 負樣本 · 真人近音（"hey sister" …） | 50 | 48 | **96 %** |
| 負樣本 · TTS 合成近音 | 50 | 50 | **100 %** |
| 負樣本 · 一般語音 | 100 | 100 | **100 %** |
| **合計** | 225 | 215 | **95.6 %** |
| — Recall（正樣本抓到率） | 25 | 17 | 68 % |
| — 1 − FAR（負樣本擋掉率） | 200 | 198 | 99 % |

**怎麼讀這張表**

1. **TTS 合成樣本、一般語音：兩個門檻都近乎滿分。** 模型判斷「這段音像不像喚醒詞」本身沒問題。
2. **門檻 0.5（預設）→ 體驗最順。** 真人講喚醒詞的 recall 從 0.9 的 33 % 拉到 ~58 %，使用者不用
   刻意字正腔圓、用自然隨口的語氣就會醒 —— 這對一般消費級應用（智慧家電、桌面 / App 助理）
   才是對的取捨。代價很小：200 筆負樣本裡 7 筆誤觸發，其中 6 筆還是真人念的極接近近音詞
   （"hey sister" 這類），一般日常語音（100 筆）完全沒被觸發。
3. **門檻 0.9 → 嚴格、幾乎不誤觸發。** 200 筆負樣本只漏 2 筆，但使用者得字正腔圓地把整個
   喚醒詞唸出來，隨口帶過就喚不醒。適合：
   - 車載 / 免持操作的工業設備（誤觸發＝誤動作）
   - always-on 電池裝置（每次誤喚醒都在耗電）
   - 喚醒詞跟其他日常說法很接近、需要硬性區隔（例如要讓 "hey assistant" 不被 "hey sister"、
     "hey listen" 之類帶過的話觸發）
4. **真人錄音的喚醒詞在兩個門檻下都是最弱的一組**（0.5 約 58 %、0.9 約 33 %），TTS 合成的則 100 %。
   原因單純：範例模型 100 % 用合成語音訓練，沒聽過真人的音色、口氣、遠場與環境，合成語音的分布
   跟真人有落差。這是「零錄音」訓練換來的取捨。

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
kws-from-text — 自訂中英文關鍵詞偵測模型
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

<br>

---
---

<br>

<a id="english"></a>

# kws-from-text — Custom Chinese / English Keyword-Spotting Models

**Type one keyword and train your own voice keyword-spotting model. No recording, no labelling —
Chinese and English both supported.**

- **Just one phrase:** type "小幫手" or "hey assistant" — record nothing, label nothing
- **Chinese and English:** Mandarin from a natural-sounding TTS, English from 900+ TTS speakers, all synthesised for you
- **One command:** build the env, fetch data, generate samples, train, export — `run_all.py` does the lot
- **Very small:** ~22k parameters, a ~47&nbsp;KB fully-INT8 `.tflite` — drops onto a
  microcontroller / NPU, runs always-on and fully offline (full spec under *Model spec*)

*(中文版在上方 — [回到中文](#chinese))*

---

## Live demo

`docs/index.html` is a **single-file, front-end-only** demo: JavaScript runs the exact same
trained 47&nbsp;KB model (weights embedded inline) and detects "hey assistant" and "小幫手"
from the mic in real time — nothing uploaded, no network, no backend.

<!-- DEMO video: drag a screen recording into any GitHub issue/PR comment box; GitHub returns a
     https://github.com/user-attachments/assets/... link. Replace the line below with that link
     (GitHub auto-embeds a player). -->
> **Demo video:** _(coming soon)_

- **Deploy:** push to GitHub → Settings → Pages → Source `Deploy from a branch`, folder `/docs`
  → live at `https://<your-account>.github.io/<repo>/` in a few minutes.
- **Preview locally:** `cd docs && python -m http.server 8000`, open `http://localhost:8000`.

> The mic needs **HTTPS or localhost** (browser secure-context rule). GitHub Pages is HTTPS;
> opening the file directly via `file://` won't get mic access. Chrome / Edge recommended. If
> nothing happens after you press "start listening", the page shows a diagnostic after 2&nbsp;s
> (blocks received, peak level, which input device was picked) — a peak of 0 is usually the
> Windows privacy setting blocking desktop apps, or another app (Discord, etc.) holding the mic.

---

## Quick start

**Needs:** Python 3.9–3.13, internet, ~10 GB disk. First run is **1–2 hours** (downloads GSC
2.3 GB + generates ~28,000 TTS samples + trains). No GPU required (an NVIDIA card speeds it up
a lot — see *GPU-accelerated generation*).

```bash
git clone https://github.com/hsiang26/kws-from-text.git && cd kws-from-text
python run_all.py --wake-word "hey assistant"
```

`run_all.py` runs in order: build `venv/` and install packages → download / generate datasets
→ synthesise positives + near-miss negatives with Piper → train → export. Output lands in
`models/hey_assistant_<timestamp>/`:

| File | Purpose |
|---|---|
| `model_int8.tflite` | INT8-quantised model (~47 KB) |
| `metadata.json` | everything inference needs: sample rate, feature params, quant scale/zero-point, positive-class index, suggested threshold, keyword |
| `training_report.txt` | this run's config, data mix, training curves, evaluation |
| `best.keras` | best checkpoint during training (float) |
| `test_mix_audio/` | test mixdown samples pre-generated before training |

Chinese is identical: `python run_all.py --wake-word "小幫手"` (CJK characters auto-switch the pipeline).

---

## Using a trained model

The repo ships two **ready-made example models** (`models/examples/hey_assistant/`,
`models/examples/小幫手/`) you can test without training:

```bash
# real-time mic detection (needs pip install -r requirements-mic.txt)
python inference/mic_demo.py --model models/examples/hey_assistant

# offline scoring of a .wav file or a whole folder
python inference/infer.py --model models/examples/hey_assistant --input some_clip.wav

# Precision / Recall / FAR / F1 + per-category breakdown against a test set
python inference/infer.py --model models/examples/hey_assistant --labels-csv test_set/labels.csv
```

**Deploying into your own project:** load `model_int8.tflite`, do feature extraction per the
`metadata.json` params (44100 Hz, 1.0 s, 1024-pt FFT, 40 mel, 10 MFCC, per-utterance CMVN),
quantise, feed it in. `kws/features.py` is the single reference implementation — training and
inference both use it.

### Threshold (`--threshold`)

The model parameters (features, network, negative-pool ratios, augmentation) **are already
tuned, with measurements to back it** (see *Measurements & analysis*) — use them as-is. **The
only thing you adjust per scenario is the decision threshold `--threshold`, default `0.5`.**

Output is a 2-class softmax; the threshold is "how high must the keyword score be to count".
The score distribution is bimodal (true keyword ≈ 0.99, everything else ≈ 0), so the default
0.5 is enough for most cases:

| Threshold | Good for | Trade-off |
|---|---|---|
| **0.5** (default) | consumer applications: smart appliances, desktop / app assistants, interactive devices. The user speaks naturally and casually and it wakes — the smoothest experience | very occasionally triggered by a phrase that sounds almost identical |
| **0.9** | scenarios where a false trigger is costly: in-car / hands-free industrial gear, always-on battery devices, or when the keyword is close to everyday phrasing and needs hard separation | the user has to say the whole keyword clearly and completely for it to wake |

`inference/*.py` defaults to `suggested_threshold` (0.5) from `metadata.json`; `--threshold 0.9` overrides.

---

## How it works

**5 steps:** `run_all.py` → ① build venv / install ② prepare datasets ③ TTS positives +
near-miss negatives ④ create run dir ⑤ train + export.

**Sample generation** (`kws/generate.py` + `kws/tts/`) —

- **English keyword** → **Piper** `en_US-libritts_r-medium` (LibriTTS-R VITS, **904
  speakers**); each sample draws a random speaker + random rate / prosody + sentence-final
  punctuation, then librosa time-stretch / pitch-shift afterwards. ONNX, CPU, offline by default.
- **Chinese keyword** → positives use **edge-tts** (real Mandarin, `zh-tw` voices);
  near-miss negatives use Piper (partial phrases of the keyword transliterated to pinyin,
  `小幫手` → `xiao` / `bang shou` …).

At training time `kws/train.py` mixes in noise / FMA music (SNR 4–20 dB), reverb and low-pass
itself, so generation here emits clean samples.

**Negatives are 5 pools in the trainer**, ratios set by `config.py`'s `RATIO_*` (sum to 1.0):

| Pool | Content | Generated by | Default ratio |
|---|---|---|---|
| `near_miss` | keyword near-homophones / partial words (`say assistant` / `hey resistant` / `insistent` …) | `kws/generate.py`, per keyword | 0.25 |
| `generic` | generic English words / short phrases (numbers, weekdays, assistant commands, everyday talk) | `kws.generate --kind generic`, **generated once**, shared | 0.25 |
| `gsc` | Google Speech Commands single words | `datasets/prepare.py` | 0.20 |
| `fma` | FMA music | `datasets/prepare.py` | 0.10 |
| `noise` | synthetic ambient noise (white/pink/brown/fan/wind) | `datasets/make_noise.py` | 0.20 |

**Features & model** — 44100 Hz, 1.0 s of audio → MFCC-10 (20 ms hop, 1024-pt FFT, 40 mel,
per-utterance CMVN) → tensor (49, 10, 1) → Hello Edge DS-CNN "Small" → 2 classes. Training,
inference and the web demo all run the same `kws/features.py` (bit-aligned). Full spec under
*Model spec*.

---

## Model spec

### Architecture — Hello Edge DS-CNN "Small"

A depthwise-separable CNN, from *Hello Edge: Keyword Spotting on Microcontrollers*
(Zhang et al. 2017, arXiv:1711.07128):

```
input MFCC (49, 10, 1)
  Conv2D  64 @ (10x4), stride 2, SAME    + BN + ReLU        -> (25, 5, 64)
  4x |  DepthwiseConv2D (3x3)            + BN + ReLU
     |  Conv2D  64 @ (1x1)               + BN + ReLU         -> (25, 5, 64)
  GlobalAveragePooling                                       -> (64,)
  Dense 2 + softmax                                          -> [not-keyword, keyword]
```

Depthwise-separable convolution cuts each block's compute to about 1/8 of a standard 3×3 conv.
The whole network is **~2.7M multiply-accumulates (MAC) per inference** — a fixed property of
the topology, independent of the processor / accelerator (only wall-clock time varies by hardware).

### Spec sheet

| Item | Value |
|---|---|
| Trainable parameters | **22,530** (~22k) |
| Model file | fully INT8-quantised `.tflite`, **~47 KB** (with BN folded into the convs, weights ~21 KB) |
| Runtime memory | ~21 KB resident weights + one (25×5×64) INT8 activation buffer (~8 KB); no dynamic allocation |
| Input | mono PCM, **44100 Hz**, **exactly 1.0 s** (44100 samples; zero-padded if short, truncated if long) |
| Feature | MFCC-10 → tensor **(49, 10, 1)** (20 ms hop, 1024-pt FFT, 40 mel, per-utterance CMVN) |
| Output | 2-class softmax `[not-keyword, keyword]` |
| Decision | `keyword score ≥ threshold` (default 0.5, see *Threshold*) |
| Streaming use | 1-second sliding window, one inference every ~100–120 ms (see `inference/mic_demo.py`) |
| Quantisation target | weights + activations all INT8 — drops onto [TFLite Micro](https://github.com/tensorflow/tflite-micro) / CMSIS-NN / NPUs, no FPU needed |

### Constraints & characteristics

- **One model detects one keyword.** For several keywords, train several models and run them in parallel.
- **Keyword length: the whole phrase must fit comfortably in ~1 second.** The model only sees a
  1.0 s audio window; a longer phrase gets its tail cut off. Most reliable is **2–4 Chinese
  characters / 1–3 English words**; a single syllable is too short — too few cues, easy to false-trigger.
- **Not speaker-specific:** no voiceprint / speaker enrolment, anyone's voice triggers it (that
  is what a keyword trigger should do, but it also can't reject a deliberate impersonation).
- **Not speech recognition:** it only answers "did this 1 second contain the phrase" — no
  transcription, no timing of where the word occurred.
- Input must be 44100 Hz; resample first if your capture device runs at another rate (the
  `inference/` tools do this automatically).
- Language: `--wake-word` containing CJK characters → Chinese path, otherwise English path;
  other languages are untested.

### Proof

`docs/index.html` runs the same model live in a browser tab with **hand-written JavaScript**
(own radix-2 FFT + MFCC + DS-CNN forward), one inference every ~120 ms — pure front-end, no
WebAssembly, no GPU. For contrast, typical CNN / RNN keyword-spotting models for the same task
are hundreds of KB to a few MB, with parameter counts in the hundreds of thousands.

---

## GPU-accelerated generation (optional)

Piper defaults to ONNX on CPU: 12000 + 6000 samples ≈ **30–60 min**. With an **NVIDIA GPU** you
can switch to PyTorch batch generation (the original `piper-sample-generator` route, which also
does SLERP speaker interpolation):

```bash
python datasets/prepare.py piper-torch      # build venv-psg + install torch(CUDA) + download .pt (~3 GB, one-time)
python run_all.py --wake-word "hey assistant" --piper-engine torch
```

Measured on RTX 3050 Laptop (4 GB): **~150 samples/sec**, whole batch in **~2–3 min**.
Or set `PIPER_ENGINE = "torch"` in `config.py`.

---

## Measurements & analysis

The `hey_assistant` example model on a **225-sample** test set (25 positive / 200 negative) at
two thresholds. Positives split into "real human recording" and "TTS synthesis"; negatives into
"human-spoken near-miss", "TTS near-miss" and "generic speech (Google Speech Commands)".

**Threshold 0.5 (default)**

| Group | n | Correct | Accuracy |
|---|---:|---:|---:|
| Positive · real recording (original) | 5 | 3 | **60 %** |
| Positive · real recording (speed / pitch / gain) | 7 | 4 | **57 %** |
| Positive · TTS synthesis | 13 | 13 | **100 %** |
| Negative · human near-miss ("hey sister" …) | 50 | 44 | **88 %** |
| Negative · TTS near-miss | 50 | 49 | **98 %** |
| Negative · generic speech | 100 | 100 | **100 %** |
| **Total** | 225 | 213 | **94.7 %** |
| — Recall (positive catch rate) | 25 | 20 | 80 % |
| — 1 − FAR (negative reject rate) | 200 | 193 | 96.5 % |

**Threshold 0.9**

| Group | n | Correct | Accuracy |
|---|---:|---:|---:|
| Positive · real recording (original) | 5 | 2 | **40 %** |
| Positive · real recording (speed / pitch / gain) | 7 | 2 | **29 %** |
| Positive · TTS synthesis | 13 | 13 | **100 %** |
| Negative · human near-miss ("hey sister" …) | 50 | 48 | **96 %** |
| Negative · TTS near-miss | 50 | 50 | **100 %** |
| Negative · generic speech | 100 | 100 | **100 %** |
| **Total** | 225 | 215 | **95.6 %** |
| — Recall (positive catch rate) | 25 | 17 | 68 % |
| — 1 − FAR (negative reject rate) | 200 | 198 | 99 % |

**How to read this**

1. **TTS samples and generic speech: near-perfect at both thresholds.** The model has no trouble
   judging "does this sound like the keyword".
2. **Threshold 0.5 (default) → smoothest experience.** Recall on human-spoken keywords rises
   from 33 % (at 0.9) to ~58 %; the user doesn't have to enunciate carefully — speaking
   naturally and casually wakes it, which is the right trade-off for consumer applications
   (smart appliances, desktop / app assistants). The cost is small: 7 false triggers out of 200
   negatives, 6 of them human-spoken very-close near-misses ("hey sister" and similar), with
   zero false triggers on the 100 everyday-speech clips.
3. **Threshold 0.9 → strict, almost no false triggers.** Only 2 of 200 negatives slip through,
   but the user must enunciate the whole keyword clearly — a mumbled pass won't wake it. Good for:
   - in-car / hands-free industrial gear (a false trigger means a wrong action)
   - always-on battery devices (every false wake drains power)
   - a keyword that is close to everyday phrasing and needs hard separation (e.g. keeping
     "hey assistant" from firing on a throwaway "hey sister" / "hey listen")
4. **Human-recorded keywords are the weakest group at both thresholds** (~58 % at 0.5, ~33 %
   at 0.9), while TTS ones are 100 %. The reason is simple: this example model was trained 100 %
   on synthetic speech and has never heard a real human's timbre, delivery, far-field or room —
   synthetic speech is distributed differently from real speech. That is the trade-off of
   zero-recording training.

---

## Tuning

`config.py` is commented and **works out of the box**. If you must tune, the common knobs are:

- `RATIO_NEAR_MISS` / `RATIO_GENERIC` / `RATIO_GSC` / `RATIO_FMA` / `RATIO_PURE_NOISE` — the 5 negative-pool ratios (sum 1.0)
- `POS_COUNT` / `NEAR_MISS_COUNT` / `GENERIC_COUNT` — per-pool generation size
- `MAX_EPOCHS` / `LEARNING_RATE` / `BATCH_SIZE` / `PATIENCE`
- `INPUT_NORM` / `AUG_LOWPASS_*` / `AUG_REVERB_*` — robustness augmentation
- `PIPER_ENGINE` (onnx / torch), `ZH_VOICE_PACK` (zh-tw / zh-cn / zh-hk)

`run_all.py` flags: `--smoke` (tiny sample, 2 epochs, validates the whole path),
`--piper-engine {onnx,torch}`, `--tts-backend {piper,edge}`, `--pos-count` / `--near-miss-count`,
`--test-database <dir>` (auto-evaluate after training), `--skip-gen` / `--skip-train`.

---

## Project layout

```
config.py              all tunable parameters
run_all.py             one command: env → datasets → generate → train

kws/                   pipeline package
  paths.py               all paths in one place
  features.py            wav → MFCC-10 + INT8 quant (shared by training / inference / web demo)
  generate.py            generate positives + near-miss negatives (calls kws/tts/ backends)
  negative_phrases.py    near-miss / partial-word corpus + generic English corpus
  train.py               Hello-Edge DS-CNN training + export tflite + metadata
  tts/
    piper.py               Piper ONNX backend (CPU, default)
    piper_torch.py          Piper PyTorch backend (NVIDIA GPU batch)
    edge.py                 edge-tts backend (Chinese positives)

datasets/              (contents .gitignored; fetched by prepare.py)
  prepare.py             download GSC + FMA + Piper model, generate noise / generic pools
  make_noise.py          numpy-synthesised ambient noise
  speech_commands/  music/  noise/  generic_speech/     negative pools
  tts/<label>/           per-keyword positive/ + negative/ (near-miss)
  _models/piper/         Piper model

models/
  <label>_<timestamp>/  (.gitignored) training output
  examples/               (in git) two ready-made example models

inference/
  infer.py               offline scorer (.wav / folder / labels.csv)
  mic_demo.py            real-time mic detection

docs/index.html         pure-browser web demo (GitHub Pages)
```

---

## Current limitations

- **Chinese positives and negatives use different TTS engines** (positive = edge-tts real
  Mandarin, negative = Piper English pinyin), so the model might classify on timbre rather than
  speech content — real-world effect depends on backtesting.
- **Few Chinese positive voices:** edge-tts Mandarin has only ~11 speakers.
- **External downloads can break:** GSC / FMA / Piper models are all pulled from third parties;
  if a source goes down the repo is half-crippled.

---

## Acknowledgements & license

### License

Released under **GPL-3.0-or-later** (`LICENSE` has the full text). Why GPL: the default TTS
engine `piper-tts` (and the optional `edge-tts`, and espeak-ng bundled with Piper) are all GPL-3.0.

```
kws-from-text — Custom Chinese / English Keyword-Spotting Models
Copyright (C) 2026 Jerome Hsiang

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. This program is distributed WITHOUT ANY WARRANTY. See the GNU General
Public License for more details <https://www.gnu.org/licenses/>.
```

### TTS engines

| | License | Note |
|---|---|---|
| [Piper / piper-tts](https://github.com/OHF-Voice/piper1-gpl) | GPL-3.0-or-later | default sample-generation engine (incl. espeak-ng phonemisation) |
| `en_US-libritts_r-medium` voice model | — | official Piper release, 904 speakers; training data [LibriTTS-R](https://www.openslr.org/141/) (Koizumi et al. 2023), derived from LibriTTS (Zen et al. 2019), **CC BY 4.0** |
| [edge-tts](https://github.com/rany2/edge-tts) | GPL-3.0 | optional backend, used for Chinese keyword positives |
| [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) | MIT | GPU batch-generation route (`--piper-engine torch`) |

### Datasets

| | License | Citation / source |
|---|---|---|
| Google Speech Commands v0.02 | **CC BY 4.0** | Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209 · `download.tensorflow.org/data/speech_commands_v0.02.tar.gz` |
| FMA: Free Music Archive (FMA-small) | per-track Creative Commons (some CC BY-NC) | Defferrard et al. (2017). *FMA: A Dataset for Music Analysis.* ISMIR. arXiv:1612.01840 · Hugging Face `rudraml/fma` |

The repo itself **does not contain** any of these dataset audio files; `datasets/prepare.py`
fetches them from their respective official sources.

### Method / architecture

- **openWakeWord** (Apache-2.0) — the overall "synthetic TTS positives + adversarial near-miss
  negatives" idea comes from here; `kws/negative_phrases.py`'s near-miss negatives are a
  simplified version of its `generate_adversarial_texts` (using `pronouncing` + CMUdict, plus a
  generic English corpus).
- **Hello Edge: Keyword Spotting on Microcontrollers** (Zhang, Suda, Lai, Chandra, 2017,
  arXiv:1711.07128) + ARM [ML-KWS-for-MCU](https://github.com/ARM-software/ML-KWS-for-MCU)
  (Apache-2.0) — the DS-CNN architecture.
- `pronouncing` (BSD-2) + CMU Pronouncing Dictionary; `pypinyin` (MIT).

Other Python dependencies (TensorFlow, librosa …) are under their own licenses — see `requirements*.txt`.
