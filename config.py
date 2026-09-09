# -*- coding: utf-8 -*-
"""
config.py
=========
全自動化地端喚醒詞訓練 pipeline 的所有可調參數。

run_all.py 會把這些值當作 kws/train.py 的命令列參數傳進去。
只有喚醒詞（--wake-word）需要每次手動指定，其餘都吃這裡的預設。
"""

# ==========================================
# 1. 資料集組成參數
# ==========================================
MODEL_MODE               = "lite"    # 模型版本，不用調
TARGET_UNKNOWN_COUNT     = 12000     # 訓練時每個 epoch 抽多少負樣本
# 負樣本 5 個來源的比例，總和須為 1.0：
RATIO_NEAR_MISS          = 0.25      # 喚醒詞近音 / 部分詞（datasets/tts/<label>/negative/，每個喚醒詞自己生）
RATIO_GENERIC            = 0.25      # 通用英文語音（datasets/generic_speech/，全部喚醒詞共用，生一次）
RATIO_GSC                = 0.2       # Google Speech Commands（單字）
RATIO_FMA                = 0.1       # FMA 音樂
RATIO_PURE_NOISE         = 0.2       # 純環境噪音
MIX_PROB_NOISE           = 0.8       # 訓練時把環境噪音混入音訊的機率
MIX_PROB_FMA             = 0.6       # 訓練時把 FMA 音樂混入音訊的機率
RATIO_COMMAND_TO_UNKNOWN = 1         # 正樣本（喚醒詞）對負樣本的數量比例
NUM_TEST_SAMPLES         = 50        # 訓練前預先生成的測試混音樣本數量，不用調
SNR_MIN                  = 4.0       # 混音時最低信噪比（dB），越小噪音越大
SNR_MAX                  = 20.0      # 混音時最高信噪比（dB），越大音訊越乾淨

# ==========================================
# 2. 訓練超參數（對應 .tcl 第 2 段）
# ==========================================
SEED          = 42        # 隨機種子，固定可重現
LEARNING_RATE = "5e-4"    # 學習率
BATCH_SIZE    = 100       # 每批次樣本數
MAX_EPOCHS    = 100       # 最多訓練回合數（可能因早停提前結束）
PATIENCE      = 10        # 早停耐心值

# ==========================================
# 3. Robustness：音量正規化 / 頻寬 / 殘響增強（對應 .tcl 第 3 段）
# ==========================================
INPUT_NORM        = "peak"   # peak / rms / none；必須跟部署端硬體正規化 IP 一致
INPUT_NORM_TARGET = 0.9
INPUT_NORM_FLOOR  = 0.001

AUG_LOWPASS_PROB   = 0.35    # 低通增強機率（模擬電話 / 藍牙 / 爛麥克風）
AUG_LOWPASS_MIN_HZ = 3200
AUG_LOWPASS_MAX_HZ = 8000

AUG_REVERB_PROB    = 0.4     # 殘響增強機率（模擬離麥克風遠 / 房間反射）
AUG_REVERB_T60_MIN = 0.15
AUG_REVERB_T60_MAX = 0.7
AUG_REVERB_MIX_MIN = 0.2
AUG_REVERB_MIX_MAX = 0.7

# ==========================================
# 4. TTS 樣本生成（可插拔後端，見 kws/tts/）
# ==========================================
POS_COUNT        = 12000  # 客製化正樣本（喚醒詞）要生幾筆
NEAR_MISS_COUNT  = 6000   # 每個喚醒詞的近音 / 部分詞負樣本要生幾筆（datasets/tts/<label>/negative/）
GENERIC_COUNT    = 10000  # 共用的通用英文語音負樣本要生幾筆（datasets/generic_speech/，只生一次）
TEST_FRAC        = 0.1    # 切到 test split 的比例
# 註：*_COUNT 是「生成池」的大小；訓練時實際用多少由上面的 RATIO_* × TARGET_UNKNOWN_COUNT 決定，
#     只要池子夠大就好（近音 0.25 × 12000 ≈ 3000，池 6000 夠）。

TTS_BACKEND = "piper"  # piper（英文預設，ONNX 本地，904 語者）/ edge（線上，中文喚醒詞用）

# --- piper 後端 ---
PIPER_ENGINE = "onnx"     # onnx = ONNX 本地 CPU（預設，零額外設定）
                          # torch = PyTorch + NVIDIA GPU 批次生成（~30x 快 + SLERP 語者內插）；
                          #         要先跑 `python datasets/prepare.py piper-torch`（建 venv-psg + 下載 .pt）
PIPER_VOICE = "en_US-libritts_r-medium"  # onnx 用 .onnx / torch 用 .pt（同名）
PIPER_MAX_SPEAKERS = None  # None = 用模型全部 904；設數字可限制
PIPER_WORKERS = 2          # onnx 引擎的合成 thread 數。看到失敗/雜訊就設 1。
PIPER_TORCH_BATCH = 32     # torch 引擎的 GPU batch。RTX 3050 4GB OK；大 GPU 可設 64+。

# --- edge 後端 ---
VOICE_PACK = "en"          # en / en-us（英文喚醒詞若 --tts-backend edge 時用）
ZH_VOICE_PACK = "zh-tw"    # 中文喚醒詞的正樣本 voice pack：zh-tw / zh-cn / zh-hk
TTS_CONCURRENCY = 6        # 同時併發的 edge-tts 請求數

# 中文喚醒詞：正樣本 edge-tts（真中文），負樣本 piper（英文 / 拼音）。
# run_all.py --wake-word 有 CJK 字元就自動走這條，不用改任何設定。

# ==========================================
# 5. 推論 / 評估
# ==========================================
SUGGESTED_THRESHOLD = 0.5   # 寫進 metadata.json；infer.py / mic_demo.py 的預設判定門檻。
                            #   模型輸出 2 類 softmax，>0.5 = 喚醒詞分數贏過 unknown（argmax）。
                            #   用 --threshold 依需求手動調：要少漏抓用 0.5；要少誤觸發用 0.7~0.9。

# 預設關閉。要對自己的測試集算 precision/recall/FAR，用
#   python inference/infer.py --input <dir>            （檔名/子夾判斷 ground truth）
# 或 run_all.py --test-database <dir>（需自備 labels.csv，格式見 inference/README.md）。
SKIP_EXTERNAL_EVAL = True

# ==========================================
# 6. 煙霧測試（run_all.py --smoke 會套用這組覆蓋值）
# ==========================================
SMOKE_OVERRIDES = dict(
    POS_COUNT=30,
    NEAR_MISS_COUNT=30,
    GENERIC_COUNT=40,
    TARGET_UNKNOWN_COUNT=300,
    MAX_EPOCHS=2,
    NUM_TEST_SAMPLES=5,
)
