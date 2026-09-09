# datasets/

pipeline 需要的音訊資料。**除了本說明與腳本，這個資料夾的內容都不進 git**
（體積太大、且多為第三方資產）。第一次使用先跑：

```bash
python datasets/prepare.py
```

| 子資料夾 | 內容 | 來源 | 授權 | 取得方式 |
|---|---|---|---|---|
| `speech_commands/` | 一般英文單字語音（~64k 檔，30+ 個詞資料夾），@44.1kHz | Google Speech Commands v0.02 | CC BY 4.0 | `prepare.py speech_commands` 下載官方 tar 並重採樣 |
| `music/` | 音樂片段 ~120 段，@44.1kHz | FMA-small（Free Music Archive） | 逐曲 CC 授權 | `prepare.py music`，從 Hugging Face `rudraml/fma` 串流 |
| `noise/` | 合成環境噪音（white/pink/brown/fan/wind，各 1000 段），@44.1kHz | 本 repo `make_noise.py` 用 numpy 生成 | 本 repo 授權 | `prepare.py noise` |
| `_models/piper/` | Piper 語音模型 `en_US-libritts_r-medium`（.onnx + .onnx.json，~79MB） | Hugging Face `rhasspy/piper-voices` | GPL-3.0（見 README.md 的「致謝與授權」） | `prepare.py piper` |
| `generic_speech/` | 通用英文語音負樣本（piper 合成 ~10000 段），**全部喚醒詞共用，只生一次** | piper（本 repo 內建詞表） | 產物 | `prepare.py generic` 或 `run_all.py` 自動生 |
| `tts/<label>/` | 每個喚醒詞的正樣本 + 近音負樣本 | Piper（預設）或 edge-tts | 產物 | `run_all.py` 自動產生 |

TTS 產物固定規格：**44100 Hz、mono、PCM_16**。訓練 (`kws/train.py`) 會在訓練時把
`noise/` 和 `music/` 以 SNR 4–20 dB 隨機混進正負樣本。

## 各自準備

```bash
python datasets/prepare.py speech_commands          # 只抓 GSC（~2.3GB 下載）
python datasets/prepare.py music                     # 只抓 FMA（需 pip install -r datasets/requirements.txt）
python datasets/prepare.py noise --per-type 1000     # 只生成噪音
python datasets/prepare.py piper                     # 只抓 Piper 模型
python datasets/prepare.py generic --count 10000     # 只生共用通用語音負樣本（piper，~30分）
```

`datasets/_gsc_raw/`（GSC 原始下載）重採樣完可自行刪除。

## 用自己的資料

- 一般語音負樣本：把 wav 丟進 `speech_commands/<任意子夾>/`（會被遞迴 glob 到）。
- 想帶入既有的 TTS 樣本：複製 wav 到 `tts/<label>/positive/{train,test}/` 和
  `tts/<label>/negative/{train,test}/`（44.1kHz mono）。
