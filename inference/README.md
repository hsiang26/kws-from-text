# inference/

拿 `models/<run>/model_int8.tflite` 跑推論。特徵提取跟訓練共用 `wuw/features.py`
（44.1kHz、1.0 秒、MFCC-10），所以分數跟訓練時一致。

每個 run 資料夾裡的 `metadata.json` 帶了推論需要的一切：取樣率 / 特徵參數 /
量化 scale·zero-point / 正類 index / 建議門檻 / 喚醒詞原文。

## 離線評分（`infer.py`）

```bash
python inference/infer.py --input clip.wav
python inference/infer.py --input datasets/tts/hey_assistant/positive/test
python inference/infer.py --model models/hey_assistant_20260908_231511 --input x.wav --threshold 0.7
```

- `--model` 省略 = 用 `models/` 裡最新的一個 run。
- 傳資料夾 = 遞迴掃所有 `.wav`。
- 若路徑判斷得出 ground truth（含 `positive` / `negative` / `pos*` / `neg*`），
  會順便算 Precision / Recall / FAR / F1。

### 用 labels.csv 測試集

```bash
python inference/infer.py --labels-csv test_database/labels.csv --input test_database
```

`labels.csv` 欄位（`relative_path` 相對 `--input`）：

```
filename,relative_path,label,category
clip1.wav,positive/clip1.wav,positive,real
clip2.wav,negative/clip2.wav,negative,tts
```

輸出：整體 Precision / Recall / FAR / F1 / Accuracy + 各 `category` 準確率 + 完整 FN / FP 清單。

`run_all.py --test-database <dir>` 則是訓練「後」自動跑同一套評估（走 `kws/train.py` 內建的
`evaluate_external_test_set`，會多寫一份報告到 run 資料夾）。

## 麥克風即時 demo（`mic_demo.py`）

```bash
pip install -r requirements-mic.txt      # sounddevice
python inference/mic_demo.py
```

滑動視窗 1.0s、每 0.2s 推論一次，連續兩次超過門檻就印 `🔔 WAKE`。
`--device` 選輸入裝置，`--threshold` 調靈敏度，Ctrl+C 結束。
