# -*- coding: utf-8 -*-
"""
kws/tts/base.py —— TTS 後端介面。

每個後端把「要合成的文字」× 自己的參數多樣性（語者 / 語速 / 韻律…）展開成一批 job，
再各自用自己的方式（asyncio / thread pool）平行合成，yield 出原始波形。
下游 (kws/generate.py) 統一做修剪、增強、放進 1 秒視窗、寫檔。
"""

PUNCT_LATIN = ["", ".", ",", "?", "!"]
PUNCT_CJK = ["", "。", "，", "？", "！"]


class TTSBackend:
    name = "base"

    def plan(self, texts, count, rng, is_positive, punct_list):
        """
        texts : list[(base_text, group)]  —— 正樣本通常只有 1 條 (喚醒詞)；負樣本是一堆片語。
        回傳 count 個 job dict（不夠時可少於 count）。每個 job：
            {
              "text":   str,     # 實際要送 TTS 的字（已含句尾標點）
              "group":  str,     # positive / near_miss / generic
              "voice":  str,     # 給 manifest 看的語者標識（edge 語者名 / piper speaker_id）
              "params": dict,    # 後端專屬合成參數，也寫進 manifest
              "augment": bool,   # 事後 librosa 增強（組合用完後才 True）
            }
        """
        raise NotImplementedError

    def synth_all(self, jobs, concurrency):
        """yield (job, wav_float32_mono | None, sample_rate | None, err_str | None)。"""
        raise NotImplementedError

    def describe(self):
        return self.name
