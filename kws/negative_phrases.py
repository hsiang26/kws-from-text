# -*- coding: utf-8 -*-
"""
kws/negative_phrases.py
=======================
產生 hard-negative 要合成的「文字」清單。kws/generate.py 拿去餵 TTS 後端。

    near_miss_phrases(wake_word, rng) -> list[str]
        由喚醒詞衍生的近音詞 / 部分詞：
          * 喚醒詞的單字、各種真子片語（去頭 / 去尾 / 單字）
          * CMU 發音字典 (pronouncing) 找每個字的近音字，再跟其他字重組
          * 套模板：hey there X / my X / okay X / X please ...
          * 連續包含完整喚醒詞的片語會被剔掉
        像：hey sister / play assistant / insistent / persistent / hey resistant ...
        中文：目前只做「部分片語轉拼音」（xiao / bang shou / …），聲母韻母近音是 TODO。

    generic_phrases(rng) -> list[str]
        跟喚醒詞無關的通用英文詞 / 短句（數字 / 星期 / 語音助理指令 / 日常對話…）。
        給 datasets/generic_speech/ 這個「全部喚醒詞共用」的池用。
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------------------
# Group B：通用英文語料（硬編碼）
# --------------------------------------------------------------------------------------
_GENERIC_WORDS = [
    # 數字
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fifteen", "twenty", "thirty", "fifty",
    "hundred", "thousand", "first", "second", "third",
    # 星期 / 月份 / 時間
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "tonight",
    "minute", "hour", "week", "month", "year", "weekend", "midnight", "noon",
    # 顏色
    "red", "orange", "yellow", "green", "blue", "purple", "black", "white",
    "brown", "grey", "pink", "silver", "golden",
    # 常見名詞
    "water", "coffee", "table", "window", "door", "kitchen", "bedroom", "office",
    "phone", "camera", "battery", "screen", "keyboard", "speaker", "headphone",
    "weather", "temperature", "traffic", "news", "music", "movie", "podcast",
    "alarm", "timer", "calendar", "reminder", "message", "email", "photo",
    "light", "lamp", "heater", "fan", "television", "computer", "network",
    "street", "city", "country", "airport", "station", "hospital", "school",
    "dog", "cat", "bird", "tree", "flower", "mountain", "river", "ocean",
    "family", "friend", "teacher", "doctor", "driver", "manager", "student",
    # 常見動詞 / 形容詞
    "open", "close", "start", "stop", "pause", "resume", "cancel", "confirm",
    "increase", "decrease", "brighten", "dim", "louder", "quieter", "repeat",
    "hello", "goodbye", "thanks", "please", "sorry", "welcome", "okay", "yeah",
    "morning", "hungry", "tired", "happy", "ready", "busy", "quiet", "loud",
    "beautiful", "terrible", "expensive", "important", "different", "possible",
]

_GENERIC_PHRASES = [
    # 語音助理常見指令（刻意跟喚醒詞無關）
    "turn on the light", "turn off the lights", "turn up the volume",
    "turn down the volume", "set a timer for ten minutes", "set an alarm for seven",
    "what time is it", "what's the weather today", "what's the date today",
    "play some music", "play the next song", "pause the music", "stop the music",
    "skip this track", "resume playback", "add milk to my shopping list",
    "remind me to call mom", "call the office", "send a message to john",
    "read my messages", "check my calendar", "how far is the airport",
    "navigate to the train station", "how do you say hello in french",
    "tell me a joke", "give me the news", "how tall is mount everest",
    "convert ten dollars to euros", "spell the word necessary",
    "what's on my schedule tomorrow", "cancel my nine o'clock meeting",
    "start a workout", "how many steps did i take today",
    "lower the temperature", "raise the heat a little", "close the garage door",
    "lock the front door", "is it going to rain tomorrow",
    "set the thermostat to seventy degrees", "turn off everything downstairs",
    # 一般對話 / 問句 / 問候
    "good morning everyone", "good afternoon", "good evening", "how are you doing",
    "nice to meet you", "see you later", "have a great day", "thank you so much",
    "i really appreciate it", "let me think about that", "that sounds good to me",
    "i am not sure about this", "can you help me with something",
    "where did i put my keys", "the coffee is getting cold",
    "we should leave in five minutes", "the meeting starts at noon",
    "traffic is really bad today", "the wifi is down again",
    "did you watch the game last night", "i need to charge my phone",
    "the printer is out of paper", "let's grab lunch tomorrow",
    "the package should arrive on friday", "remember to water the plants",
    "it is a beautiful day outside", "the movie was way too long",
    "please close the window it is cold", "the battery is almost empty",
    "turn left at the next intersection", "the report is due next week",
]

# --------------------------------------------------------------------------------------
# CMU 發音字典近音搜尋
# --------------------------------------------------------------------------------------
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
           "IH", "IY", "OW", "OY", "UH", "UW"}


def _phones(word):
    import pronouncing
    p = pronouncing.phones_for_word(word.lower())
    return p[0] if p else None


def _near_words(word, rng, limit=12):
    """回傳跟 word 發音相近的真實英文單字清單（不含 word 本身 / 同音字）。"""
    try:
        import pronouncing
    except ImportError:
        return []

    word = word.lower()
    phones = _phones(word)
    if not phones:
        return []

    tokens = phones.split()
    cand = set()

    # 1) 押韻字
    try:
        for w in pronouncing.rhymes(word):
            if w.isalpha():
                cand.add(w)
    except Exception:
        pass

    # 2) 把 1 個音素替成 wildcard 後在字典裡搜（近音、非押韻）
    for i in range(len(tokens)):
        pat_tokens = list(tokens)
        # 去掉重音數字讓 pattern 寬鬆一點
        base = re.sub(r"\d", "", tokens[i])
        pat_tokens[i] = r"[A-Z]+[012]?" if base in _VOWELS else r"[A-Z]+"
        pattern = "^" + " ".join(re.sub(r"(?<=[A-Z])\d", r"[012]?", t) if j != i else t
                                 for j, t in enumerate(pat_tokens)) + "$"
        try:
            for w in pronouncing.search(pattern):
                if w.isalpha() and w != word:
                    cand.add(w)
        except Exception:
            continue

    homophones = set()
    try:
        homophones = {w for w in pronouncing.search("^" + re.escape(phones) + "$")}
    except Exception:
        pass

    out = [w for w in cand
           if w != word and w not in homophones and 3 <= len(w) <= 14]
    rng.shuffle(out)
    return out[:limit]


# --------------------------------------------------------------------------------------
# 由喚醒詞衍生 near-miss 片語
# --------------------------------------------------------------------------------------
_TEMPLATES = ["hey there {w}", "okay {w}", "my {w}", "the {w}", "a {w}",
              "{w} please", "{w} now", "hello {w}", "say {w}", "play {w}"]


def _partial_phrases(words):
    """喚醒詞的真子片語：單字、去頭、去尾、相鄰子序列（不含完整片語）。"""
    out = set()
    n = len(words)
    for w in words:
        out.add(w)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if (i, j) == (0, n):
                continue
            out.add(" ".join(words[i:j]))
    return {p for p in out if p.strip()}


def _contains_wake_word(phrase, words):
    """phrase 的 token 序列裡是否「連續」出現完整喚醒詞。
    這種片語 1.0s crop 後幾乎等於正樣本，必須從負樣本剔掉。"""
    toks = phrase.split()
    n = len(words)
    return any(toks[k:k + n] == words for k in range(len(toks) - n + 1))


def _build_latin(wake_word, rng):
    words = [w for w in re.split(r"\s+", wake_word.strip().lower()) if w]
    near_miss = set()

    # 部分片語（單字、去頭、去尾、相鄰子序列）
    partials = _partial_phrases(words)
    near_miss |= partials

    # 逐字近音，重組回完整長度的片語（"hey" -> say/play/dismay + "assistant"）
    per_word_near = {i: _near_words(w, rng, limit=14) for i, w in enumerate(words)}
    for i, alts in per_word_near.items():
        for alt in alts:
            variant = list(words)
            variant[i] = alt
            near_miss.add(" ".join(variant))
            near_miss.add(alt)  # 近音單字本身也算（"sister" "system" "assistance"）

    # 兩個位置同時換（多字詞才有意義，限量避免爆炸）
    if len(words) >= 2:
        i, j = 0, len(words) - 1
        for a in per_word_near.get(i, [])[:5]:
            for b in per_word_near.get(j, [])[:5]:
                variant = list(words)
                variant[i], variant[j] = a, b
                near_miss.add(" ".join(variant))

    # 模板：只套在「部分片語 / 近音單字」上，絕不套完整喚醒詞
    seeds = list(partials) + [a for alts in per_word_near.values() for a in alts]
    rng.shuffle(seeds)
    for s in seeds[:40]:
        for t in _TEMPLATES:
            near_miss.add(t.format(w=s))

    # 剔掉 = 喚醒詞、或連續包含完整喚醒詞的片語
    near_miss = sorted(
        p for p in near_miss
        if p.strip() and not _contains_wake_word(p, words))
    rng.shuffle(near_miss)
    return near_miss


def _build_cjk(wake_word, rng):
    """
    中文喚醒詞的 hard negatives。

    設計（跟 kws/generate.py 的 pick_backends 搭配）：中文負樣本一律用 **piper 英文合成**，
    所以這裡回傳的都是**拼音 / 英文**字串。

    目前只做「喚醒詞的部分片語（去頭 / 去尾 / 單字）轉拼音」——
    小幫手 → xiao / bang / shou / xiao bang / bang shou。
    聲母 / 韻母 / 聲調近音替換之後再加（見檔尾 TODO）。
    """
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        raise SystemExit("中文喚醒詞需要 pypinyin：pip install pypinyin")

    syl = [s.lower() for s in lazy_pinyin(wake_word) if s.strip().isalpha()]
    if not syl:
        return []
    full = " ".join(syl)
    n = len(syl)
    out = set()
    for i in range(n):
        for j in range(i + 1, n + 1):
            if (i, j) == (0, n):
                continue
            out.add(" ".join(syl[i:j]))
    out = sorted(p for p in out if p and p != full)
    rng.shuffle(out)
    return out
    # TODO(中文近音): 用 pypinyin 的聲母/韻母 拆解，做
    #   - 換聲母：xiao -> shao/qiao/miao ...
    #   - 換韻母：bang -> ben/bin/ban ...
    #   - 換聲調（piper 英文無聲調，意義不大，但保留給之後 zh piper 版）
    #   再用一份「常見中文近音真詞」表回查（小幫手 -> 小胖手 / 小幫忙 / 笑幫手），
    #   轉拼音後回傳。目前 --wake-word 中文只做「部分片語拼音」。


# --------------------------------------------------------------------------------------
# 對外入口
# --------------------------------------------------------------------------------------
def near_miss_phrases(wake_word, rng):
    """喚醒詞衍生的近音 / 部分詞片語清單（英文用 CMU 字典；中文目前只做拼音部分詞）。"""
    is_latin = all(ord(c) < 128 for c in wake_word)
    return _build_latin(wake_word, rng) if is_latin else _build_cjk(wake_word, rng)


def generic_phrases(rng):
    """跟喚醒詞無關的通用英文詞 / 短句（給 datasets/generic_speech/ 共用池用）。"""
    g = list(_GENERIC_WORDS) + list(_GENERIC_PHRASES)
    rng.shuffle(g)
    return g


if __name__ == "__main__":
    # 手動檢查：python -m kws.negative_phrases "hey assistant"
    import random
    import sys

    ww = sys.argv[1] if len(sys.argv) > 1 else "hey assistant"
    rng = random.Random(42)
    near = near_miss_phrases(ww, rng)
    gen = generic_phrases(rng)
    print(f"wake word: {ww!r}")
    print(f"\nnear_miss ({len(near)}):")
    for p in near[:60]:
        print("  ", p)
    print(f"\ngeneric ({len(gen)}):")
    for p in gen[:20]:
        print("  ", p)
