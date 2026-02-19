# -*- coding: utf-8 -*-
import re
import evaluate
from datasets import Features, Value, Sequence

# 简化版归一化：小写、去标点、去冠词、常见收缩词、数字词→数字
_CONTRACTIONS = {
    "aint":"ain't","arent":"aren't","cant":"can't","couldnt":"couldn't","couldve":"could've",
    "didnt":"didn't","doesnt":"doesn't","dont":"don't","hadnt":"hadn't","hasnt":"hasn't",
    "havent":"haven't","hed":"he'd","hes":"he's","howd":"how'd","howll":"how'll","hows":"how's",
    "id":"i'd","im":"i'm","ive":"i've","isnt":"isn't","itd":"it'd","itll":"it'll","its":"it's",
    "shouldnt":"shouldn't","thatd":"that'd","thats":"that's","theres":"there's",
    "theyre":"they're","theyve":"they've","wasnt":"wasn't","werent":"weren't",
    "whatd":"what'd","whats":"what's","whod":"who'd","wholl":"who'll","whos":"who's",
    "wont":"won't","wouldnt":"wouldn't","yall":"y'all","youd":"you'd","youre":"you're","youve":"you've",
}
_ARTICLES = {"a", "an", "the"}
_NUM_MAP = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10","eleven":"11","twelve":"12"
}

# 粗略去标点（保留字母数字下划线用于 \b 词界定）
_PUNCT = re.compile(r"[^\w\s]")

def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = _PUNCT.sub("", s)
    words = [w for w in s.split() if w not in _ARTICLES]
    words = [_CONTRACTIONS.get(w, w) for w in words]
    words = [_NUM_MAP.get(w, w) for w in words]
    return " ".join(words)

def _includes_match(pred_norm: str, ref_norm: str) -> bool:
    """
    只要 pred_norm 中“包含” ref_norm 的完整词序列（按词边界匹配）即为命中。
    例：pred='no this is not a creamy soup'，ref='no' -> 命中
    """
    if not ref_norm:
        return False
    # 词边界匹配，避免 'not' 命中 'nothing'
    pattern = r"\b" + re.escape(ref_norm) + r"\b"
    return re.search(pattern, pred_norm) is not None

def _score_one(pred: str, refs: list[str]) -> float:
    """pred: 预测答案; refs: 该题10个标注答案的列表"""
    p = _normalize(pred)
    gs = [_normalize(x) for x in refs if x is not None]
    n = sum(_includes_match(p, g) for g in gs)
    return min(n / 3.0, 1.0)

class VQAScore(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=(
                "VQAv2-style accuracy with includes-match: "
                "acc = min(#match/3, 1). A reference counts as matched if "
                "its normalized text is contained in the normalized prediction "
                "at word boundaries."
            ),
            citation="https://visualqa.org/evaluation.html",
            inputs_description=(
                "predictions: List[str]\n"
                "references:  List[List[str]]  # 每题10个参考答案\n"
            ),
            features=Features({
                "predictions": Value("string"),
                "references":  Sequence(Value("string")),
            }),
            homepage="https://visualqa.org/",
        )

    def _compute(self, predictions, references, return_per_question: bool = False):
        per_q = [_score_one(p, r) for p, r in zip(predictions, references)]
        acc = float(sum(per_q) / max(len(per_q), 1))
        out = {"vqa_accuracy": acc}
        if return_per_question:
            out["per_question"] = per_q
        return out

def _metric(**kwargs):
    return VQAScore()
