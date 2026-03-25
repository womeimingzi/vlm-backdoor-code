# Copyright 2025 The Author.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# A lightweight CIDEr metric for HuggingFace Evaluate (compatible I/O with your ROUGE wrapper)

import math
from collections import Counter, defaultdict

import datasets
import numpy as np

import evaluate


_CITATION = """\
@inproceedings{vedantam2015cider,
  title={CIDEr: Consensus-based Image Description Evaluation},
  author={Vedantam, Ramakrishna and Lawrence Zitnick, C. and Parikh, Devi},
  booktitle={CVPR},
  year={2015}
}
"""

_DESCRIPTION = """\
CIDEr (Consensus-based Image Description Evaluation) measures consensus of a candidate sentence
with a set of reference sentences using TF-IDF weighted n-gram matching (n=1..4) and a length penalty.
This implementation is a dependency-light, close approximation to CIDEr-D used in COCO captioning.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (List[str]): candidate texts.
    references (List[str] or List[List[str]]): reference(s) for each candidate.
    n_gram (int): maximum n-gram (default 4).
    sigma (float): Gaussian penalty sigma for length difference (default 6.0).
    tokenizer (callable): optional callable(text) -> List[str]. If None, a simple whitespace-lower tokenizer is used.
    return_per_example (bool): if True, also return per-example CIDEr scores.

Returns:
    cider (float): mean CIDEr over the dataset.
    (optional) per_example (List[float]): per-example CIDEr when return_per_example=True.
"""

class Tokenizer:
    def __init__(self, fn):
        self.fn = fn
    def tokenize(self, text):
        return self.fn(text)

def _default_tokenize(s):
    # simple, case-insensitive, whitespace tokenization
    return s.lower().strip().split()

def _extract_ngrams(tokens, n_gram=4):
    """
    returns: dict n -> Counter of ngrams
    """
    out = {}
    T = len(tokens)
    for n in range(1, n_gram + 1):
        c = Counter(tuple(tokens[i:i+n]) for i in range(0, max(T - n + 1, 0)))
        out[n] = c
    return out

def _compute_df(ref_lists, n_gram=4):
    """
    Document frequency for each n-gram: number of samples whose ANY reference contains the n-gram.
    ref_lists: List[List[List[token]]], i.e., per-sample list of ref-token lists.
    """
    df = [defaultdict(int) for _ in range(n_gram + 1)]  # index by n
    N = len(ref_lists)
    for refs in ref_lists:
        # for this sample, collect unique ngrams across all refs
        seen_per_n = [set() for _ in range(n_gram + 1)]
        for ref in refs:
            grams = _extract_ngrams(ref, n_gram=n_gram)
            for n in range(1, n_gram + 1):
                seen_per_n[n].update(grams[n].keys())
        for n in range(1, n_gram + 1):
            for g in seen_per_n[n]:
                df[n][g] += 1
    return df  # list of dicts

def _tfidf_vec(ngrams_counter, df_n, N, eps=1e-12):
    """
    Build TF-IDF vector (dict) for a given n level.
    TF: raw count; IDF: log(N / (df+eps))
    """
    vec = {}
    for g, tf in ngrams_counter.items():
        df = df_n.get(g, 0)
        idf = math.log((N + eps) / (df + eps))
        vec[g] = tf * idf
    return vec

def _cosine_sim(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    # dot
    dot = 0.0
    for g, v in vec1.items():
        if g in vec2:
            dot += v * vec2[g]
    # norms
    n1 = math.sqrt(sum(v*v for v in vec1.values()))
    n2 = math.sqrt(sum(v*v for v in vec2.values()))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)

def _gaussian_length_penalty(len_cand, len_ref_avg, sigma=6.0):
    # As used in CIDEr-D: penalty = exp( - (len_cand - len_ref_avg)^2 / (2*sigma^2) )
    return math.exp(- ((len_cand - len_ref_avg) ** 2) / (2.0 * (sigma ** 2)))

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Cider(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence")),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/tylin/coco-caption"],
            reference_urls=["https://cocodataset.org/#captions-eval"],
        )

    def _compute(
        self,
        predictions,
        references,
        n_gram=2,
        sigma=6.0,
        tokenizer=None,
        return_per_example=False,
    ):
        # normalize multi-ref
        multi_ref = isinstance(references[0], list)
        if not multi_ref:
            references = [[r] for r in references]

        # tokenizer
        if tokenizer is not None:
            tok = Tokenizer(tokenizer).tokenize
        else:
            tok = _default_tokenize

        # tokenize all
        predictions = [p[len(prefix):].lstrip() if (prefix:= "this image shows").lower() in p.lower().strip().split(":")[0] else p for p in predictions]
        pred_tok = [tok(p) for p in predictions]
        ref_tok_lists = [[tok(r) for r in refs] for refs in references]

        # precompute DF over the whole corpus (per n)
        N = len(pred_tok)
        df = _compute_df(ref_tok_lists, n_gram=n_gram)

        # precompute reference stats per example
        ref_ngrams_per_ex = []
        ref_len_avg = []
        ref_tfidf_per_ex = []  # list of list (per-ref) of dicts per n
        for refs in ref_tok_lists:
            # avg reference length for penalty
            lengths = [len(r) for r in refs]
            ref_len_avg.append(float(np.mean(lengths)) if lengths else 0.0)

            # n-gram counters & TF-IDF vectors for each ref
            ngrams_refs = [ _extract_ngrams(r, n_gram=n_gram) for r in refs ]
            ref_ngrams_per_ex.append(ngrams_refs)

            ref_tfidf_n_list = []
            for rr in ngrams_refs:
                per_n = {}
                for n in range(1, n_gram + 1):
                    per_n[n] = _tfidf_vec(rr[n], df[n], N)
                ref_tfidf_n_list.append(per_n)
            ref_tfidf_per_ex.append(ref_tfidf_n_list)

        # compute cider for each example
        per_example = []
        for i in range(N):
            cand = pred_tok[i]
            cand_len = len(cand)
            cand_ngrams = _extract_ngrams(cand, n_gram=n_gram)

            # candidate TF-IDF vectors per n
            cand_tfidf = { n: _tfidf_vec(cand_ngrams[n], df[n], N) for n in range(1, n_gram + 1) }

            # similarity to each reference (average across refs), then average across n with equal weights
            scores_n = []
            for n in range(1, n_gram + 1):
                sims = []
                for per_ref in ref_tfidf_per_ex[i]:
                    sims.append(_cosine_sim(cand_tfidf[n], per_ref[n]))
                scores_n.append(float(np.mean(sims)) if sims else 0.0)

            # equal weight across n=1..4
            sim = float(np.mean(scores_n)) if scores_n else 0.0

            # length penalty
            lp = _gaussian_length_penalty(cand_len, ref_len_avg[i], sigma=sigma)

            per_example.append(sim * lp * 10.0)  # scale by 10 like CIDEr

        result = { "cider": float(np.mean(per_example)*100) if len(per_example) > 0 else 0.0 }
        if return_per_example:
            result["per_example"] = per_example
        return result
