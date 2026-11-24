"""Corpus BLEU-1..4 against all reference captions per image."""
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from visionscribe.vocab import tokenize

_WEIGHTS = {
    "bleu1": (1, 0, 0, 0),
    "bleu2": (0.5, 0.5, 0, 0),
    "bleu3": (1 / 3, 1 / 3, 1 / 3, 0),
    "bleu4": (0.25, 0.25, 0.25, 0.25),
}


def bleu_scores(references: list[list[str]], hypotheses: list[str]) -> dict[str, float]:
    refs = [[tokenize(r) for r in rs] for rs in references]
    hyps = [tokenize(h) for h in hypotheses]
    smooth = SmoothingFunction().method1
    return {k: round(corpus_bleu(refs, hyps, weights=w, smoothing_function=smooth), 4)
            for k, w in _WEIGHTS.items()}
