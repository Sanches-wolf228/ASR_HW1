from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder

class Hypothesis(NamedTuple):
    text: str
    prob: float

class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        if len(inds) == 0:
            return ""
        res = []
        if inds[0] != 0:
            res.append(inds[0])
        for i in range(1, len(inds)):
            if (inds[i] != 0) and (inds[i] != inds[i-1]):
                res.append(inds[i])
        return ''.join([self.ind2char[x] for x in res])

    def _extend_and_merge(self, dp, prob):
        new_dp = defaultdict(float)
        for (res, last_char), v in dp.items():
            for i in range(len(prob)):
                if self.ind2char[i] == last_char:
                    new_dp[(res, last_char)] += v * prob[i]
                else:
                    new_dp[((res + last_char).replace(self.EMPTY_TOK, ''), self.ind2char[i])] += v * prob[i]
        return new_dp

    def _cut_beams(self, dp, beam_size):
        return dict(list(sorted(dp.items(), key = lambda x: x[1]))[-beam_size:])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        if probs_length == None:
            probs_length = char_length
        assert voc_size == len(self.ind2char)
        dp = {
            ("", self.EMPTY_TOK) : 1.0
        }
        for i in range(min(char_length, probs_length)):
            prob = probs[i]
            dp = self._extend_and_merge(dp, prob)
            dp = self._cut_beams(dp, beam_size)

        dp = list(sorted([((res + last_char).strip().replace(self.EMPTY_TOK, ''), proba) for (res, last_char), proba in dp.items()],
                         key = lambda x: -x[1]))
        
        return [Hypothesis(res, proba) for res, proba in dp]
