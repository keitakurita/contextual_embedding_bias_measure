from typing import *
import numpy as np
import torch
from collections import defaultdict
from pytorch_pretrained_bert import BertConfig, BertForMaskedLM
from .bert_utils import Config, BertPreprocessor

def softmax(arr, axis=1):
    e = np.exp(arr)
    return e / e.sum(axis=axis, keepdims=True)

class BiasScorer:
    def __init__(self, model_type="bert-base-uncased"):
        self.model = BertForMaskedLM.from_pretrained(model_type)
        self.model.eval() # Important! Disable dropout
        self.processor = BertPreprocessor(model_type, 128)
        self.cache = {}

    def get_logits(self, sentence: str) -> np.ndarray:
        return self.model(self.processor.to_bert_model_input(sentence))[0, :, :].cpu().detach().numpy()

    def get_mask_fill_logits(self,
                             sentence: str, words: Iterable[str],
                             use_last_mask=False, apply_softmax=True) -> Dict[str, float]:
        mask_i = self.processor.get_index(sentence, "[MASK]", last=use_last_mask, accept_wordpiece=True)
        logits = defaultdict(list)
        out_logits = self.get_logits(sentence)
        if apply_softmax:
            out_logits = softmax(out_logits)
        return {w: out_logits[mask_i, self.processor.token_to_index(w, accept_wordpiece=True)] for w in words}

    def bias_score(self,
                   sentence: str, gender_words: Iterable[Iterable[str]],
                   word: str, gender_comes_first=True, cache=False) -> Dict[str, float]:
        """
        Input a sentence of the form "GGG is XXX"
        XXX is a placeholder for the target word
        GGG is a placeholder for the gendered words (the subject)
        We will predict the bias when filling in the gendered words and
        filling in the target word.

        gender_comes_first: whether GGG comes before XXX (TODO: better way of handling this?)
        """
        # probability of filling [MASK] with "he" vs. "she" when target is "programmer"
        mwords, fwords = gender_words
        all_words = mwords + fwords
        subject_fill_logits = self.get_mask_fill_logits(
            sentence.replace("XXX", word).replace("GGG", "[MASK]"),
            all_words, use_last_mask=not gender_comes_first,
        )
        subject_fill_bias = np.log(sum(subject_fill_logits[mw] for mw in mwords)) - \
                            np.log(sum(subject_fill_logits[fw] for fw in fwords))
        # male words are simply more likely than female words
        # correct for this by masking the target word and measuring the prior probabilities
        bland_sentence = sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]")
        if bland_sentence in self.cache:
            return self.cache[bland_sentence]
        else:
            subject_fill_prior_logits = self.get_mask_fill_logits(
                bland_sentence,
                all_words, use_last_mask=gender_comes_first,
            )
            subject_fill_bias_prior_correction = \
                    np.log(sum(subject_fill_prior_logits[mw] for mw in mwords)) - \
                    np.log(sum(subject_fill_prior_logits[fw] for fw in fwords))
            if self.cache:
                self.cache[bland_sentence] = subject_fill_bias_prior_correction

        return {
            "stimulus": word,
            "bias": subject_fill_bias,
            "prior_correction": subject_fill_bias_prior_correction,
            "bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
           }
