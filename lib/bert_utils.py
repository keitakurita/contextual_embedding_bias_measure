import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *

from allennlp.common.util import get_spacy_model
from spacy.attrs import ORTH
from spacy.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.tokenizers import Token
from allennlp.data import Vocabulary

T = TypeVar('T')
nlp = get_spacy_model("en_core_web_sm", pos_tags=False, parse=True, ner=False)
nlp.tokenizer.add_special_case("[MASK]", [{ORTH: "[MASK]"}])

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

def flatten(x: List[List[T]]) -> List[T]:
    return [item for sublist in x for item in sublist]

def spacy_tok(s: str) -> List[str]:
    return [w.text for w in nlp(s)]

class TokenizationError(Exception):
    pass

class BertPreprocessor:
    def __init__(self, model_type: str, max_seq_len: int=128):
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        self.token_indexer = PretrainedBertIndexer(
            pretrained_model=self.model_type,
            max_pieces=self.max_seq_len,
            do_lowercase=True,
        )
        self.vocab = Vocabulary()
        self.token_indexer._add_encoding_to_vocabulary(self.vocab)
        self.full_vocab = {v:k for k, v in self.token_indexer.vocab.items()}

    def tokenize(self, x: str) -> List[Token]:
        return [Token(w) for w in flatten([
                self.token_indexer.wordpiece_tokenizer(w)
                for w in spacy_tok(x)]
        )[:self.max_seq_len]]

    def index_to_token(self, idx: int) -> str:
        return self.full_vocab[idx]

    def indices_to_tokens(self, indices: Iterable[int]) -> List[str]:
        return [self.index_to_word(x) for x in indices]

    def token_to_index(self, token: str,
                      accept_wordpiece: bool=False,
                      ) -> int:
        wordpieces = self.tokenize(token)
        if len(wordpieces) > 1 and not accept_wordpiece:
            raise TokenizationError(f"{token} is not a single wordpiece")
        else: token = wordpieces[0].text
        return self.token_indexer.vocab[token]

    def get_index(self, sentence: str,
                  word: str,
                  accept_wordpiece: bool=False,
                  last: bool=False) -> int:
        toks = self.tokenize(sentence)
        wordpieces = self.tokenize(word)
        if len(wordpieces) > 1 and not accept_wordpiece:
            raise TokenizationError(f"{word} is not a single wordpiece")
        else: word = wordpieces[0].text # use first wordpiece

        if not last:
            for i, t in enumerate(toks):
                if t.text == word:
                    return i + 1 # take the [CLS] token into account
        else:
            for i, t in enumerate(reversed(toks)):
                if t.text == word:
                    return len(toks) - 1 - i
        raise ValueError(f"No {word} tokenn tokens {toks} found")

    def to_bert_model_input(self, input_sentence: str) -> np.ndarray:
        input_toks = self.tokenize(input_sentence)
        batch = self.token_indexer.tokens_to_indices(input_toks, self.vocab, "tokens")
        token_ids = torch.LongTensor(batch["tokens"]).unsqueeze(0)
        return token_ids
