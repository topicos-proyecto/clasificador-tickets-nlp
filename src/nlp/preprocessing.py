"""Transformaciones compartidas por entrenamiento, inferencia y trazas didácticas."""
from __future__ import annotations

import re
import unicodedata

from nltk.stem.snowball import SpanishStemmer
from spacy.lang.es.stop_words import STOP_WORDS

NEGATIONS = frozenset({'no', 'ni', 'nunca', 'jamás', 'jamas', 'tampoco', 'sin'})
SPANISH_STOP_WORDS = frozenset(STOP_WORDS) - NEGATIONS
URL = re.compile(r'(?i)\b(?:https?://|www\.)\S+')
TOKEN = re.compile(r'[^\W_]+', re.UNICODE)


def normalize(text: str) -> str:
    """Normaliza Unicode y minúsculas; elimina URL, puntuación y emojis; conserva tildes y dígitos."""
    if not isinstance(text, str):
        raise TypeError('El texto debe ser una cadena.')
    return ' '.join(TOKEN.findall(URL.sub(' ', unicodedata.normalize('NFKC', text).casefold())))


class SpanishTokenizer:
    """Tokenizador serializable de TF-IDF; carga el modelo de spaCy solo al lematizar."""

    def __init__(self, method: str = 'stemming', remove_stopwords: bool = True):
        if method not in ('stemming', 'lemmatization'):
            raise ValueError('method debe ser stemming o lemmatization')
        self.method = method
        self.remove_stopwords = remove_stopwords
        self._nlp = None
        self._stemmer = SpanishStemmer()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_nlp'] = None
        return state

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
            except OSError as exc:
                raise RuntimeError('Instala el modelo es_core_news_sm 3.8.0; consulta el README.') from exc
        from spacy.tokens import Doc
        doc = self._nlp(Doc(self._nlp.vocab, words=tokens))
        return [token.lemma_.casefold() for token in doc]

    def trace(self, text: str) -> dict:
        normalized = normalize(text)
        tokens = normalized.split()
        removed = [t for t in tokens if self.remove_stopwords and t in SPANISH_STOP_WORDS]
        kept = [t for t in tokens if not self.remove_stopwords or t not in SPANISH_STOP_WORDS]
        transformed = ([self._stemmer.stem(t) for t in kept] if self.method == 'stemming'
                       else self._lemmatize(kept))
        return {'normalized': normalized, 'tokens': tokens, 'stopwords_removed': removed,
                'tokens_kept': kept, 'tokens_transformed': transformed}

    def __call__(self, text: str) -> list[str]:
        return self.trace(text)['tokens_transformed']
