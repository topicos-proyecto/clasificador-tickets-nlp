import joblib
import numpy as np
import pytest

from src.models.train import build_pipeline
from src.nlp.preprocessing import SpanishTokenizer, normalize


def test_normalization_urls_emoji_and_accents():
    assert normalize('¡No pagué 20! Visita https://ejemplo.com 😊') == 'no pagué 20 visita'
    assert normalize('') == ''
    assert normalize('😀!!!') == ''
    with pytest.raises(TypeError):
        normalize(None)


def test_stopwords_and_negations_are_explained():
    tokenizer = SpanishTokenizer()
    trace = tokenizer.trace('No puedo usar el cupón y nunca recibí descuento')
    assert {'no', 'nunca'} <= set(trace['tokens_kept'])
    assert {'el', 'y'} <= set(trace['stopwords_removed'])
    assert tokenizer('') == []
    assert SpanishTokenizer(remove_stopwords=False).trace('el cupón')['tokens_kept'] == ['el', 'cupón']
    assert SpanishTokenizer()('comprando productos')[0] == 'compr'


def test_lemmatization_uses_spanish_model():
    tokenizer = SpanishTokenizer('lemmatization', remove_stopwords=False)
    trace = tokenizer.trace('Las ofertas llegaron')
    assert trace['tokens_transformed'] == ['el', 'oferta', 'llegar']
    assert tokenizer('') == []


def test_pipeline_fits_only_training_and_roundtrips(tmp_path):
    model = build_pipeline(ngrams=2)
    model.fit(['factura cobrada', 'cupón descuento'], ['Finanzas', 'Marketing'])
    tfidf = model.named_steps['tfidf']
    before = dict(tfidf.vocabulary_)
    examples = ['palabrainventada cupón', 'factura cobrada']
    predictions = model.predict(examples)
    probabilities = model.predict_proba(examples)
    assert tfidf.vocabulary_ == before
    assert 'palabrainvent' not in before
    assert any(' ' in term for term in before)
    path = tmp_path / 'pipeline.joblib'
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert restored.predict(examples).tolist() == predictions.tolist()
    np.testing.assert_allclose(restored.predict_proba(examples), probabilities)
