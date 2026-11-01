import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.models.compare import (LABELS, classifier, compare, pretokenized_vectorizer,
                                ranking, vectorizer)
from src.nlp.preprocessing import SpanishTokenizer


def test_pretokenization_matches_serialized_pipeline_features():
    texts = ['No puedo aplicar el cupón de descuento', 'Mis pedidos llegaron hoy']
    for method in ('stemming', 'lemmatization'):
        tokenized = [' '.join(SpanishTokenizer(method)(text)) for text in texts]
        for ngrams in (1, 2):
            direct, precomputed = vectorizer(method, ngrams), pretokenized_vectorizer(ngrams)
            a = direct.fit_transform(texts)
            b = precomputed.fit_transform(tokenized)
            assert direct.vocabulary_ == precomputed.vocabulary_
            np.testing.assert_allclose(a.toarray(), b.toarray())


def test_vectorizer_does_not_learn_validation_vocabulary():
    tfidf = pretokenized_vectorizer(2)
    tfidf.fit(['factura pago', 'envio paquete'])
    before = dict(tfidf.vocabulary_)
    tfidf.transform(['palabraexclusiva validacion'])
    assert tfidf.vocabulary_ == before
    assert 'palabraexclusiva' not in tfidf.vocabulary_


def test_stratified_comparison_is_complete_and_reproducible():
    words = ['ayuda horario', 'factura pago', 'envío paquete', 'oferta cupón',
             'error aplicación', 'precio compra']
    texts = [f'{word} mensaje {n}' for word in words for n in range(3)]
    labels = [label for label in LABELS for _ in range(3)]
    first = compare(texts, labels)
    second = compare(texts, labels)
    assert first == second
    assert len(first) == 16
    assert all(r['status'] == 'ok' and len(r['fold_f1_macro']) == 3 for r in first)
    assert min(first, key=ranking) == min(second, key=ranking)
    assert isinstance(classifier('naive_bayes', 1), MultinomialNB)
    assert isinstance(classifier('logistic_regression', 1), LogisticRegression)


def test_command_reads_only_train_and_validation(tmp_path, monkeypatch):
    import csv
    import json
    import sys
    import src.models.compare as module

    data = tmp_path / 'inputs'
    data.mkdir()
    for name in ('train', 'validation'):
        with (data / f'{name}.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'department'])
            writer.writeheader()
            for label in LABELS:
                writer.writerow({'text': f'consulta sobre {label}', 'department': label})
    selected = {'method': 'stemming', 'ngrams': 1, 'model': 'naive_bayes',
                'parameter_name': 'alpha', 'parameter_value': 0.5,
                'fold_f1_macro': [0.5, 0.5, 0.5], 'errors': [], 'status': 'ok',
                'f1_macro_mean': 0.5, 'f1_macro_std': 0.0}
    paths = []
    actual_loader = module.load_data
    def record_load(path):
        paths.append(path.name)
        return actual_loader(path)
    monkeypatch.setattr(module, 'DATA', data)
    monkeypatch.setattr(module, 'compare', lambda texts, labels: [selected])
    monkeypatch.setattr(module, 'load_data', record_load)
    monkeypatch.setattr(sys, 'argv', ['compare', '--output-dir', str(tmp_path / 'out')])
    module.main()
    assert paths == ['train.csv', 'validation.csv']
    report = json.loads((tmp_path / 'out/report.json').read_text(encoding='utf-8'))
    assert report['test_used'] is False
    assert set(report['validation']['confusion_matrix_labels']) == set(LABELS)
