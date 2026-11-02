import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.models.compare import (GRID, LABELS, best_by_mean, choose_tfidf, classifier, compare,
                                configurations, key, paired_comparison, pretokenized_vectorizer,
                                select, vectorizer)
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


def test_pretokenization_matches_with_min_and_max_df():
    texts = ['factura pago cobro', 'factura envío paquete', 'factura oferta cupón', 'pago envío']
    tokenized = [' '.join(SpanishTokenizer('stemming')(text)) for text in texts]
    direct, precomputed = vectorizer('stemming', 2, 2, 0.9), pretokenized_vectorizer(2, 2, 0.9)
    np.testing.assert_allclose(direct.fit_transform(texts).toarray(), precomputed.fit_transform(tokenized).toarray())
    assert direct.vocabulary_ == precomputed.vocabulary_


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
    assert len(first) == len(configurations()) == 4 * (len(GRID['naive_bayes']) + len(GRID['logistic_regression']))
    assert all(r['status'] == 'ok' and len(r['fold_f1_macro']) == 3 for r in first)
    assert select(first) == select(second)
    assert isinstance(classifier('naive_bayes', 1), MultinomialNB)
    assert isinstance(classifier('logistic_regression', 1), LogisticRegression)


def test_grid_expands_without_duplicates():
    configs = configurations()
    assert len({key(c) for c in configs}) == len(configs)
    per_model = {m: sum(c['model'] == m for c in configs) for m in ('naive_bayes', 'logistic_regression')}
    assert per_model == {'naive_bayes': 4 * len(GRID['naive_bayes']), 'logistic_regression': 4 * len(GRID['logistic_regression'])}
    assert {c['class_weight'] for c in configs if c['model'] == 'logistic_regression'} == {None, 'balanced'}
    assert all(c['class_weight'] is None for c in configs if c['model'] == 'naive_bayes')


def test_classifier_respects_c_and_class_weight():
    model = classifier('logistic_regression', 30.0, 'balanced')
    assert model.C == 30.0 and model.class_weight == 'balanced' and model.random_state == 42
    assert classifier('logistic_regression', 1.0).class_weight is None


def result(model, mean, std=0.01, value=1.0, ngrams=1, method='stemming', weight=None, status='ok'):
    return {'method': method, 'ngrams': ngrams, 'model': model,
            'parameter_name': 'alpha' if model == 'naive_bayes' else 'C', 'parameter_value': value,
            'class_weight': weight, 'fold_f1_macro': [mean] * 3, 'errors': [], 'status': status,
            'f1_macro_mean': mean if status == 'ok' else None, 'f1_macro_std': std if status == 'ok' else None}


def test_selection_prefers_simplest_within_one_std():
    nb = result('naive_bayes', 0.960, value=0.5)
    lr = result('logistic_regression', 0.965, value=30.0, weight='balanced')
    assert select([lr, nb]) is nb  # la diferencia (0,005) es menor que una desviación (0,01)


def test_selection_takes_best_when_gap_exceeds_one_std():
    nb = result('naive_bayes', 0.950, value=0.5)
    lr = result('logistic_regression', 0.970, value=30.0, weight='balanced')
    assert select([nb, lr]) is lr
    assert best_by_mean([nb, lr]) is lr


def test_selection_tiebreaks_among_candidates():
    heavy = result('logistic_regression', 0.970, value=100.0)
    light = result('logistic_regression', 0.968, value=10.0)
    weighted = result('logistic_regression', 0.970, value=10.0, weight='balanced')
    bigram = result('logistic_regression', 0.971, value=10.0, ngrams=2)
    assert select([heavy, light, weighted, bigram]) is light


def test_selection_excludes_failed_configurations():
    failed = result('naive_bayes', 0.99, status='failed')
    ok = result('logistic_regression', 0.90, value=10.0)
    assert select([failed, ok]) is ok
    assert select([failed]) is None


def test_paired_comparison_uses_best_per_model():
    nb = result('naive_bayes', 0.95, value=0.5)
    lr = result('logistic_regression', 0.97, value=30.0)
    paired = paired_comparison([nb, lr, result('logistic_regression', 0.90, value=1.0)])
    assert paired['logistic_regression']['f1_macro_mean'] == 0.97
    assert abs(paired['mean_difference'] - 0.02) < 1e-12
    assert paired_comparison([nb]) is None


def test_tfidf_defaults_kept_unless_gain_exceeds_one_std():
    def row(min_df, max_df, mean):
        return {'min_df': min_df, 'max_df': max_df, 'status': 'ok', 'f1_macro_mean': mean, 'f1_macro_std': 0.01}
    assert choose_tfidf([row(1, 1.0, 0.95), row(2, 1.0, 0.955)]) == {'min_df': 1, 'max_df': 1.0}
    assert choose_tfidf([row(1, 1.0, 0.95), row(2, 0.9, 0.97)]) == {'min_df': 2, 'max_df': 0.9}


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
    monkeypatch.setattr(module, 'compare', lambda texts, labels, **kwargs: [selected])
    monkeypatch.setattr(module, 'load_data', record_load)
    monkeypatch.setattr(sys, 'argv', ['compare', '--output-dir', str(tmp_path / 'out')])
    module.main()
    assert paths == ['train.csv', 'validation.csv']
    report = json.loads((tmp_path / 'out/report.json').read_text(encoding='utf-8'))
    assert report['test_used'] is False
    for field in ('model', 'method', 'ngrams', 'parameter_name', 'parameter_value', 'f1_macro_mean', 'f1_macro_std'):
        assert field in report['selected']  # contrato con /metrics del backend
    assert report['selected']['min_df'] == 1 and report['vocabulary_size'] > 0
    assert report['paired_comparison'] is None and report['runner_up_other_model'] is None
    assert set(report['validation']['confusion_matrix_labels']) == set(LABELS)


def test_existing_pipeline_is_not_overwritten_without_flag(tmp_path, monkeypatch):
    import sys
    import pytest
    import src.models.compare as module

    (tmp_path / 'pipeline.joblib').write_bytes(b'previo')
    monkeypatch.setattr(sys, 'argv', ['compare', '--output-dir', str(tmp_path)])
    with pytest.raises(SystemExit, match='ya existe'):
        module.main()
    assert (tmp_path / 'pipeline.joblib').read_bytes() == b'previo'
