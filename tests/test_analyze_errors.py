import csv
import json

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.models import analyze_errors as ae
from src.models.compare import classifier, vectorizer
from src.models.train import LABELS

CORPUS = {
    'Atención al Cliente': ['horario de atención', 'canal de contacto', 'quiero hablar con alguien', 'orientación general'],
    'Finanzas': ['factura pendiente de pago', 'cobro duplicado tarjeta', 'reembolso de mi compra', 'recibo del pago'],
    'Logística': ['mi paquete no llegó', 'rastrear el envío', 'entrega retrasada', 'dirección del envío'],
    'Marketing': ['cupón de descuento no funciona', 'promoción de noviembre', 'campaña de referidos', 'oferta anunciada'],
    'Soporte TI': ['error al iniciar sesión', 'la aplicación falla', 'página no carga', 'no puedo acceder al sistema'],
    'Ventas': ['cotizar diez licencias', 'precio del plan empresarial', 'quiero contratar', 'comparar planes'],
}
TEXTS = [t for label in LABELS for t in CORPUS[label]]
TARGETS = [label for label in LABELS for _ in CORPUS[label]]
EVAL = [('horario de atención', 'Atención al Cliente'),
        ('no se descargó la factura y la página falla', 'Finanzas'),
        ('mi paquete llegó pero el cupón de descuento no funciona', 'Logística'),
        ('xqzw plmk', 'Ventas')]


def fit(model_name, value):
    model = Pipeline([('tfidf', vectorizer('stemming', 1)), ('classifier', classifier(model_name, value))])
    return model.fit(TEXTS, TARGETS)


@pytest.fixture(scope='module')
def nb():
    return fit('naive_bayes', 0.5)


@pytest.fixture(scope='module')
def lr():
    return fit('logistic_regression', 10.0)


def write_csv(path, rows, source='borrador_ia'):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'department', 'source', 'intent', 'language'])
        for text, label in rows:
            writer.writerow([text, label, source, 'x', 'es'])
    return path


@pytest.mark.parametrize('fixture', ['nb', 'lr'])
def test_attribution_sums_to_model_score(fixture, request):
    model = request.getfixturevalue(fixture)
    tfidf, clf = model.named_steps['tfidf'], model.named_steps['classifier']
    X = tfidf.transform([t for t, _ in EVAL]).tocsr()
    scores = ae.class_scores(clf, X)
    if fixture == 'lr':
        np.testing.assert_allclose(scores, clf.decision_function(X))
    else:
        np.testing.assert_allclose(scores, clf.predict_joint_log_proba(X))
    for i in range(X.shape[0]):
        for c in range(len(clf.classes_)):
            total = sum(ae.term_contributions(clf, X[i], c).values()) + ae.class_bias(clf, c)
            assert total == pytest.approx(scores[i, c])


@pytest.mark.parametrize('fixture', ['nb', 'lr'])
def test_explanation_difference_matches_score_gap(fixture, request):
    model = request.getfixturevalue(fixture)
    tfidf, clf = model.named_steps['tfidf'], model.named_steps['classifier']
    names = tfidf.get_feature_names_out()
    X = tfidf.transform(['no se descargó la factura y la página falla']).tocsr()
    scores = ae.class_scores(clf, X)[0]
    classes = list(clf.classes_)
    a, b = classes.index('Soporte TI'), classes.index('Finanzas')
    explanation = ae.explain_difference(clf, names, X[0], a, b, top=50)
    assert explanation['score_difference'] == pytest.approx(scores[a] - scores[b])
    assert all(t['weight'] > 0 for t in explanation['top_terms_predicted'] + explanation['top_terms_expected'])


def test_global_terms_per_class(lr, nb):
    for model in (lr, nb):
        terms = ae.global_terms(model.named_steps['classifier'], model.named_steps['tfidf'].get_feature_names_out(), 5)
        assert set(terms) == set(LABELS) and all(len(v) == 5 for v in terms.values())
    assert any(t['term'].startswith('cupon') or t['term'].startswith('promocion')
               for t in ae.global_terms(lr.named_steps['classifier'],
                                        lr.named_steps['tfidf'].get_feature_names_out(), 8)['Marketing'])


def test_wilson_interval():
    assert ae.wilson(0, 0) == (0.0, 1.0)
    low, high = ae.wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3) and high == pytest.approx(0.7634, abs=1e-3)
    assert ae.wilson(10, 10)[1] == 1.0 and ae.wilson(0, 10)[0] == 0.0


def test_confusion_pairs_by_hand():
    actual = ['A', 'A', 'A', 'B', 'B', 'C']
    predicted = ['B', 'B', 'A', 'A', 'B', 'C']
    pairs = ae.confusion_pairs(actual, predicted)
    assert pairs == [{'actual': 'A', 'predicted': 'B', 'count': 2, 'rate_of_actual': 2 / 3},
                     {'actual': 'B', 'predicted': 'A', 'count': 1, 'rate_of_actual': 0.5}]


def test_mcnemar_exact():
    assert ae.mcnemar_exact(0, 0) == 1.0
    assert ae.mcnemar_exact(1, 5) == pytest.approx(0.21875)
    assert ae.compare_models(['a', 'b', 'c'], ['a', 'b', 'x'], ['a', 'x', 'c']) == {
        'corrected_by_main': 1, 'introduced_by_main': 1, 'both_wrong': 0, 'both_right': 1,
        'mcnemar_exact_p': 1.0, 'warning': 'Con pocos desacuerdos la prueba casi nunca alcanza significación.'}


def trace(tokens, kept=None, transformed=None):
    kept = tokens if kept is None else kept
    return {'tokens': tokens, 'tokens_kept': kept, 'tokens_transformed': kept if transformed is None else transformed}


def test_heuristic_rules_positive_and_negative():
    assert ae.is_short(trace(['hola', 'mundo'])) and not ae.is_short(trace(['a', 'b', 'c', 'd']))
    assert ae.has_negation(trace(['no', 'llega'])) and not ae.has_negation(trace(['llega']))
    assert ae.has_conjunction(trace(['factura', 'y', 'pedido'])) and not ae.has_conjunction(trace(['factura']))
    vocabulary = {'factura': 0, 'pago': 1}
    assert ae.oov_ratio(trace(['x'], transformed=['factura', 'zzz']), vocabulary) == 0.5
    assert ae.oov_ratio(trace(['x'], transformed=['factura', 'pago']), vocabulary) == 0.0
    assert ae.oov_ratio(trace([], transformed=[]), vocabulary) == 0.0
    top = {'Finanzas': {'factura'}, 'Logística': {'pedido'}}
    assert ae.classes_with_distinctive_terms({'factura', 'pedido'}, top) == ['Finanzas', 'Logística']
    assert ae.classes_with_distinctive_terms({'otra'}, top) == []


def tags_for(**overrides):
    base = dict(trace=trace(['una', 'frase', 'larga', 'con', 'varias', 'palabras'],
                            transformed=['frase', 'larga', 'palabras', 'varias']),
                vocabulary={'frase': 0, 'larga': 1, 'palabras': 2, 'varias': 3}, nnz=4,
                present_features={'frase'}, top_terms={'Finanzas': {'factura'}, 'Logística': {'pedido'}},
                expected='Finanzas', leading_term='frase', margin=0.5, low_margin=0.1)
    base.update(overrides)
    return ae.cause_tags(**base)


def test_cause_tags_combination():
    assert tags_for() == []
    assert tags_for(nnz=0) == ['sin_terminos_conocidos']
    assert 'margen_bajo' in tags_for(margin=0.05)
    assert 'termino_ambiguo' in tags_for(leading_term='factura')
    assert 'termino_ambiguo' not in tags_for(leading_term=None)
    multi = tags_for(trace=trace(['factura', 'y', 'pedido', 'x', 'y', 'z'], transformed=['factura', 'pedido']),
                     vocabulary={'factura': 0, 'pedido': 1}, present_features={'factura', 'pedido'})
    assert 'multi_intencion' in multi
    assert tags_for(trace=trace(['no', 'a', 'b', 'c'], transformed=['a']), vocabulary={'a': 0}).count('negacion') == 1


def test_rejects_reserved_test_partition(tmp_path):
    reserved = {'abc123'}
    with pytest.raises(ValueError, match='reservada'):
        ae.guard_dataset(tmp_path / 'test.csv', reserved)
    with pytest.raises(ValueError, match='reservada'):
        ae.guard_dataset(tmp_path / 'test_es.csv', reserved)
    renamed = write_csv(tmp_path / 'otro.csv', EVAL)
    ae.guard_dataset(renamed, reserved)  # nombre y huella distintos: permitido
    with pytest.raises(ValueError, match='reservada'):
        ae.guard_dataset(renamed, {ae.sha256(renamed)})


def test_reserved_hashes_read_from_manifests(tmp_path):
    folder = tmp_path / 'data/processed/model'
    folder.mkdir(parents=True)
    (folder / 'manifest.json').write_text(json.dumps({'outputs_sha256': {'train': 't', 'test': 'h-test'}}))
    assert ae.reserved_test_hashes(tmp_path) == {'h-test'}


def test_manifest_verification(tmp_path):
    data = write_csv(tmp_path / 'h.csv', EVAL, 'propio_humano')
    good, bad = tmp_path / 'good.json', tmp_path / 'bad.json'
    good.write_text(json.dumps({'dataset_sha256': ae.sha256(data)}))
    bad.write_text(json.dumps({'dataset_sha256': 'otra'}))
    ae.verify_manifest(data, good)
    with pytest.raises(ValueError, match='no coincide'):
        ae.verify_manifest(data, bad)


def test_run_outputs_and_determinism(tmp_path, nb, lr):
    model_a, model_b = tmp_path / 'a.joblib', tmp_path / 'b.joblib'
    joblib.dump(lr, model_a)
    joblib.dump(nb, model_b)
    data = write_csv(tmp_path / 'eval.csv', EVAL)
    datasets = {'simulado': data, 'humano': tmp_path / 'no_existe.csv'}
    first = ae.run(model_a, datasets, tmp_path / 'o1', compare_with=model_b)
    ae.run(model_a, datasets, tmp_path / 'o2', compare_with=model_b)
    assert (tmp_path / 'o1/error_analysis.json').read_bytes() == (tmp_path / 'o2/error_analysis.json').read_bytes()
    assert first['skipped_datasets'] == [{'name': 'humano', 'reason': 'archivo no encontrado'}]
    result = first['per_dataset']['simulado']
    assert result['summary']['records'] == 4 and result['summary']['errors'] == len(result['errors']) >= 1
    assert set(result['cause_tags']) == set(ae.TAGS) and 'comparison' in result
    assert 'sin_terminos_conocidos' in {t for e in result['errors'] for t in e['tags']}
    rows = list(csv.DictReader((tmp_path / 'o1/error_review.csv').open(encoding='utf-8')))
    assert len(rows) == result['summary']['errors'] and rows[0]['motivo_manual'] == ''


def test_redaction_leaves_no_text_in_outputs(tmp_path, lr):
    model = tmp_path / 'm.joblib'
    joblib.dump(lr, model)
    rows = [('mensaje privadísimo sobre mi factura de enero', 'Logística'), ('horario de atención', 'Soporte TI')]
    data = write_csv(tmp_path / 'humano.csv', rows, 'propio_humano')
    ae.run(model, {'humano': data}, tmp_path / 'out', redact_text=True)
    for path in (tmp_path / 'out').iterdir():
        content = path.read_text(encoding='utf-8')
        assert 'privadísimo' not in content and 'horario de atención' not in content, path.name
    assert json.loads((tmp_path / 'out/error_analysis.json').read_text(encoding='utf-8'))['redacted'] is True


def test_run_refuses_test_partition(tmp_path, lr):
    model = tmp_path / 'm.joblib'
    joblib.dump(lr, model)
    with pytest.raises(ValueError, match='reservada'):
        ae.run(model, {'test': write_csv(tmp_path / 'test.csv', EVAL)}, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
