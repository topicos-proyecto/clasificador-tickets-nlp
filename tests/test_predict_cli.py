import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.models import predict_cli as cli
from src.models.train import build_pipeline


@pytest.fixture
def saved_models(tmp_path):
    texts = ['horario atención', 'consulta general', 'factura cobrada', 'cobro duplicado',
             'envío pedido', 'entrega paquete', 'cupón descuento', 'promoción oferta',
             'contraseña acceso', 'error aplicación', 'cotizar producto', 'precio compra']
    labels = [label for label in ('Atención al Cliente', 'Finanzas', 'Logística',
                                 'Marketing', 'Soporte TI', 'Ventas') for _ in range(2)]
    models = []
    for name, classifier in [('nb', None), ('lr', LogisticRegression(random_state=42)),
                             ('knn', KNeighborsClassifier(n_neighbors=1, metric='cosine',
                                                         algorithm='brute'))]:
        model = build_pipeline()
        if classifier is not None:
            model.set_params(classifier=classifier)
        model.fit(texts, labels)
        path = tmp_path / f'{name}.joblib'
        joblib.dump(model, path)
        models.append((path, model))
    return models


def enter_messages(monkeypatch, *messages):
    answers = iter(messages)
    monkeypatch.setattr('builtins.input', lambda _: next(answers))


@pytest.mark.parametrize('model_index', [0, 1, 2], ids=['nb', 'lr', 'knn'])
def test_probabilities_match_serialized_pipeline_classes(saved_models, model_index):
    path, _ = saved_models[model_index]
    model = cli.load_model(path)
    message = 'No puedo consultar mi factura cobrada'
    before = dict(model.named_steps['tfidf'].vocabulary_)
    result = cli.predict_message(model, message)
    expected = model.predict_proba([message])[0]
    assert result['label'] == model.predict([message])[0] == 'Finanzas'
    assert result['probabilities'] == pytest.approx(dict(zip(model.classes_, expected)))
    assert sum(result['probabilities'].values()) == pytest.approx(1)
    assert model.named_steps['tfidf'].vocabulary_ == before


def test_cli_compares_three_models_with_probabilities_and_traces(saved_models, monkeypatch, capsys):
    enter_messages(monkeypatch, 'hola', 'No puedo consultar mi factura cobrada', 'salir')
    arguments = [argument for path, _ in saved_models for argument in ('--model', str(path))]
    assert cli.main(arguments) == 0
    output = capsys.readouterr().out
    assert 'describe tu solicitud con más detalle' in output
    for index, (path, model) in enumerate(saved_models, 1):
        assert f'Modelo {index}: {type(model.named_steps["classifier"]).__name__} | {path}' in output
    for step in ('Normalización:', 'Tokens:', 'Stop words retiradas:', 'Tokens conservados:',
                 'Stemming:', 'TF-IDF:'):
        assert output.count(step) == 3
    for label in saved_models[0][1].classes_:
        assert output.count(label) >= 3
    assert output.count('Probabilidades por área:') == 3
    assert output.count('%') == 21
    assert 'Comparación para el mismo mensaje:' in output
    assert 'Los modelos coinciden en el área: Finanzas.' in output
    assert 'estimaciones no calibradas' in output


@pytest.mark.parametrize('method', ['stemming', 'lemmatization'])
def test_trace_uses_each_models_tokenizer(method):
    model = build_pipeline(method=method, ngrams=2)
    model.fit(['facturas cobradas', 'cupones usados'], ['Finanzas', 'Marketing'])
    message = '¡No pagué las facturas! https://ejemplo.com 😀'
    result = cli.predict_message(model, message)
    trace = result['preprocessing']
    assert trace == model.named_steps['tfidf'].tokenizer.trace(message)
    assert trace['normalized'] == 'no pagué las facturas'
    assert 'no' in trace['tokens_kept']
    assert 'las' in trace['stopwords_removed']
    assert result['recognized_features'] > 0


def test_unknown_vocabulary_does_not_produce_misleading_probabilities(saved_models, monkeypatch, capsys):
    path, model = saved_models[0]
    result = cli.predict_message(model, 'zzxyzzxy')
    assert result['label'] is None
    assert result['probabilities'] == {}
    enter_messages(monkeypatch, 'zzxyzzxy', 'salir')
    assert cli.main(['--model', str(path)]) == 0
    output = capsys.readouterr().out
    assert 'No se reconocieron términos del vocabulario' in output
    assert 'Probabilidades por área:' not in output


def test_default_model_still_works(saved_models, monkeypatch, capsys):
    monkeypatch.setattr(cli, 'MODEL', saved_models[0][0])
    enter_messages(monkeypatch, 'factura cobrada', 'salir')
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert 'Área sugerida: Finanzas' in output
    assert 'Comparación para el mismo mensaje:' not in output


def test_missing_comparison_model_returns_error_before_prompt(saved_models, tmp_path, capsys):
    missing = tmp_path / 'ausente.joblib'
    assert cli.main(['--model', str(saved_models[0][0]), '--model', str(missing)]) == 1
    captured = capsys.readouterr()
    assert str(missing) in captured.err
    assert 'No se encontró el modelo' in captured.err
    assert 'Clasificador de mensajes' not in captured.out


@pytest.mark.parametrize('contents', [None, {'classifier': 'no es un pipeline'}])
def test_incompatible_artifact_returns_readable_error(contents, tmp_path, capsys):
    path = tmp_path / 'incompatible.joblib'
    joblib.dump(contents, path)
    assert cli.main(['--model', str(path)]) == 1
    assert 'Se requiere un pipeline entrenado' in capsys.readouterr().err


def test_corrupt_artifact_returns_readable_error(tmp_path, capsys):
    path = tmp_path / 'corrupto.joblib'
    path.write_bytes(b'')
    assert cli.main(['--model', str(path)]) == 1
    assert 'No se pudo cargar' in capsys.readouterr().err


@pytest.mark.parametrize('exception', [EOFError, KeyboardInterrupt])
def test_cli_exits_cleanly_when_input_is_interrupted(saved_models, monkeypatch, capsys, exception):
    def interrupted_input(_):
        raise exception
    monkeypatch.setattr('builtins.input', interrupted_input)
    assert cli.main(['--model', str(saved_models[0][0])]) == 0
    assert 'Hasta luego.' in capsys.readouterr().out


@pytest.mark.parametrize('label, diagnostic', [('Ventas', 'sugieren áreas diferentes'),
                                               (None, 'Hay modelos sin clasificación')])
def test_comparison_reports_disagreement_and_unclassified_models(label, diagnostic, capsys):
    cli.print_comparison([{'label': 'Finanzas', 'probabilities': {'Finanzas': 0.6}},
                          {'label': label, 'probabilities': {label: 0.7} if label else {}}])
    assert diagnostic in capsys.readouterr().out


def test_prediction_failure_does_not_prevent_other_models_from_running(saved_models, monkeypatch, capsys):
    original_predict = cli.predict_message

    def failing_nb(model, message):
        if type(model.named_steps['classifier']).__name__ == 'MultinomialNB':
            raise RuntimeError('modelo no disponible')
        return original_predict(model, message)

    monkeypatch.setattr(cli, 'predict_message', failing_nb)
    enter_messages(monkeypatch, 'factura cobrada', 'salir')
    arguments = [argument for path, _ in saved_models[:2] for argument in ('--model', str(path))]
    assert cli.main(arguments) == 0
    output = capsys.readouterr().out
    assert 'modelo no disponible' in output
    assert 'Área sugerida: Finanzas' in output
    assert 'Hay modelos sin clasificación' in output
