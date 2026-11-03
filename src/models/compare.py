"""Compara modelos y variantes NLP sin acceder al conjunto test."""
from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import nltk
import numpy as np
import sklearn
import spacy
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.models.train import DATA, LABELS, ROOT, digest, load_data
from src.nlp.preprocessing import SpanishTokenizer

SEED = 42
METHODS = ('stemming', 'lemmatization')
NGRAMS = (1, 2)
MODELS = ('naive_bayes', 'logistic_regression', 'knn')
MAX_ITER = 5000
# Hiperparámetro principal de cada modelo; el resto de campos de GRID son secundarios.
PRIMARY = {'naive_bayes': 'alpha', 'logistic_regression': 'C', 'knn': 'n_neighbors'}
GRID = {
    'naive_bayes': [{'alpha': alpha} for alpha in (0.01, 0.03, 0.1, 0.3, 0.5, 1.0)],
    'logistic_regression': [{'C': c, 'class_weight': weight}
                            for c in (0.1, 1.0, 10.0, 30.0, 100.0) for weight in (None, 'balanced')],
    # KNN con distancia coseno sobre TF-IDF; k mayor = frontera más suave (más regularizado).
    'knn': [{'n_neighbors': k, 'weights': weights}
            for k in (1, 3, 5, 7, 11, 15, 25) for weights in ('uniform', 'distance')],
}
TFIDF_GRID = {'min_df': (1, 2, 3), 'max_df': (1.0, 0.9)}
DEFAULT_TFIDF = {'min_df': 1, 'max_df': 1.0}
SELECTION_RULE = ('Mayor media de F1 macro en CV. Se aplica la regla de una desviación estándar: entre las '
                  'configuraciones cuya media no queda por debajo de (mejor media - desviación de la mejor) se '
                  'elige la más simple: unigramas, luego el modelo más simple (Naive Bayes, después Regresión '
                  'Logística y por último KNN, que guarda todo el entrenamiento y es el más costoso en inferencia), '
                  'mayor regularización (alpha mayor, C menor, k mayor), sin pesos de clase ni ponderación por '
                  'distancia y stemming; la última desambiguación es la mayor media.')
DIAGNOSTICS = [
    ('la aplicación no me funciona', 'Soporte TI'),
    ('buenas, tengo un problema y no se me descargó la factura', 'Finanzas'),
    ('tengo problemas con la aplicación y no me deja iniciar sesión', 'Soporte TI'),
    ('tengo un pedido para las 3 am', 'Logística'),
]


def classifier(name: str, value: float, class_weight: str | None = None, weights: str | None = None):
    if name == 'naive_bayes':
        return MultinomialNB(alpha=value)
    if name == 'logistic_regression':
        return LogisticRegression(C=value, class_weight=class_weight, max_iter=MAX_ITER, random_state=SEED)
    if name == 'knn':
        return KNeighborsClassifier(n_neighbors=int(value), weights=weights or 'uniform', metric='cosine',
                                    algorithm='brute', n_jobs=1)
    raise ValueError(f'Modelo desconocido: {name}')


def vectorizer(method: str, ngrams: int, min_df=1, max_df=1.0) -> TfidfVectorizer:
    return TfidfVectorizer(tokenizer=SpanishTokenizer(method), token_pattern=None,
                           lowercase=False, ngram_range=(1, ngrams), min_df=min_df, max_df=max_df)


def pretokenized_vectorizer(ngrams: int, min_df=1, max_df=1.0) -> TfidfVectorizer:
    # La tokenización no aprende del corpus; TF-IDF se ajusta por separado en cada fold.
    return TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                           ngram_range=(1, ngrams), min_df=min_df, max_df=max_df)


def configurations(grid: dict | None = None) -> list[dict]:
    """Expande la rejilla en configuraciones únicas, en orden determinista."""
    grid = GRID if grid is None else grid
    expanded = []
    for method in METHODS:
        for ngrams in NGRAMS:
            for model in MODELS:
                for params in grid.get(model, []):
                    value = params[PRIMARY[model]]
                    expanded.append({'method': method, 'ngrams': ngrams, 'model': model,
                                     'parameter_name': PRIMARY[model], 'parameter_value': value,
                                     'class_weight': params.get('class_weight'),
                                     'weights': params.get('weights')})
    return expanded


def key(config: dict) -> tuple:
    return (config['method'], config['ngrams'], config['model'],
            config['parameter_value'], config.get('class_weight'), config.get('weights'))


def ranking(result: dict) -> tuple:
    """Orden de simplicidad: menor es mejor. Se usa solo entre candidatos equivalentes."""
    regularization = (result['parameter_value'] if result['model'] == 'logistic_regression'
                      else -result['parameter_value'])
    weighted = result.get('class_weight') is not None or result.get('weights') == 'distance'
    return (result['ngrams'], MODELS.index(result['model']), regularization,
            weighted, METHODS.index(result['method']), -result['f1_macro_mean'])


def valid_results(results: list[dict]) -> list[dict]:
    return [r for r in results if r.get('status') == 'ok']


def best_by_mean(results: list[dict]) -> dict | None:
    """Mejor media de CV sin regla de simplicidad (los empates se resuelven con `ranking`)."""
    valid = valid_results(results)
    return min(valid, key=lambda r: (-r['f1_macro_mean'], ranking(r))) if valid else None


def select(results: list[dict]) -> dict | None:
    """Regla de una desviación estándar: la configuración más simple cercana a la mejor."""
    best = best_by_mean(results)
    if best is None:
        return None
    threshold = best['f1_macro_mean'] - best['f1_macro_std']
    candidates = [r for r in valid_results(results) if r['f1_macro_mean'] >= threshold - 1e-12]
    return min(candidates, key=ranking)


def describe(result: dict) -> dict:
    return {k: result.get(k) for k in ('model', 'method', 'ngrams', 'parameter_name', 'parameter_value',
                                       'class_weight', 'weights', 'f1_macro_mean', 'f1_macro_std')}


def paired_comparison(results: list[dict]) -> dict | None:
    """Diferencia por partición entre la mejor LR y el mejor NB (LR - NB), con las mismas particiones."""
    nb = best_by_mean([r for r in results if r['model'] == 'naive_bayes'])
    lr = best_by_mean([r for r in results if r['model'] == 'logistic_regression'])
    if nb is None or lr is None:
        return None
    diffs = [float(a - b) for a, b in zip(lr['fold_f1_macro'], nb['fold_f1_macro'])]
    return {'logistic_regression': describe(lr), 'naive_bayes': describe(nb),
            'fold_differences_lr_minus_nb': diffs, 'mean_difference': float(np.mean(diffs)),
            'std_difference': float(np.std(diffs)),
            'lr_exceeds_nb_by_more_than_one_std': bool(np.mean(diffs) > lr['f1_macro_std'])}


def pairwise_comparisons(results: list[dict]) -> list[dict]:
    """Diferencias pareadas (a - b) por partición entre los mejores de cada par de modelos."""
    best = {m: best_by_mean([r for r in results if r['model'] == m]) for m in MODELS}
    pairs = []
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            if best[a] is None or best[b] is None:
                continue
            diffs = [float(x - y) for x, y in zip(best[a]['fold_f1_macro'], best[b]['fold_f1_macro'])]
            pairs.append({'a': a, 'b': b, 'fold_differences_a_minus_b': diffs,
                          'mean_difference': float(np.mean(diffs)), 'std_difference': float(np.std(diffs))})
    return pairs


def variants(results: list[dict]) -> list[dict]:
    """Mejor media de CV por modelo, método y n-gramas (efecto de stemming/lematización y bigramas)."""
    output = []
    for model in MODELS:
        for method in METHODS:
            for ngrams in NGRAMS:
                best = best_by_mean([r for r in results if (r['model'], r['method'], r['ngrams']) == (model, method, ngrams)])
                if best:
                    output.append(describe(best))
    return output


def evaluation(y_true, y_pred) -> dict:
    """Métricas agregadas, por clase y matriz de confusión (filas = real, columnas = predicho)."""
    return {'records': len(y_true),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'f1_macro': float(f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, labels=LABELS, average='weighted', zero_division=0)),
            'classification_report': classification_report(y_true, y_pred, labels=LABELS, output_dict=True,
                                                           zero_division=0),
            'confusion_matrix_labels': LABELS,
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}


def out_of_fold(config: dict, transformed, labels, folds) -> np.ndarray:
    """Predicciones fuera de partición con los mismos folds del CV (TF-IDF ajustado en cada fold)."""
    predicted = np.empty(len(labels), dtype=object)
    for train_idx, val_idx in folds:
        tfidf = pretokenized_vectorizer(config['ngrams'], config.get('min_df', 1), config.get('max_df', 1.0))
        x_train = tfidf.fit_transform(transformed[train_idx])
        estimator = make_estimator(config)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            estimator.fit(x_train, labels[train_idx])
        predicted[val_idx] = estimator.predict(tfidf.transform(transformed[val_idx]))
    return predicted


def build_model(config: dict) -> Pipeline:
    return Pipeline([('tfidf', vectorizer(config['method'], config['ngrams'], config.get('min_df', 1),
                                          config.get('max_df', 1.0))),
                     ('classifier', make_estimator(config))])


def measure_cost(model: Pipeline, texts: list[str]) -> dict:
    """Latencia de predicción (pipeline completo y solo clasificador) y tamaño serializado."""
    sample = (list(texts) * (1000 // max(len(texts), 1) + 1))[:1000]
    started = time.perf_counter()
    features = model.named_steps['tfidf'].transform(sample)
    tokenize_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    model.named_steps['classifier'].predict(features)
    classifier_ms = (time.perf_counter() - started) * 1000
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return {'predict_ms_per_1000_pipeline': round(tokenize_ms + classifier_ms, 1),
            'predict_ms_per_1000_classifier': round(classifier_ms, 1),
            'artifact_bytes': buffer.getbuffer().nbytes}


def finalists(results: list[dict], selected: dict, train_texts, train_labels, eval_sets: dict,
              n_splits: int = 3) -> tuple[dict, dict]:
    """Mejor configuración de cada modelo con métricas completas.

    Para el modelo de la familia elegida se usa la configuración seleccionada (la que se despliega);
    para los demás, la de mayor media de CV. Devuelve (métricas, pipelines entrenados).
    """
    labels = np.asarray(train_labels)
    try:
        folds = make_folds(train_texts, labels, n_splits)
    except ValueError:  # conjuntos diminutos (pruebas): sin predicciones fuera de partición
        folds = None
    cache, output, models = {}, {}, {}
    for name in MODELS:
        best = best_by_mean([r for r in results if r['model'] == name])
        if best is None:
            continue
        config = selected if name == selected['model'] else {**best, **DEFAULT_TFIDF}
        print(f'Finalista {name}: {describe(config)}', flush=True)
        oof = None
        if folds is not None:
            if config['method'] not in cache:
                cache[config['method']] = pretokenize(train_texts, config['method'])
            oof = out_of_fold(config, cache[config['method']], labels, folds)
        model = build_model(config)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(train_texts, train_labels)
        fit_seconds = time.perf_counter() - started
        entry = {'config': {**describe(config), 'min_df': config.get('min_df', 1), 'max_df': config.get('max_df', 1.0)},
                 'is_selected': name == selected['model'],
                 'best_by_mean': describe(best),
                 'cv': {'f1_macro_mean': config['f1_macro_mean'], 'f1_macro_std': config['f1_macro_std'],
                        'fold_f1_macro': config['fold_f1_macro'],
                        'out_of_fold': evaluation(labels, oof) if oof is not None else None},
                 'vocabulary_size': len(model.named_steps['tfidf'].vocabulary_)}
        for set_name, (texts, set_labels) in eval_sets.items():
            entry[set_name] = evaluation(set_labels, model.predict(texts))
        first_eval = next(iter(eval_sets.values()))[0]
        entry['cost'] = {'fit_seconds': round(fit_seconds, 2), **measure_cost(model, first_eval)}
        output[name], models[name] = entry, model
    return output, models


def make_folds(texts, labels, n_splits: int = 3, groups=None):
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(cv.split(texts, labels))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(cv.split(texts, labels, groups))


def pretokenize(texts, method: str):
    tokenizer = SpanishTokenizer(method)
    cache = {}
    for text in texts:
        if text not in cache:
            cache[text] = ' '.join(tokenizer(text))
    return np.asarray([cache[text] for text in texts], dtype=object)


def make_estimator(config: dict):
    return classifier(config['model'], config['parameter_value'], config.get('class_weight'), config.get('weights'))


def score_fold(config: dict, x_train, y_train, x_val, y_val) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        estimator = make_estimator(config)
        estimator.fit(x_train, y_train)
        predicted = estimator.predict(x_val)
    return float(f1_score(y_val, predicted, labels=LABELS, average='macro', zero_division=0))


def finish(result: dict, n_splits: int) -> None:
    valid = not result['errors'] and len(result['fold_f1_macro']) == n_splits
    result['status'] = 'ok' if valid else 'failed'
    result['f1_macro_mean'] = float(np.mean(result['fold_f1_macro'])) if valid else None
    result['f1_macro_std'] = float(np.std(result['fold_f1_macro'])) if valid else None


def compare(train_texts: list[str], train_labels: list[str], n_splits: int = 3,
            grid: dict | None = None, timings: dict | None = None, groups=None) -> list[dict]:
    labels = np.asarray(train_labels)
    if set(labels) != set(LABELS) or min(Counter(labels).values()) < n_splits:
        raise ValueError('Entrenamiento debe contener las seis clases con suficientes filas para CV.')
    folds = make_folds(train_texts, labels, n_splits, groups)
    results = {}
    for config in configurations(grid):
        results[key(config)] = {**config, 'fold_f1_macro': [], 'errors': []}
    for method in METHODS:
        started = time.perf_counter()
        try:
            print(f'Procesando {method}...', flush=True)
            transformed = pretokenize(train_texts, method)
        except Exception as exc:
            for k, result in results.items():
                if k[0] == method:
                    result['errors'].append(f'Procesamiento: {type(exc).__name__}: {exc}')
            continue
        for ngrams in NGRAMS:
            for fold, (train_idx, val_idx) in enumerate(folds, 1):
                members = [r for k, r in results.items() if k[0] == method and k[1] == ngrams]
                try:
                    tfidf = pretokenized_vectorizer(ngrams)
                    x_train = tfidf.fit_transform(transformed[train_idx])
                    x_val = tfidf.transform(transformed[val_idx])
                except Exception as exc:
                    for result in members:
                        result['errors'].append(f'Fold {fold}, TF-IDF: {type(exc).__name__}: {exc}')
                    continue
                for result in members:
                    try:
                        result['fold_f1_macro'].append(score_fold(
                            result, x_train, labels[train_idx], x_val, labels[val_idx]))
                    except Exception as exc:
                        result['errors'].append(f'Fold {fold}: {type(exc).__name__}: {exc}')
                print(f'  {method}, ngramas 1-{ngrams}, fold {fold}/{n_splits}', flush=True)
        if timings is not None:
            timings[method] = round(time.perf_counter() - started, 1)
    for result in results.values():
        finish(result, n_splits)
    return list(results.values())


def tfidf_sweep(train_texts, train_labels, base: dict, n_splits: int = 3) -> list[dict]:
    """Barrido pequeño de min_df/max_df solo para la configuración elegida."""
    labels = np.asarray(train_labels)
    folds = make_folds(train_texts, labels, n_splits)
    transformed = pretokenize(train_texts, base['method'])
    rows = []
    for min_df in TFIDF_GRID['min_df']:
        for max_df in TFIDF_GRID['max_df']:
            row = {'min_df': min_df, 'max_df': max_df, 'fold_f1_macro': [], 'errors': []}
            try:
                full = pretokenized_vectorizer(base['ngrams'], min_df, max_df).fit(transformed)
                row['vocabulary_size'] = len(full.vocabulary_)
            except Exception as exc:
                row['errors'].append(f'Vocabulario: {type(exc).__name__}: {exc}')
            for fold, (train_idx, val_idx) in enumerate(folds, 1):
                if row['errors']:
                    break
                try:
                    tfidf = pretokenized_vectorizer(base['ngrams'], min_df, max_df)
                    x_train = tfidf.fit_transform(transformed[train_idx])
                    x_val = tfidf.transform(transformed[val_idx])
                    row['fold_f1_macro'].append(score_fold(base, x_train, labels[train_idx],
                                                           x_val, labels[val_idx]))
                except Exception as exc:
                    row['errors'].append(f'Fold {fold}: {type(exc).__name__}: {exc}')
            finish(row, n_splits)
            rows.append(row)
            print(f'  TF-IDF min_df={min_df}, max_df={max_df}: {row["status"]}', flush=True)
    return rows


def choose_tfidf(sweep: list[dict]) -> dict:
    """Se conservan los valores por defecto salvo mejora mayor que una desviación estándar."""
    default = next((r for r in sweep if (r['min_df'], r['max_df']) == (1, 1.0) and r['status'] == 'ok'), None)
    candidates = valid_results(sweep)
    if default is None or not candidates:
        return dict(DEFAULT_TFIDF)
    best = max(candidates, key=lambda r: (r['f1_macro_mean'], -r['min_df'], r['max_df']))
    if best['f1_macro_mean'] - default['f1_macro_mean'] > default['f1_macro_std']:
        return {'min_df': best['min_df'], 'max_df': best['max_df']}
    return dict(DEFAULT_TFIDF)


def read_intents(path: Path) -> list[str]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if 'intent' not in (reader.fieldnames or ()):
            raise ValueError(f'{path}: falta la columna intent para la sensibilidad agrupada')
        return [r['intent'] or r['text'] for r in reader]


def grouped_sensitivity(train_texts, train_labels, intents, results: list[dict]) -> dict:
    """CV agrupado por intención SID: mucho más estricto que el CV principal (intenciones nunca vistas)."""
    finalists = [r for r in (best_by_mean([x for x in results if x['model'] == m]) for m in MODELS) if r]
    labels = np.asarray(train_labels)
    folds = make_folds(train_texts, labels, 3, intents)
    output = []
    for base in finalists:
        transformed = pretokenize(train_texts, base['method'])
        scores, errors = [], []
        for fold, (train_idx, val_idx) in enumerate(folds, 1):
            try:
                tfidf = pretokenized_vectorizer(base['ngrams'])
                x_train = tfidf.fit_transform(transformed[train_idx])
                x_val = tfidf.transform(transformed[val_idx])
                scores.append(score_fold(base, x_train, labels[train_idx], x_val, labels[val_idx]))
            except Exception as exc:
                errors.append(f'Fold {fold}: {type(exc).__name__}: {exc}')
        output.append({**describe(base), 'grouped_fold_f1_macro': scores, 'errors': errors,
                       'grouped_f1_macro_mean': float(np.mean(scores)) if scores and not errors else None})
    return {'grouping': 'intención SID (intent)', 'n_groups': len(set(intents)), 'finalists': output,
            'note': 'Análisis de sensibilidad: no interviene en la selección.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA,
                        help='Directorio que contiene train.csv y validation.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/comparison')
    parser.add_argument('--overwrite', action='store_true',
                        help='Permite sobrescribir un pipeline.joblib existente en --output-dir')
    parser.add_argument('--tfidf-grid', action='store_true',
                        help='Barrido de min_df/max_df sobre la configuración elegida')
    parser.add_argument('--simulated', type=Path, default=ROOT / 'data/templates/customer_simulated_eval.csv',
                        help='Conjunto simulado con IA evaluado de forma exploratoria (no selecciona)')
    parser.add_argument('--grouped-cv', action='store_true',
                        help='Añade un análisis de sensibilidad con CV agrupado por intención SID')
    args = parser.parse_args()
    if (args.output_dir / 'pipeline.joblib').exists() and not args.overwrite:
        raise SystemExit(f'{args.output_dir}/pipeline.joblib ya existe; usa otro --output-dir o --overwrite.')
    train_path, val_path = args.data_dir / 'train.csv', args.data_dir / 'validation.csv'
    train_texts, train_labels = load_data(train_path)
    timings: dict = {}
    results = compare(train_texts, train_labels, timings=timings)
    valid = valid_results(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not valid:
        (args.output_dir / 'failures.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise SystemExit('No hubo configuraciones válidas; ver failures.json.')
    best = select(results)
    strict = best_by_mean(results)
    print('Configuración seleccionada:', best, flush=True)
    tfidf_params, sweep = dict(DEFAULT_TFIDF), None
    if args.tfidf_grid:
        sweep = tfidf_sweep(train_texts, train_labels, best)
        tfidf_params = choose_tfidf(sweep)
    best = {**best, **tfidf_params}
    val_texts, val_labels = load_data(val_path)
    eval_sets = {'validation': (val_texts, val_labels)}
    extra_inputs = {}
    real_path = args.data_dir / 'validation_real.csv'
    if real_path.is_file():
        eval_sets['validation_real'] = load_data(real_path)
        extra_inputs['validation_real'] = digest(real_path)
    if args.simulated and args.simulated.is_file():
        eval_sets['simulated'] = load_data(args.simulated)
        extra_inputs['simulated'] = digest(args.simulated)
    finalist_metrics, finalist_models = finalists(results, best, train_texts, train_labels, eval_sets)
    model = finalist_models[best['model']]
    predictions = model.predict(val_texts)
    diagnostics = [{'text': text, 'expected': expected, 'predicted': predicted}
                   for (text, expected), predicted in zip(DIAGNOSTICS, model.predict([t for t, _ in DIAGNOSTICS]))]
    errors = [{'text': text, 'actual': actual, 'predicted': predicted}
              for text, actual, predicted in zip(val_texts, val_labels, predictions) if actual != predicted]
    other = [r for r in valid if r['model'] != best['model']]
    report = {
        'selection': SELECTION_RULE,
        'seed': SEED, 'folds': 3, 'test_used': False,
        'inputs_sha256': {'train': digest(train_path), 'validation': digest(val_path), **extra_inputs},
        'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                     'nltk': nltk.__version__, 'spacy': spacy.__version__, 'joblib': joblib.__version__,
                     'spacy_model': spacy.util.get_package_version('es_core_news_sm')},
        'counts': {'train': len(train_texts), **{k: len(v[0]) for k, v in eval_sets.items()}},
        'grid': GRID, 'n_configurations': len(results), 'elapsed_seconds': timings,
        'configurations': results, 'selected': best,
        'best_by_mean': describe(strict),
        'runner_up_other_model': describe(best_by_mean(other)) if other else None,
        'paired_comparison': paired_comparison(results),
        'pairwise_comparisons': pairwise_comparisons(results),
        'variants': variants(results),
        'threshold': strict['f1_macro_mean'] - strict['f1_macro_std'],
        'finalists': finalist_metrics,
        'tfidf_grid': sweep, 'vocabulary_size': len(model.named_steps['tfidf'].vocabulary_),
        'validation': {'f1_macro': float(f1_score(val_labels, predictions, labels=LABELS, average='macro', zero_division=0)),
                       'accuracy': float(accuracy_score(val_labels, predictions)),
                       'classification_report': classification_report(val_labels, predictions, labels=LABELS,
                                                                        output_dict=True, zero_division=0),
                       'confusion_matrix_labels': LABELS,
                       'confusion_matrix': confusion_matrix(val_labels, predictions, labels=LABELS).tolist(),
                       'errors': errors},
        'diagnostic_cases': diagnostics,
        'limitations': 'SID es sintético y train puede contener variantes similares entre folds; '
                       'validation tiene solo 4 ejemplos de Marketing. «simulated» son 24 mensajes redactados con IA '
                       '(exploratorio, no interviene en la selección). No se evaluó test ni datos humanos.',
    }
    if args.grouped_cv:
        try:
            report['grouped_sensitivity'] = grouped_sensitivity(
                train_texts, train_labels, read_intents(train_path), results)
        except Exception as exc:  # análisis opcional: no debe impedir guardar el modelo
            report['grouped_sensitivity'] = {'error': f'{type(exc).__name__}: {exc}'}
    joblib.dump(model, args.output_dir / 'pipeline.joblib')
    report['model_sha256'] = digest(args.output_dir / 'pipeline.joblib')
    (args.output_dir / 'finalists').mkdir(exist_ok=True)
    for name, finalist in finalist_models.items():
        path = args.output_dir / 'finalists' / f'{name}.joblib'
        joblib.dump(finalist, path)
        report['finalists'][name]['model_sha256'] = digest(path)
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"F1 macro en validación: {report['validation']['f1_macro']:.4f}; informe: {args.output_dir / 'report.json'}")


if __name__ == '__main__':
    main()
