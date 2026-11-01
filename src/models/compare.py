"""Compara modelos y variantes NLP sin acceder al conjunto test."""
from __future__ import annotations

import argparse
import json
import platform
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.models.train import DATA, LABELS, ROOT, digest, load_data
from src.nlp.preprocessing import SpanishTokenizer

SEED = 42
METHODS = ('stemming', 'lemmatization')
NGRAMS = (1, 2)
MODELS = ('naive_bayes', 'logistic_regression')
PARAMETERS = {'naive_bayes': (0.5, 1.0), 'logistic_regression': (0.1, 1.0)}
DIAGNOSTICS = [
    ('la aplicación no me funciona', 'Soporte TI'),
    ('buenas, tengo un problema y no se me descargó la factura', 'Finanzas'),
    ('tengo problemas con la aplicación y no me deja iniciar sesión', 'Soporte TI'),
    ('tengo un pedido para las 3 am', 'Logística'),
]


def classifier(name: str, value: float):
    if name == 'naive_bayes':
        return MultinomialNB(alpha=value)
    if name == 'logistic_regression':
        return LogisticRegression(C=value, max_iter=2000, random_state=SEED)
    raise ValueError(f'Modelo desconocido: {name}')


def vectorizer(method: str, ngrams: int) -> TfidfVectorizer:
    return TfidfVectorizer(tokenizer=SpanishTokenizer(method), token_pattern=None,
                           lowercase=False, ngram_range=(1, ngrams), min_df=1, max_df=1.0)


def pretokenized_vectorizer(ngrams: int) -> TfidfVectorizer:
    # La tokenización no aprende del corpus; TF-IDF se ajusta por separado en cada fold.
    return TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                           ngram_range=(1, ngrams), min_df=1, max_df=1.0)


def key(method: str, ngrams: int, model: str, value: float) -> tuple:
    return method, ngrams, model, value


def ranking(result: dict) -> tuple:
    # Mayor F1; en empate priorizar unigramas, NB, stemming y regularización mayor.
    return (-result['f1_macro_mean'], result['ngrams'], MODELS.index(result['model']),
            METHODS.index(result['method']), result['parameter_value'] if result['model'] == 'logistic_regression'
            else -result['parameter_value'])


def compare(train_texts: list[str], train_labels: list[str], n_splits: int = 3) -> list[dict]:
    labels = np.asarray(train_labels)
    if set(labels) != set(LABELS) or min(Counter(labels).values()) < n_splits:
        raise ValueError('Entrenamiento debe contener las seis clases con suficientes filas para CV.')
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(cv.split(train_texts, labels))
    results = {}
    for method in METHODS:
        for ngrams in NGRAMS:
            for model in MODELS:
                for value in PARAMETERS[model]:
                    results[key(method, ngrams, model, value)] = {
                        'method': method, 'ngrams': ngrams, 'model': model,
                        'parameter_name': 'alpha' if model == 'naive_bayes' else 'C',
                        'parameter_value': value, 'fold_f1_macro': [], 'errors': []}
        try:
            print(f'Procesando {method}...', flush=True)
            tokenizer = SpanishTokenizer(method)
            cache = {}
            transformed = []
            for text in train_texts:
                if text not in cache:
                    cache[text] = ' '.join(tokenizer(text))
                transformed.append(cache[text])
            transformed = np.asarray(transformed, dtype=object)
        except Exception as exc:
            for k, result in results.items():
                if k[0] == method:
                    result['errors'].append(f'Procesamiento: {type(exc).__name__}: {exc}')
            continue
        for ngrams in NGRAMS:
            for fold, (train_idx, val_idx) in enumerate(folds, 1):
                try:
                    tfidf = pretokenized_vectorizer(ngrams)
                    x_train = tfidf.fit_transform(transformed[train_idx])
                    x_val = tfidf.transform(transformed[val_idx])
                except Exception as exc:
                    for model in MODELS:
                        for value in PARAMETERS[model]:
                            results[key(method, ngrams, model, value)]['errors'].append(
                                f'Fold {fold}, TF-IDF: {type(exc).__name__}: {exc}')
                    continue
                for model in MODELS:
                    for value in PARAMETERS[model]:
                        result = results[key(method, ngrams, model, value)]
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter('error', ConvergenceWarning)
                                estimator = classifier(model, value)
                                estimator.fit(x_train, labels[train_idx])
                                predicted = estimator.predict(x_val)
                            result['fold_f1_macro'].append(float(f1_score(
                                labels[val_idx], predicted, labels=LABELS, average='macro', zero_division=0)))
                        except Exception as exc:
                            result['errors'].append(f'Fold {fold}: {type(exc).__name__}: {exc}')
                print(f'  {method}, ngramas 1-{ngrams}, fold {fold}/{n_splits}', flush=True)
    for result in results.values():
        valid = not result['errors'] and len(result['fold_f1_macro']) == n_splits
        result['status'] = 'ok' if valid else 'failed'
        result['f1_macro_mean'] = float(np.mean(result['fold_f1_macro'])) if valid else None
        result['f1_macro_std'] = float(np.std(result['fold_f1_macro'])) if valid else None
    return list(results.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA,
                        help='Directorio que contiene train.csv y validation.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/comparison')
    args = parser.parse_args()
    train_path, val_path = args.data_dir / 'train.csv', args.data_dir / 'validation.csv'
    train_texts, train_labels = load_data(train_path)
    results = compare(train_texts, train_labels)
    valid = sorted((r for r in results if r['status'] == 'ok'), key=ranking)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not valid:
        (args.output_dir / 'failures.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise SystemExit('No hubo configuraciones válidas; ver failures.json.')
    best = valid[0]
    print('Configuración seleccionada:', best, flush=True)
    model = Pipeline([('tfidf', vectorizer(best['method'], best['ngrams'])),
                      ('classifier', classifier(best['model'], best['parameter_value']))])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(train_texts, train_labels)
    val_texts, val_labels = load_data(val_path)
    predictions = model.predict(val_texts)
    diagnostics = [{'text': text, 'expected': expected, 'predicted': predicted}
                   for (text, expected), predicted in zip(DIAGNOSTICS, model.predict([t for t, _ in DIAGNOSTICS]))]
    errors = [{'text': text, 'actual': actual, 'predicted': predicted}
              for text, actual, predicted in zip(val_texts, val_labels, predictions) if actual != predicted]
    report = {
        'selection': 'Mayor media F1 macro CV; empate: unigramas, NB, stemming y mayor regularización.',
        'seed': SEED, 'folds': 3, 'test_used': False,
        'inputs_sha256': {'train': digest(train_path), 'validation': digest(val_path)},
        'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                     'nltk': nltk.__version__, 'spacy': spacy.__version__, 'joblib': joblib.__version__,
                     'spacy_model': spacy.util.get_package_version('es_core_news_sm')},
        'counts': {'train': len(train_texts), 'validation': len(val_texts)},
        'configurations': results, 'selected': best,
        'validation': {'f1_macro': float(f1_score(val_labels, predictions, labels=LABELS, average='macro', zero_division=0)),
                       'classification_report': classification_report(val_labels, predictions, labels=LABELS,
                                                                        output_dict=True, zero_division=0),
                       'confusion_matrix_labels': LABELS,
                       'confusion_matrix': confusion_matrix(val_labels, predictions, labels=LABELS).tolist(),
                       'errors': errors},
        'diagnostic_cases': diagnostics,
        'limitations': 'SID es sintético y train puede contener variantes similares entre folds; '
                       'validation tiene solo 4 ejemplos de Marketing. No se evaluó test ni datos humanos.',
    }
    joblib.dump(model, args.output_dir / 'pipeline.joblib')
    report['model_sha256'] = digest(args.output_dir / 'pipeline.joblib')
    (args.output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"F1 macro en validación: {report['validation']['f1_macro']:.4f}; informe: {args.output_dir / 'report.json'}")


if __name__ == '__main__':
    main()
