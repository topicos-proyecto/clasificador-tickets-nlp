"""Análisis automático de errores de un pipeline entrenado; diagnóstico, no ajuste.

No lee la partición `test`. Las etiquetas de causa son heurísticas que una persona debe confirmar.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import classification_report, f1_score

from src.models.compare import DIAGNOSTICS
from src.models.evaluate_human import sha256
from src.models.train import LABELS, ROOT, load_data
from src.nlp.preprocessing import NEGATIONS

# Umbrales de las reglas heurísticas (constantes con nombre; no se calibran con test).
SHORT_MAX_TOKENS = 3
OOV_RATIO_MIN = 0.5
TOP_TERMS_PER_CLASS = 20
LOW_SUPPORT = 30
MARGIN_THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
CONJUNCTIONS = frozenset({'y', 'pero', 'además', 'ademas', 'también', 'tambien'})
TAGS = ('sin_terminos_conocidos', 'vocabulario_desconocido', 'texto_corto', 'multi_intencion',
        'negacion', 'termino_ambiguo', 'margen_bajo')
RESERVED_NAME = re.compile(r'^test(_.*)?\.csv$', re.IGNORECASE)
LIMITATIONS = ('Diagnóstico exploratorio. Las etiquetas de causa son heurísticas y requieren revisión '
               'humana; las métricas de conjuntos pequeños tienen alta incertidumbre y no demuestran '
               'generalización a clientes reales.')


# ---------------------------------------------------------------- protecciones y carga

def reserved_test_hashes(root: Path = ROOT) -> set[str]:
    """Huellas de `test` registradas en los manifiestos de data/processed."""
    found: set[str] = set()

    def walk(node, parent_key=None):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == 'test' and isinstance(v, str):
                    found.add(v)
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, parent_key)

    for manifest in sorted((root / 'data/processed').glob('*/manifest.json')):
        try:
            walk(json.loads(manifest.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            continue
    return found


def guard_dataset(path: Path, reserved: set[str]) -> None:
    if RESERVED_NAME.match(path.name) or (path.is_file() and sha256(path) in reserved):
        raise ValueError(f'{path}: la partición test está reservada y no puede analizarse.')


def verify_manifest(dataset: Path, manifest: Path) -> None:
    expected = json.loads(manifest.read_text(encoding='utf-8')).get('dataset_sha256')
    if sha256(dataset) != expected:
        raise ValueError(f'La huella de {dataset} no coincide con el manifiesto congelado.')


def load_dataset(path: Path) -> tuple[list[str], list[str], list[str]]:
    texts, labels = load_data(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        sources = sorted({r.get('source', '') for r in reader if r.get('source')})
    return texts, labels, sources


# ---------------------------------------------------------------- estadística

def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(only_a_wrong: int, only_b_wrong: int) -> float:
    """p-valor exacto bilateral sobre los desacuerdos entre dos modelos."""
    total = only_a_wrong + only_b_wrong
    return 1.0 if total == 0 else float(binomtest(min(only_a_wrong, only_b_wrong), total, 0.5).pvalue)


def confusion_pairs(actual: list[str], predicted: list[str]) -> list[dict]:
    support = Counter(actual)
    pairs = Counter((a, p) for a, p in zip(actual, predicted) if a != p)
    rows = [{'actual': a, 'predicted': p, 'count': n, 'rate_of_actual': n / support[a]}
            for (a, p), n in pairs.items()]
    return sorted(rows, key=lambda r: (-r['count'], r['actual'], r['predicted']))


def per_class_summary(actual: list[str], predicted: list[str]) -> dict:
    report = classification_report(actual, predicted, labels=LABELS, output_dict=True, zero_division=0)
    output = {}
    for label in LABELS:
        support = int(report[label]['support'])
        hits = sum(a == p == label for a, p in zip(actual, predicted))
        entry = {k: float(report[label][k]) for k in ('precision', 'recall', 'f1-score')}
        entry['support'] = support
        if support < LOW_SUPPORT:
            low, high = wilson(hits, support)
            entry['recall_wilson_95'] = [low, high]
            entry['warning'] = f'Soporte bajo ({support}): interpretar con cautela.'
        output[label] = entry
    return output


def margin_summary(margins: np.ndarray, correct: np.ndarray) -> dict:
    def stats(values: np.ndarray) -> dict:
        if not len(values):
            return {'n': 0, 'mean': None, 'median': None}
        return {'n': int(len(values)), 'mean': float(values.mean()), 'median': float(np.median(values))}

    errors = margins[~correct]
    return {'correct': stats(margins[correct]), 'errors': stats(errors),
            'errors_below': {str(t): int((errors < t).sum()) for t in MARGIN_THRESHOLDS}}


def calibration_table(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.digitize(confidence, edges[1:-1]), bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        accuracy, mean_conf = float(correct[mask].mean()), float(confidence[mask].mean())
        ece += mask.sum() / len(confidence) * abs(accuracy - mean_conf)
        rows.append({'range': [float(edges[b]), float(edges[b + 1])], 'n': int(mask.sum()),
                     'accuracy': accuracy, 'mean_confidence': mean_conf})
    return {'bins': rows, 'ece': float(ece),
            'warning': 'Con pocos ejemplos por intervalo la tabla es muy ruidosa.'}


# ---------------------------------------------------------------- atribución de términos

def class_scores(classifier, X) -> np.ndarray:
    """Puntaje lineal por clase (LR: decision_function; NB: log-verosimilitud conjunta)."""
    if hasattr(classifier, 'coef_'):
        return np.asarray(classifier.decision_function(X))
    if hasattr(classifier, 'feature_log_prob_'):
        return np.asarray(X @ classifier.feature_log_prob_.T) + classifier.class_log_prior_
    raise TypeError(f'Modelo no soportado: {type(classifier).__name__}')


def class_bias(classifier, class_index: int) -> float:
    if hasattr(classifier, 'coef_'):
        return float(classifier.intercept_[class_index])
    return float(classifier.class_log_prior_[class_index])


def class_weights(classifier) -> np.ndarray:
    return classifier.coef_ if hasattr(classifier, 'coef_') else classifier.feature_log_prob_


def term_contributions(classifier, row, class_index: int) -> dict[int, float]:
    """Aporte de cada término presente en `row` al puntaje de una clase."""
    weights = class_weights(classifier)[class_index]
    return {int(j): float(weights[j] * v) for j, v in zip(row.indices, row.data)}


def explain_difference(classifier, names, row, predicted_index: int, expected_index: int, top: int = 5) -> dict:
    """Qué términos empujaron hacia la clase predicha frente a la esperada (suma exacta del puntaje)."""
    pred = term_contributions(classifier, row, predicted_index)
    exp = term_contributions(classifier, row, expected_index)
    diff = {j: pred[j] - exp[j] for j in pred}
    ordered = sorted(diff.items(), key=lambda item: (-item[1], names[item[0]]))
    return {
        'top_terms_predicted': [{'term': str(names[j]), 'weight': w} for j, w in ordered[:top] if w > 0],
        'top_terms_expected': [{'term': str(names[j]), 'weight': -w}
                               for j, w in sorted(diff.items(), key=lambda i: (i[1], names[i[0]]))[:top] if w < 0],
        'bias_difference': class_bias(classifier, predicted_index) - class_bias(classifier, expected_index),
        'score_difference': sum(diff.values()) + class_bias(classifier, predicted_index)
        - class_bias(classifier, expected_index),
    }


def global_terms(classifier, names, top: int = TOP_TERMS_PER_CLASS) -> dict[str, list[dict]]:
    """Términos más distintivos por clase (coeficientes en LR; log-razón frente al resto en NB)."""
    weights = np.asarray(class_weights(classifier))
    output = {}
    for i, label in enumerate(classifier.classes_):
        if hasattr(classifier, 'coef_'):
            score = weights[i]
        else:
            score = weights[i] - np.delete(weights, i, axis=0).mean(axis=0)
        order = sorted(range(len(score)), key=lambda j: (-score[j], names[j]))[:top]
        output[str(label)] = [{'term': str(names[j]), 'weight': float(score[j])} for j in order]
    return output


# ---------------------------------------------------------------- etiquetas heurísticas

def oov_ratio(trace: dict, vocabulary: dict) -> float:
    """Fracción de términos ausentes del vocabulario de entrenamiento (palabras nuevas o errores de escritura)."""
    tokens = trace['tokens_transformed']
    return sum(t not in vocabulary for t in tokens) / len(tokens) if tokens else 0.0


def is_short(trace: dict) -> bool:
    return len(trace['tokens_kept']) <= SHORT_MAX_TOKENS


def has_negation(trace: dict) -> bool:
    return any(t in NEGATIONS for t in trace['tokens'])


def has_conjunction(trace: dict) -> bool:
    return any(t in CONJUNCTIONS for t in trace['tokens'])


def classes_with_distinctive_terms(features: set[str], top_terms: dict[str, set[str]]) -> list[str]:
    return sorted(label for label, terms in top_terms.items() if features & terms)


def cause_tags(*, trace: dict, vocabulary: dict, nnz: int, present_features: set[str],
               top_terms: dict[str, set[str]], expected: str, leading_term: str | None,
               margin: float, low_margin: float) -> list[str]:
    tags = set()
    if nnz == 0:
        tags.add('sin_terminos_conocidos')
    if oov_ratio(trace, vocabulary) >= OOV_RATIO_MIN:
        tags.add('vocabulario_desconocido')
    if is_short(trace):
        tags.add('texto_corto')
    if has_conjunction(trace) and len(classes_with_distinctive_terms(present_features, top_terms)) >= 2:
        tags.add('multi_intencion')
    if has_negation(trace):
        tags.add('negacion')
    if leading_term is not None and leading_term in top_terms.get(expected, set()):
        tags.add('termino_ambiguo')
    if margin <= low_margin:
        tags.add('margen_bajo')
    return [t for t in TAGS if t in tags]


# ---------------------------------------------------------------- análisis de un conjunto

def analyze_dataset(model, texts: list[str], labels: list[str], *, top: int = 5,
                    top_terms: dict[str, set[str]] | None = None) -> dict:
    tfidf, classifier = model.named_steps['tfidf'], model.named_steps['classifier']
    names = tfidf.get_feature_names_out()
    classes = [str(c) for c in classifier.classes_]
    X = tfidf.transform(texts).tocsr()
    probabilities = classifier.predict_proba(X)
    predicted = [str(p) for p in model.predict(texts)]
    order = np.sort(probabilities, axis=1)
    margins = order[:, -1] - order[:, -2]
    correct = np.asarray([a == p for a, p in zip(labels, predicted)])
    low_margin = float(np.percentile(margins, 25)) if len(margins) else 0.0
    if top_terms is None:
        top_terms = {k: {t['term'] for t in v} for k, v in global_terms(classifier, names).items()}
    errors = []
    for i in np.flatnonzero(~correct):
        row = X[i]
        trace = tfidf.tokenizer.trace(texts[i])
        explanation = explain_difference(classifier, names, row, classes.index(predicted[i]),
                                         classes.index(labels[i]), top)
        leading = explanation['top_terms_predicted'][0]['term'] if explanation['top_terms_predicted'] else None
        tags = cause_tags(trace=trace, vocabulary=tfidf.vocabulary_, nnz=int(row.nnz),
                          present_features={str(names[j]) for j in row.indices}, top_terms=top_terms,
                          expected=labels[i], leading_term=leading, margin=float(margins[i]),
                          low_margin=low_margin)
        errors.append({'line': int(i) + 2, 'text': texts[i], 'expected': labels[i], 'predicted': predicted[i],
                       'margin': float(margins[i]), 'tags': tags, 'trace': trace, **explanation})
    tag_counts = Counter(t for e in errors for t in e['tags'])
    return {
        'summary': {'records': len(texts), 'errors': len(errors), 'accuracy': float(correct.mean()) if len(texts) else 0.0,
                    'accuracy_wilson_95': list(wilson(int(correct.sum()), len(texts))),
                    'f1_macro': float(f1_score(labels, predicted, labels=LABELS, average='macro', zero_division=0))},
        'per_class': per_class_summary(labels, predicted),
        'confusion_pairs': confusion_pairs(labels, predicted),
        'margins': margin_summary(margins, correct),
        'calibration': calibration_table(probabilities.max(axis=1), correct),
        'cause_tags': {t: tag_counts.get(t, 0) for t in TAGS},
        'low_margin_threshold': low_margin,
        'errors': errors,
        'predictions': predicted,
    }


def compare_models(labels: list[str], main: list[str], other: list[str]) -> dict:
    main_ok = [a == p for a, p in zip(labels, main)]
    other_ok = [a == p for a, p in zip(labels, other)]
    corrected = sum(m and not o for m, o in zip(main_ok, other_ok))
    introduced = sum(o and not m for m, o in zip(main_ok, other_ok))
    return {'corrected_by_main': corrected, 'introduced_by_main': introduced,
            'both_wrong': sum(not m and not o for m, o in zip(main_ok, other_ok)),
            'both_right': sum(m and o for m, o in zip(main_ok, other_ok)),
            'mcnemar_exact_p': mcnemar_exact(introduced, corrected),
            'warning': 'Con pocos desacuerdos la prueba casi nunca alcanza significación.'}


def diagnostic_cases(model, top: int = 5) -> list[dict]:
    tfidf, classifier = model.named_steps['tfidf'], model.named_steps['classifier']
    names, classes = tfidf.get_feature_names_out(), [str(c) for c in classifier.classes_]
    X = tfidf.transform([t for t, _ in DIAGNOSTICS]).tocsr()
    output = []
    for i, ((text, expected), predicted) in enumerate(zip(DIAGNOSTICS, model.predict([t for t, _ in DIAGNOSTICS]))):
        entry = {'text': text, 'expected': expected, 'predicted': str(predicted)}
        entry.update(explain_difference(classifier, names, X[i], classes.index(str(predicted)),
                                        classes.index(expected), top) if predicted != expected else {})
        output.append(entry)
    return output


# ---------------------------------------------------------------- salidas

def redact(analysis: dict) -> dict:
    """Quita textos y trazas de los errores (los agregados se conservan)."""
    clean = json.loads(json.dumps(analysis))
    for dataset in clean['per_dataset'].values():
        dataset['errors'] = [{k: v for k, v in e.items()
                              if k in ('line', 'expected', 'predicted', 'margin', 'tags')}
                             for e in dataset['errors']]
    clean['diagnostic_cases'] = []
    clean['redacted'] = True
    return clean


def render_markdown(analysis: dict) -> str:
    lines = ['# Análisis automático de errores', '',
             f"Modelo: `{analysis['model_sha256']}`", '', f"> {analysis['limitations']}", '']
    for name, data in analysis['per_dataset'].items():
        s = data['summary']
        lines += [f'## Conjunto `{name}`', '',
                  f"- Registros: {s['records']}; errores: {s['errors']}; exactitud: {s['accuracy']:.3f} "
                  f"(IC 95 %: {s['accuracy_wilson_95'][0]:.3f}–{s['accuracy_wilson_95'][1]:.3f}); F1 macro: {s['f1_macro']:.3f}",
                  f"- ECE: {data['calibration']['ece']:.3f}", '',
                  '| Clase | Precisión | Recall | F1 | Soporte |', '|---|---:|---:|---:|---:|']
        for label, c in data['per_class'].items():
            note = ' ⚠' if 'warning' in c else ''
            lines.append(f"| {label}{note} | {c['precision']:.2f} | {c['recall']:.2f} | {c['f1-score']:.2f} | {c['support']} |")
        lines += ['', '**Confusiones más frecuentes**', '']
        lines += [f"- {p['actual']} → {p['predicted']}: {p['count']} ({p['rate_of_actual']:.0%} de la clase)"
                  for p in data['confusion_pairs'][:8]] or ['- Sin errores.']
        lines += ['', '**Causas probables (heurísticas, no excluyentes)**', '']
        lines += [f'- `{t}`: {n}' for t, n in data['cause_tags'].items() if n] or ['- Ninguna.']
        if 'comparison' in data:
            c = data['comparison']
            lines += ['', f"**Frente al modelo comparado:** corrige {c['corrected_by_main']}, introduce "
                          f"{c['introduced_by_main']}, comunes erróneos {c['both_wrong']} (McNemar exacto p = {c['mcnemar_exact_p']:.3f})."]
        lines.append('')
    return '\n'.join(lines) + '\n'


REVIEW_COLUMNS = ['dataset', 'linea', 'text', 'expected', 'predicted', 'margin', 'tags',
                  'top_terms_predicted', 'motivo_manual', 'etiqueta_discutible', 'comentario']


def write_review_csv(path: Path, analysis: dict, redacted: bool) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for name, data in analysis['per_dataset'].items():
            for e in data['errors']:
                writer.writerow({
                    'dataset': name, 'linea': e['line'], 'text': '' if redacted else e['text'],
                    'expected': e['expected'], 'predicted': e['predicted'], 'margin': f"{e['margin']:.4f}",
                    'tags': ';'.join(e['tags']),
                    'top_terms_predicted': '' if redacted else ';'.join(t['term'] for t in e['top_terms_predicted']),
                    'motivo_manual': '', 'etiqueta_discutible': '', 'comentario': ''})


def run(model_path: Path, datasets: dict[str, Path], output_dir: Path, *, compare_with: Path | None = None,
        manifests: dict[str, Path] | None = None, redact_text: bool = False, top: int = 5) -> dict:
    reserved, manifests = reserved_test_hashes(), manifests or {}
    for path in datasets.values():
        guard_dataset(path, reserved)
    model = joblib.load(model_path)
    other = joblib.load(compare_with) if compare_with else None
    tfidf, classifier = model.named_steps['tfidf'], model.named_steps['classifier']
    names = tfidf.get_feature_names_out()
    globals_ = global_terms(classifier, names)
    top_sets = {k: {t['term'] for t in v} for k, v in globals_.items()}
    analysis = {'model_sha256': sha256(model_path),
                'compared_model_sha256': sha256(compare_with) if compare_with else None,
                'datasets': {}, 'skipped_datasets': [], 'per_dataset': {}, 'global_terms': globals_,
                'diagnostic_cases': diagnostic_cases(model, top), 'limitations': LIMITATIONS}
    for name, path in datasets.items():
        if not path.is_file():
            analysis['skipped_datasets'].append({'name': name, 'reason': 'archivo no encontrado'})
            continue
        if name in manifests:
            verify_manifest(path, manifests[name])
        texts, labels, sources = load_dataset(path)
        data = analyze_dataset(model, texts, labels, top=top, top_terms=top_sets)
        predictions = data.pop('predictions')
        if other is not None:
            data['comparison'] = compare_models(labels, predictions, [str(p) for p in other.predict(texts)])
        analysis['datasets'][name] = {'sha256': sha256(path), 'records': len(texts), 'sources': sources,
                                      'manifest_verified': name in manifests}
        analysis['per_dataset'][name] = data
    final = redact(analysis) if redact_text else analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'error_analysis.json').write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'error_analysis.md').write_text(render_markdown(final), encoding='utf-8')
    write_review_csv(output_dir / 'error_review.csv', final, redact_text)
    return final


def parse_pairs(values: list[str] | None, what: str) -> dict[str, Path]:
    pairs = {}
    for value in values or []:
        name, sep, path = value.partition('=')
        if not sep or not name or not path:
            raise SystemExit(f'{what}: usa el formato nombre=ruta (recibido {value!r}).')
        pairs[name] = Path(path)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'artifacts/comparison_customers/pipeline.joblib')
    parser.add_argument('--dataset', action='append', metavar='NOMBRE=RUTA', help='Conjunto a analizar (repetible)')
    parser.add_argument('--manifest', action='append', metavar='NOMBRE=RUTA',
                        help='Manifiesto congelado cuyo dataset_sha256 se verifica para ese conjunto')
    parser.add_argument('--compare-with', type=Path, help='Otro pipeline para comparar errores (McNemar)')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--redact-text', action='store_true', help='Omite textos y términos de los errores')
    parser.add_argument('--top-terms', type=int, default=5)
    args = parser.parse_args()
    datasets = parse_pairs(args.dataset, '--dataset')
    if not datasets:
        raise SystemExit('Indica al menos un --dataset nombre=ruta.')
    output_dir = args.output_dir or args.model.parent / 'error_analysis'
    try:
        result = run(args.model, datasets, output_dir, compare_with=args.compare_with,
                     manifests=parse_pairs(args.manifest, '--manifest'), redact_text=args.redact_text,
                     top=args.top_terms)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for name, data in result['per_dataset'].items():
        s = data['summary']
        print(f"{name}: {s['errors']}/{s['records']} errores; F1 macro {s['f1_macro']:.4f}")
    for item in result['skipped_datasets']:
        print(f"Omitido {item['name']}: {item['reason']}")
    print(f'Resultados: {output_dir}')


if __name__ == '__main__':
    main()
