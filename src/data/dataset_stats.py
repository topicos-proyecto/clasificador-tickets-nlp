"""Estadísticas agregadas del dataset para el panel de métricas (sin textos de mensajes).

De `test` solo se lee la columna `department` para contar filas por clase: sus textos no se
procesan ni se miden, de modo que la partición reservada no influye en ninguna decisión.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from src.models.train import LABELS, ROOT, digest

SPLITS = ('train', 'validation', 'validation_real', 'test')
TEXT_SPLITS = ('train', 'validation', 'validation_real')
# Intervalos de palabras por mensaje: 1-2, 3-4, ..., 13-14 y 15 o más.
BINS = [(low, low + 1) for low in range(1, 15, 2)] + [(15, None)]


def read_rows(path: Path, columns: tuple[str, ...]) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path}: faltan columnas {sorted(missing)}')
        return [{c: row[c] for c in columns} for row in reader]


def describe_lengths(values: list[int]) -> dict:
    if not values:
        return {'count': 0}
    array = np.asarray(values)
    return {'count': int(array.size), 'mean': round(float(array.mean()), 2),
            'median': float(np.median(array)), 'min': int(array.min()), 'max': int(array.max()),
            'p25': float(np.percentile(array, 25)), 'p75': float(np.percentile(array, 75))}


def histogram(values: list[int]) -> list[dict]:
    output = []
    for low, high in BINS:
        count = sum(1 for v in values if v >= low and (high is None or v <= high))
        output.append({'bin': f'{low}+' if high is None else f'{low}-{high}', 'count': count})
    return output


def split_stats(rows: list[dict], with_text: bool) -> dict:
    counts = Counter(r['department'] for r in rows)
    result = {'records': len(rows),
              'class_counts': {label: counts.get(label, 0) for label in LABELS},
              'class_share': {label: round(counts.get(label, 0) / len(rows), 4) if rows else 0 for label in LABELS}}
    if with_text:
        words = [len(r['text'].split()) for r in rows]
        chars = [len(r['text']) for r in rows]
        result['words'] = describe_lengths(words)
        result['chars'] = describe_lengths(chars)
        result['words_histogram'] = histogram(words)
        result['words_by_class'] = {label: describe_lengths([len(r['text'].split()) for r in rows
                                                             if r['department'] == label]) for label in LABELS}
    return result


def build(data_dir: Path, model_path: Path | None = None) -> dict:
    output = {'labels': LABELS, 'splits': {}, 'inputs_sha256': {},
              'note': 'Solo agregados. De test se cuentan etiquetas; sus textos no se procesan.'}
    for split in SPLITS:
        path = data_dir / f'{split}.csv'
        if not path.is_file():
            continue
        with_text = split in TEXT_SPLITS
        rows = read_rows(path, ('text', 'department') if with_text else ('department',))
        output['splits'][split] = split_stats(rows, with_text)
        output['inputs_sha256'][split] = digest(path)
    if model_path and model_path.is_file() and 'train' in output['splits']:
        model = joblib.load(model_path)
        tfidf = model.named_steps['tfidf']
        texts = [r['text'] for r in read_rows(data_dir / 'train.csv', ('text',))]
        tokens = [len(tfidf.tokenizer(t)) for t in texts]
        output['preprocessing'] = {'method': tfidf.tokenizer.method, 'ngrams': tfidf.ngram_range[1],
                                   'vocabulary_size': len(tfidf.vocabulary_),
                                   'tokens_per_message_train': describe_lengths(tokens),
                                   'model_sha256': digest(model_path)}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/model_customers')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/comparison_customers_v3')
    args = parser.parse_args()
    stats = build(args.data_dir, args.output_dir / 'pipeline.joblib')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'dataset_stats.json'
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print({k: v['records'] for k, v in stats['splits'].items()}, '→', path)


if __name__ == '__main__':
    main()
