"""Evalúa el modelo fijado con el conjunto humano congelado, sin leer test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data.validate_custom import COLUMNS
from src.models.train import LABELS, ROOT

DATASET = ROOT / 'data/raw/custom/human_evaluation_frozen.csv'
MODEL = ROOT / 'artifacts/comparison/pipeline.joblib'
FREEZE = ROOT / 'data/raw/custom/human_evaluation_manifest.json'
OUTPUT = ROOT / 'artifacts/human_evaluation/report.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(dataset: Path, model_path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for key, path in (('dataset_sha256', dataset), ('model_sha256', model_path)):
        if sha256(path) != manifest[key]:
            raise ValueError(f'La huella de {path} no coincide con el manifiesto congelado.')
    with dataset.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError('Cabecera inesperada en evaluación humana.')
        rows = list(reader)
    if not rows or any(r['source'] != 'propio_humano' or r['department'] not in LABELS
                       or not r['text'].strip() for r in rows):
        raise ValueError('El conjunto contiene datos vacíos, fuentes o etiquetas inválidas.')
    model = joblib.load(model_path)
    tfidf = model.named_steps['tfidf']
    estimator = model.named_steps['classifier']
    if (type(estimator).__name__ != 'MultinomialNB' or estimator.alpha != 0.5
        or tfidf.ngram_range != (1, 1) or tfidf.tokenizer.method != 'lemmatization'):
        raise ValueError('El modelo cargado no coincide con la configuración seleccionada.')
    labels = [r['department'] for r in rows]
    predictions = model.predict([r['text'] for r in rows])
    errors = [{'linea': i, 'text': r['text'], 'actual': r['department'], 'predicted': p}
              for i, (r, p) in enumerate(zip(rows, predictions), 2) if r['department'] != p]
    return {
        'dataset_sha256': manifest['dataset_sha256'], 'model_sha256': manifest['model_sha256'],
        'model': manifest['model'], 'records': len(rows),
        'f1_macro': float(f1_score(labels, predictions, labels=LABELS, average='macro', zero_division=0)),
        'classification_report': classification_report(labels, predictions, labels=LABELS,
                                                       output_dict=True, zero_division=0),
        'confusion_matrix_labels': LABELS,
        'confusion_matrix': confusion_matrix(labels, predictions, labels=LABELS).tolist(),
        'errors': errors,
        'limitations': 'Evaluación exploratoria con pocos mensajes humanos y solicitudes internas; '
                       'no se utilizó test ni se ajustó el modelo con estos mensajes.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = evaluate(DATASET, MODEL, FREEZE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Evaluación: {report['records']} mensajes; F1 macro: {report['f1_macro']:.4f}")
    print(f'Informe local: {args.output}')


if __name__ == '__main__':
    main()
