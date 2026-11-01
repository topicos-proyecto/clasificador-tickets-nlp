"""Evalúa ejemplos simulados por IA; no representa una prueba con clientes reales."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data.validate_custom import COLUMNS, validate
from src.models.train import LABELS, ROOT, digest

DATA = ROOT / 'data/templates/customer_simulated_eval.csv'
MODEL = ROOT / 'artifacts/comparison_customers/pipeline.joblib'
REPORT = ROOT / 'artifacts/comparison_customers/simulated_report.json'


def main() -> None:
    if validate(DATA, compare_sid=False):
        raise SystemExit('El conjunto simulado no es válido.')
    with DATA.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise SystemExit('Cabecera inesperada.')
        rows = list(reader)
    if not rows or any(r['source'] != 'borrador_ia' or r['department'] not in LABELS for r in rows):
        raise SystemExit('El conjunto debe contener borradores de IA con etiquetas válidas.')
    model = joblib.load(MODEL)
    expected = [r['department'] for r in rows]
    predicted = model.predict([r['text'] for r in rows])
    output = {'dataset_sha256': digest(DATA), 'model_sha256': digest(MODEL),
              'source': 'borrador_ia', 'records': len(rows),
              'correct': sum(a == b for a, b in zip(expected, predicted)),
              'f1_macro': float(f1_score(expected, predicted, labels=LABELS, average='macro', zero_division=0)),
              'classification_report': classification_report(expected, predicted, labels=LABELS,
                                                               output_dict=True, zero_division=0),
              'confusion_matrix_labels': LABELS,
              'confusion_matrix': confusion_matrix(expected, predicted, labels=LABELS).tolist(),
              'errors': [{'text': r['text'], 'expected': r['department'], 'predicted': p}
                         for r, p in zip(rows, predicted) if r['department'] != p],
              'limitations': 'Datos simulados por IA, etiquetas pendientes de revisión; no son prueba de generalización a clientes reales.'}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Aciertos: {output['correct']}/{len(rows)}; F1 macro: {output['f1_macro']:.4f}")
    print(f'Informe: {REPORT}')


if __name__ == '__main__':
    main()
