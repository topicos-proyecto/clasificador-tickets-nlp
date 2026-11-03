"""Evaluación única y final de los tres finalistas en el conjunto `test` reservado.

Se ejecuta solo después de fijar la selección con validación cruzada. El resultado no cambia la
selección: se informa tal cual aunque otro modelo obtenga mejores cifras en `test`.
El script se niega a ejecutarse si ya existe `test_report.json` (no se «mira» test varias veces).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.models.compare import evaluation
from src.models.train import ROOT, digest, load_data

DEFAULT_DIR = ROOT / 'artifacts/comparison_customers_v3'


def run(artifacts: Path, test_path: Path) -> dict:
    report_path = artifacts / 'report.json'
    output = artifacts / 'test_report.json'
    if output.exists():
        raise SystemExit(f'{output} ya existe: test se evalúa una sola vez.')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('test_used') is not False:
        raise SystemExit('El informe de comparación no declara test_used = false.')
    if digest(artifacts / 'pipeline.joblib') != report['model_sha256']:
        raise SystemExit('pipeline.joblib no coincide con report.json.')
    texts, labels = load_data(test_path)
    results = {}
    for name, finalist in report['finalists'].items():
        path = artifacts / 'finalists' / f'{name}.joblib'
        if digest(path) != finalist['model_sha256']:
            raise SystemExit(f'{path} no coincide con la huella registrada en report.json.')
        model = joblib.load(path)
        results[name] = {'config': finalist['config'], 'is_selected': finalist['is_selected'],
                         **evaluation(labels, model.predict(texts))}
    payload = {'report_sha256': digest(report_path), 'model_sha256': report['model_sha256'],
               'test_sha256': digest(test_path), 'records': len(texts), 'selected_model': report['selected']['model'],
               'finalists': results,
               'note': 'Evaluación única posterior a la selección; no se usó para elegir ni ajustar modelos.'}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_DIR)
    parser.add_argument('--test', type=Path, default=ROOT / 'data/processed/model_customers/test.csv')
    args = parser.parse_args()
    payload = run(args.artifacts, args.test)
    for name, item in payload['finalists'].items():
        print(f"{name:20s} F1 macro {item['f1_macro']:.4f}  exactitud {item['accuracy']:.4f}"
              f"{'  (seleccionado)' if item['is_selected'] else ''}")


if __name__ == '__main__':
    main()
