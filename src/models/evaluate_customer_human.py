"""Evalúa clientes humanos con el modelo orientado a clientes fijado previamente."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.evaluate_human import evaluate
from src.models.train import ROOT

DATA = ROOT / 'data/raw/custom/human_evaluation_clients_frozen.csv'
MODEL = ROOT / 'artifacts/comparison_customers/pipeline.joblib'
MANIFEST = ROOT / 'data/raw/custom/human_evaluation_clients_manifest.json'
OUTPUT = ROOT / 'artifacts/comparison_customers/human_clients_report.json'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = evaluate(DATA, MODEL, MANIFEST)
    report['limitations'] = ('Evaluación exploratoria: 30 mensajes de clientes redactados por humanos, '
                             'pocos ejemplos por área; similitudes solo detectadas de forma aproximada. '
                             'No se utilizó test ni se ajustó el modelo con estos mensajes.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Aciertos: {report['records'] - len(report['errors'])}/{report['records']}; F1 macro: {report['f1_macro']:.4f}")
    print(f'Informe: {args.output}')


if __name__ == '__main__':
    main()
