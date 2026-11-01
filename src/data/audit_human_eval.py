"""Revisa datos humanos reservados sin entrenar ni evaluar el modelo."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.data.build_splits import grams, normalized
from src.data.validate_custom import COLUMNS, DEPARTMENTS, validate

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'data/raw/custom/human_evaluation.csv'
REFERENCE = ROOT / 'data/processed/model'
THRESHOLD = 0.85


def audit(path: Path, references: list[Path]) -> dict:
    """Compara con entrenamiento y validación; nunca incluye textos de referencia en la salida."""
    if validate(path, compare_sid=False):
        raise ValueError('El CSV contiene errores de formato o duplicados.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError('Cabecera inválida.')
        rows = list(reader)
    if not rows:
        raise ValueError('El archivo humano está vacío; añade mensajes originales antes de auditar.')
    if any(row['source'].strip() != 'propio_humano' for row in rows):
        raise ValueError('La evaluación independiente solo admite source=propio_humano.')
    index = defaultdict(set)
    reference_grams: list[frozenset[str]] = []
    reference_texts: list[str] = []
    reference_names: list[str] = []
    for reference in references:
        with reference.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'text' not in reader.fieldnames:
                raise ValueError(f'Falta text en {reference}.')
            for row in reader:
                text = normalized(row['text'])
                features = grams(row['text'])
                idx = len(reference_grams)
                reference_grams.append(features)
                reference_texts.append(text)
                reference_names.append(reference.name)
                for gram in features:
                    index[gram].add(idx)
    matches = []
    for line, row in enumerate(rows, start=2):
        features = grams(row['text'])
        if not features or features == frozenset({''}):
            raise ValueError(f'Línea {line}: texto sin caracteres útiles.')
        overlap = Counter()
        for gram in features:
            overlap.update(index.get(gram, ()))
        related = set()
        for idx, common in overlap.items():
            union = len(features) + len(reference_grams[idx]) - common
            if union and common / union >= THRESHOLD:
                kind = 'exacta_normalizada' if normalized(row['text']) == reference_texts[idx] else 'similar'
                related.add((reference_names[idx], kind))
        if related:
            matches.append({'linea': line, 'coincidencias': [
                {'particion': name, 'tipo': kind} for name, kind in sorted(related)]})
    counts = Counter(row['department'].strip() for row in rows)
    return {'archivo_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'registros': len(rows), 'por_area': {dept: counts[dept] for dept in sorted(DEPARTMENTS)},
            'areas_sin_ejemplos': sorted(DEPARTMENTS - counts.keys()),
            'lineas_con_similitudes': matches,
            'referencias_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in references},
            'limitacion': 'La similitud por trigramas no detecta todas las paráfrasis; requiere revisión humana.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='?', type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        report = audit(args.csv, [REFERENCE / 'train.csv', REFERENCE / 'validation.csv',
                                  REFERENCE / 'validation_real.csv'])
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['lineas_con_similitudes']:
        print('Revisa las coincidencias antes de congelar el archivo.', file=sys.stderr)
        return 1
    if report['areas_sin_ejemplos']:
        print('Faltan áreas; la evaluación aún no representa todas las clases.', file=sys.stderr)
        return 1
    print('Auditoría técnica aprobada; aún se requiere revisión humana de etiquetas y similitudes.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
