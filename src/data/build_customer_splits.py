"""Crea una versión de particiones agregando los 30 borradores aprobados de clientes."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.data.build_splits import COLUMNS, ROOT, THRESHOLD, grams, read_csv
from src.data.validate_custom import validate

BASE = ROOT / 'data/processed/model'
DRAFT = ROOT / 'data/templates/internal_support_draft.csv'
OUT = ROOT / 'data/processed/model_customers'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if validate(DRAFT, compare_sid=False):
        raise SystemExit('Los borradores tienen errores de formato.')
    drafts = read_csv(DRAFT, COLUMNS[:-1])
    if (len(drafts) != 30 or Counter(r['department'] for r in drafts) != {'Marketing': 15, 'Soporte TI': 15}
        or any(r['source'] != 'borrador_ia' for r in drafts)):
        raise SystemExit('El archivo no coincide con los 30 borradores aprobados.')
    names = ('train', 'validation', 'validation_real', 'test')
    # Solo se usa el test para bloquear solapamientos, sin mostrar sus textos.
    splits = {name: read_csv(BASE / f'{name}.csv', COLUMNS) for name in names}
    index = defaultdict(set)
    reference = []
    for name, rows in splits.items():
        for row in rows:
            g = grams(row['text'])
            idx = len(reference)
            reference.append(g)
            for token in g:
                index[token].add(idx)
    for line, row in enumerate(drafts, 2):
        g = grams(row['text'])
        overlaps = Counter()
        for token in g:
            overlaps.update(index.get(token, ()))
        if any(overlap / (len(g) + len(reference[i]) - overlap) >= THRESHOLD
               for i, overlap in overlaps.items()):
            raise SystemExit(f'Línea {line}: posible similitud entre datos; revisar antes de generar.')
        idx = len(reference)
        reference.append(g)
        for token in g:
            index[token].add(idx)
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f'{OUT} ya contiene archivos; no se sobrescribió.')
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        rows = splits[name] + ([dict(r, original_split='customer_draft_train') for r in drafts]
                               if name == 'train' else [])
        with (OUT / f'{name}.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {'base_sha256': {n: digest(BASE / f'{n}.csv') for n in names},
                'draft_sha256': digest(DRAFT), 'draft_count': len(drafts),
                'outputs_sha256': {n: digest(OUT / f'{n}.csv') for n in names},
                'counts': {n: len(splits[n]) + (len(drafts) if n == 'train' else 0) for n in names},
                'note': 'Borradores de IA aprobados por usuario; test conservado; similitud aproximada.'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest['counts'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
