"""Prepara particiones reproducibles y aparta posibles duplicados entre ellas.

Requiere los CSV producidos por prepare_sid.py. No modifica fuentes ni ajusta modelos.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .validate_custom import validate
except ImportError:  # Ejecutado directamente como script.
    from validate_custom import validate

ROOT = Path(__file__).resolve().parents[2]
SID = ROOT / 'data/processed/sid'
OUT = ROOT / 'data/processed/model'
COLUMNS = ('text', 'department', 'source', 'intent', 'language', 'original_split')
# La prioridad protege test y evita que una fila similar termine en dos particiones.
PRIORITY = ('test', 'validation_real', 'validation', 'train')
THRESHOLD = 0.85


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(re.findall(r'\w+', text, flags=re.UNICODE))


def grams(text: str) -> frozenset[str]:
    text = normalized(text)
    if len(text) < 3:
        return frozenset([text])
    return frozenset(text[i:i + 3] for i in range(len(text) - 2))


def read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f'Cabecera inesperada en {path}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'Archivo sin registros: {path}')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--custom', type=Path, default=ROOT / 'data/templates/custom_seed_draft.csv',
                        help='CSV revisado; solo se incorporan borradores de Marketing')
    args = parser.parse_args()
    custom = args.custom.resolve()
    if validate(custom, compare_sid=True):
        raise SystemExit('CSV propio inválido; no se generaron particiones.')
    paths = {split: SID / f'{split}_es.csv' for split in PRIORITY}
    splits = {split: read_csv(path, COLUMNS) for split, path in paths.items()}
    custom_rows = read_csv(custom, COLUMNS[:-1])
    selected = [dict(row, original_split='custom_train') for row in custom_rows
                if row['department'] == 'Marketing' and row['source'] == 'borrador_ia']
    if not selected:
        raise SystemExit('No hay borradores de Marketing para entrenamiento.')

    # Índice de trigramas de particiones previas; no se comparan filas de un mismo split.
    index: dict[str, set[int]] = defaultdict(set)
    previous: list[tuple[str, frozenset[str]]] = []
    accepted: dict[str, list[dict[str, str]]] = {}
    removed: Counter[str] = Counter()
    collisions: Counter[str] = Counter()
    for split in PRIORITY:
        candidates = splits[split] + (selected if split == 'train' else [])
        kept = []
        for row in candidates:
            current = grams(row['text'])
            matches: Counter[int] = Counter()
            for gram in current:
                matches.update(index.get(gram, ()))
            found = None
            for idx in sorted(matches):
                overlap = matches[idx]
                prev_split, other = previous[idx]
                if overlap / (len(current) + len(other) - overlap) >= THRESHOLD:
                    found = prev_split
                    break
            if found is not None:
                origin = 'custom' if row['original_split'] == 'custom_train' else split
                removed[origin] += 1
                collisions[f'{origin} contra {found}'] += 1
            else:
                kept.append(row)
        accepted[split] = kept
        for row in kept:
            current = grams(row['text'])
            idx = len(previous)
            previous.append((split, current))
            for gram in current:
                index[gram].add(idx)

    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in accepted.items():
        with (OUT / f'{split}.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        'method': 'SID original splits; prioridad test, validation_real, validation, train; '
                  'Jaccard de trigramas de caracteres normalizados >= 0.85 entre particiones',
        'limitations': 'Heurística conservadora: puede omitir paráfrasis y apartar falsos positivos; '
                       'test es sintético y no demuestra generalización a mensajes humanos.',
        'custom_selection': 'solo Marketing y source=borrador_ia, revisados por el usuario',
        'inputs_sha256': {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): fingerprint(path)
                          for path in [*paths.values(), custom, ROOT / 'data/sid_intent_map.json']},
        'outputs_sha256': {split: fingerprint(OUT / f'{split}.csv') for split in PRIORITY},
        'counts': {split: len(rows) for split, rows in accepted.items()},
        'by_department': {split: dict(sorted(Counter(r['department'] for r in rows).items()))
                          for split, rows in accepted.items()},
        'custom_selected': len(selected),
        'custom_accepted': sum(r['original_split'] == 'custom_train' for r in accepted['train']),
        'quarantined_counts': dict(sorted(removed.items())),
        'quarantined_by_pair': dict(sorted(collisions.items())),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('counts', 'custom_selected', 'custom_accepted', 'quarantined_counts', 'quarantined_by_pair')}, ensure_ascii=False, indent=2))
    print(f'Particiones y manifiesto: {OUT}')


if __name__ == '__main__':
    main()
