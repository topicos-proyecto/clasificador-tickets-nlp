"""Descarga y prepara SID en español para revisar la viabilidad de 6 departamentos.

Sin dependencias externas. No entrena modelos.
Ejecución: python src/data/prepare_sid.py
Para reutilizar archivos descargados: python src/data/prepare_sid.py --offline
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "sid"
OUT = ROOT / "data" / "processed" / "sid"
MAPPING_PATH = ROOT / "data" / "sid_intent_map.json"
SOURCE_PATH = ROOT / "data" / "sid_source.json"
FILES = {
    "train": "train.csv",
    "validation": "val.csv",
    "validation_real": "val_real.csv",
    "test": "test.csv",
}
# En caso de duplicados exactos entre splits, preservamos primero los de evaluación.
PRIORITY = ("test", "validation_real", "validation", "train")
COLUMNS = ("text", "department", "source", "intent", "language", "original_split")


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 incorrecto para {path}: esperado {expected}, obtenido {actual}. "
            "Conserva o mueve el archivo existente y vuelve a descargar la revisión fijada."
        )


def download_file(filename: str, offline: bool, source: dict) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / filename
    if path.is_file():
        verify_sha256(path, source["sha256"][filename])
        print(f"Usando archivo existente: {path}")
        return path
    if offline:
        raise FileNotFoundError(f"No existe {path}; descarga el archivo antes o quita --offline")
    print(f"Descargando {filename}...")
    url = f'{source["source_url"]}/resolve/{source["revision"]}/{filename}'
    req = Request(url, headers={"User-Agent": "NLP-University-Data-Preparation/1.0"})
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(req, timeout=120) as response, tmp.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)
        verify_sha256(tmp, source["sha256"][filename])
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mapping() -> dict[str, str]:
    groups = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for department, intents in groups.items():
        for intent in intents:
            if intent in result:
                raise ValueError(f"Intención duplicada en mapeo: {intent}")
            result[intent] = department
    if len(groups) != 6:
        raise ValueError("Se esperan exactamente seis departamentos.")
    return result


def key_text(text: str) -> str:
    # Solo para deduplicar; NO es el pipeline NLP de entrenamiento.
    return re.sub(r"\s+", " ", text.strip()).casefold()


def prepare_split(
    split: str, path: Path, mapping: dict[str, str], seen: set[str]
) -> tuple[dict, Counter]:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    distribution = Counter()
    excluded_intents = Counter()
    lengths: list[int] = []
    dest = OUT / f"{split}_es.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as source, dest.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        required = {"text", "label", "language"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: faltan columnas {required}. Encontradas: {reader.fieldnames}")
        writer = csv.DictWriter(target, fieldnames=COLUMNS)
        writer.writeheader()
        for row in reader:
            stats["source_rows"] += 1
            if row["language"].strip().lower() != "es":
                continue
            stats["spanish_rows"] += 1
            text = (row.get("text") or "").strip()
            intent = (row.get("label") or "").strip()
            if not text or not intent:
                stats["missing_text_or_label"] += 1
                continue
            department = mapping.get(intent)
            if not department:
                stats["unmapped_rows"] += 1
                excluded_intents[intent] += 1
                continue
            key = key_text(text)
            if key in seen:
                stats["duplicate_rows"] += 1
                continue
            seen.add(key)
            distribution[department] += 1
            lengths.append(len(text.split()))
            writer.writerow({
                "text": text, "department": department, "source": "SID",
                "intent": intent, "language": "es", "original_split": split
            })
            stats["accepted_rows"] += 1

    info = {
        "split": split,
        "file": str(dest.relative_to(ROOT)),
        "counts": dict(stats),
        "by_department": dict(sorted(distribution.items())),
        "unmapped_intents_top": excluded_intents.most_common(20),
        "average_words": round(sum(lengths) / len(lengths), 2) if lengths else None,
        "min_words": min(lengths) if lengths else None,
        "max_words": max(lengths) if lengths else None,
    }
    return info, distribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparar SID español y reportar cobertura por departamento")
    parser.add_argument("--offline", action="store_true", help="Usar CSV originales ya descargados")
    args = parser.parse_args()
    mapping = load_mapping()
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))

    # Comprobar todos los originales antes de escribir cualquier archivo procesado.
    paths = {split: download_file(name, args.offline, source) for split, name in FILES.items()}
    paths["intents"] = download_file("intents.json", args.offline, source)

    # Verificar que las intenciones del mapeo existan en el catálogo publicado.
    catalog = json.loads(paths["intents"].read_text(encoding="utf-8"))
    available = set(catalog["intents"])
    unknown = set(mapping) - available
    if unknown:
        raise ValueError(f"Intenciones no encontradas en SID: {sorted(unknown)}")

    seen: set[str] = set()
    summaries: list[dict] = []
    total_by_department = Counter()
    for split in PRIORITY:
        info, counts = prepare_split(split, paths[split], mapping, seen)
        summaries.append(info)
        total_by_department.update(counts)
        print(f"{split}: {info['counts'].get('accepted_rows', 0)} filas aceptadas")

    manifest = {
        "source_url": source["source_url"],
        "dataset_revision": source["revision"],
        "license_declared": source["license_declared"],
        "downloaded_or_verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    summary = {
        "mapping_status": "PRELIMINARY_REQUIRES_HUMAN_REVIEW",
        "mapped_intents": len(mapping),
        "departments": sorted(set(mapping.values())),
        "all_splits_by_department": dict(sorted(total_by_department.items())),
        "splits": summaries,
        "note": "Deduplicación solo exacta/casefold; revisar paráfrasis y fugas antes de evaluar.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nConteo total por departamento:")
    for name, count in sorted(total_by_department.items()):
        print(f"  {name}: {count}")
    print(f"\nResumen: {OUT / 'summary.json'}")
    print("El siguiente paso es revisar etiquetas y añadir datos propios; aún NO entrenar.")


if __name__ == "__main__":
    main()
