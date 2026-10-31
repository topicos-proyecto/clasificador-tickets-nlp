"""Valida estructura y calidad mínima del CSV de datos propios.

Python estándar. Ejemplos:
  python src/data/validate_custom.py data/templates/custom_seed_draft.csv
  python src/data/validate_custom.py data/raw/custom/messages.csv --compare-sid
No entrena ni mezcla archivos; solo inspecciona.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ("text", "department", "source", "intent", "language")
DEPARTMENTS = {
    "Soporte TI", "Finanzas", "Ventas", "Marketing",
    "Atención al Cliente", "Logística"
}
SOURCES = {"propio_humano", "borrador_ia"}


def normalize(text: str) -> str:
    """Clave para duplicados exactos ignorando mayúsculas y espacios (no NLP del modelo)."""
    return re.sub(r"\s+", " ", text.strip()).casefold()


def sid_keys() -> set[str]:
    keys: set[str] = set()
    folder = ROOT / "data" / "processed" / "sid"
    files = [folder / f"{name}_es.csv" for name in ("train", "validation", "validation_real", "test")]
    if not all(p.is_file() for p in files):
        raise FileNotFoundError(
            f"Faltan CSV procesados en {folder}; ejecuta primero python src/data/prepare_sid.py"
        )
    for p in files:
        with p.open("r", encoding="utf-8-sig", newline="") as inp:
            for row in csv.DictReader(inp):
                if (row.get("text") or "").strip():
                    keys.add(normalize(row["text"]))
    return keys


def validate(path: Path, compare_sid: bool) -> int:
    if not path.is_file():
        print(f"ERROR: no se encontró {path}", file=sys.stderr)
        return 2
    counts: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    errors: list[str] = []
    warnings: list[str] = []
    seen: dict[str, int] = {}
    reference = sid_keys() if compare_sid else set()
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as inp:
        reader = csv.DictReader(inp)
        actual = tuple(reader.fieldnames or ())
        if actual != COLUMNS:
            print(
                f"ERROR: cabecera esperada {','.join(COLUMNS)}; encontrada {actual}",
                file=sys.stderr
            )
            return 2
        for number, row in enumerate(reader, start=2):
            rows += 1
            if None in row or any(row.get(k) is None for k in COLUMNS):
                errors.append(f"Línea {number}: CSV mal formado (comas sin comillas o columnas extra).")
                continue
            text = row["text"].strip()
            dept = row["department"].strip()
            origin = row["source"].strip()
            intent = row["intent"].strip()
            lang = row["language"].strip()
            if not text or not dept or not origin or not intent or not lang:
                errors.append(f"Línea {number}: hay un campo vacío.")
                continue
            if dept not in DEPARTMENTS:
                errors.append(f"Línea {number}: departamento desconocido: {dept!r}")
            if origin not in SOURCES:
                errors.append(f"Línea {number}: fuente no admitida: {origin!r}")
            if lang != "es":
                errors.append(f"Línea {number}: se esperaba language=es, encontrado {lang!r}")
            if len(text.split()) < 3:
                warnings.append(f"Línea {number}: texto muy corto; revisar intención: {text[:65]!r}")
            if len(text.split()) > 150:
                warnings.append(f"Línea {number}: texto extenso; revisar: {text[:65]!r}")
            key = normalize(text)
            if key in seen:
                errors.append(f"Línea {number}: duplicado exacto de línea {seen[key]}")
            else:
                seen[key] = number
            if compare_sid and key in reference:
                warnings.append(f"Línea {number}: mismo texto que uno de SID: {text[:65]!r}")
            counts[dept] += 1
            origins[origin] += 1

    print(f"Archivo: {path}")
    print(f"Filas: {rows} | Errores: {len(errors)} | Advertencias: {len(warnings)}")
    print("Distribución por departamento:")
    for dept in sorted(DEPARTMENTS):
        print(f"  {dept}: {counts[dept]}")
    print("Origen declarado:", dict(sorted(origins.items())))
    if origins["borrador_ia"]:
        print("AVISO: contiene borradores de IA; necesitan revisión y no se consideran registros humanos.")
    for item in errors[:30]:
        print("ERROR:", item)
    for item in warnings[:30]:
        print("AVISO:", item)
    if len(errors) > 30 or len(warnings) > 30:
        print("Se muestran solo los primeros 30 errores/advertencias de cada tipo.")
    print("La validación técnica NO valida la coherencia semántica de las etiquetas.")
    return 1 if errors else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revisar CSV propio antes de mezclarse con SID")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--compare-sid", action="store_true",
        help="Avisar si existen textos exactamente iguales en los CSV SID preparados"
    )
    args = parser.parse_args()
    try:
        raise_code = validate(args.csv_path, args.compare_sid)
    except (OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise_code = 2
    sys.exit(raise_code)
