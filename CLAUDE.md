# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Clasificador académico (en español) de solicitudes de **clientes** hacia seis áreas: Soporte TI, Finanzas, Ventas, Marketing, Atención al Cliente, Logística. Este repo es la parte de datos, NLP y modelos; la documentación y los mensajes del proyecto están en español, y conviene mantenerlo así. El [README.md](README.md) tiene el detalle de datos, resultados y pendientes; `docs/` tiene alcance, requisitos y arquitectura.

Los repos hermanos `../clasificador-backend` (FastAPI) y `../clasificador-frontend` (Vite/React) consumen este código: el backend **importa `src/nlp` y carga `artifacts/comparison_customers/pipeline.joblib`** de aquí. Regla central: el pipeline serializado (normalización + TF-IDF + clasificador) es la única fuente de la lógica NLP; nunca duplicarla en el backend. Cambiar `src/nlp/preprocessing.py` o el formato del artefacto afecta a ambos repos.

## Comandos

Python 3.12, entorno en `.venv312/` (ignorado por Git):

```bash
source .venv312/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps 'https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl'

python -m pytest -q                              # todas las pruebas
python -m pytest tests/test_pipeline.py -q       # un archivo
python -m pytest tests/test_pipeline.py::nombre -q   # una prueba
```

Algunas pruebas requieren artefactos/datos locales (no versionados) y fallan o se omiten sin ellos.

Cadena de datos → modelo (ejecutar desde la raíz, en este orden):

```bash
python src/data/prepare_sid.py                    # descarga/verifica SID (SHA-256 fijadas en data/sid_source.json) -> data/processed/sid/
python src/data/build_splits.py                   # particiones base (+90 borradores de Marketing)
python -m src.data.build_customer_splits          # -> data/processed/model_customers/ (no sobrescribe salida existente)
python -m src.models.train                        # baseline -> artifacts/baseline/
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.models.compare \
  --data-dir data/processed/model_customers --output-dir artifacts/comparison_customers
python -m src.models.evaluate_simulated
python -m src.models.analyze_errors --model <pipeline.joblib> --dataset nombre=<csv> [--compare-with <otro.joblib>] [--redact-text]
python -m src.models.evaluate_customer_human      # requiere archivos privados en data/raw/custom/
python src/data/validate_custom.py <csv> --compare-sid
```

`prepare_sid.py` acepta `--offline`; `train.py` acepta `--method lemmatization`, `--ngrams 2`, `--output-dir`.

## Arquitectura y restricciones metodológicas

- `src/nlp/preprocessing.py`: normalización (NFKC, minúsculas, sin URLs/puntuación/emojis, conserva tildes y números), stop words en español **conservando negaciones** (`no, ni, nunca, jamás, tampoco, sin`), stemming o lematización (spaCy `es_core_news_sm`). Compartido por entrenamiento e inferencia.
- `src/models/compare.py`: rejilla (`GRID`, 64 configuraciones) stemming/lematización × unigramas/bigramas × Naive Bayes (`alpha`) / Regresión Logística (`C`, `class_weight`), CV estratificada de 3 particiones (semilla 42) sobre entrenamiento; TF-IDF se ajusta dentro de cada partición. Selección por regla de una desviación estándar (`select`). Opciones `--tfidf-grid`, `--grouped-cv`, `--overwrite`. Guarda pipeline, huellas y métricas en `artifacts/<dir>/`; mantiene en `selected` los campos que lee `/metrics` del backend.
- `src/models/analyze_errors.py`: diagnóstico de errores (confusiones, atribución de términos, margen, causas heurísticas, McNemar). Rechaza `test*.csv` y la huella de `test`; `--redact-text` para datos humanos.
- `src/data/`: preparación de SID, particiones y validación/auditoría de datos propios. Las particiones apartan posibles cuasi-duplicados con trigramas de caracteres (heurística).
- Los scripts `evaluate_*` verifican huellas SHA-256 del modelo y del conjunto congelado antes de evaluar.

Reglas a respetar:
- La partición `test` está **reservada**: no usarla para seleccionar configuraciones ni hiperparámetros; su evaluación final está pendiente.
- Los conjuntos de evaluación humana no deben usarse para ajustar el modelo (si se usan, hacen falta mensajes nuevos).
- No duplicar textos entre entrenamiento y evaluación. Mantener la procedencia `borrador_ia` / `propio_humano` en los CSV.
- `data/raw/`, `data/processed/` y `artifacts/` (salvo la carpeta en sí) están en `.gitignore`: los datos y modelos se regeneran o se comparten por canal privado. `data/raw/custom/` contiene datos humanos que no deben versionarse.
- `compare.py` aborta si `--output-dir` ya tiene `pipeline.joblib` (salvo `--overwrite`). El candidato con la rejilla ampliada vive en `artifacts/comparison_customers_v2/` y no sustituye al que carga el backend hasta decidirlo explícitamente.
- `evaluate_human.evaluate()` tiene fijada la configuración NB `alpha=0,5` lematizada: generalizarla antes de evaluar otro modelo.
- Los resultados son exploratorios (SID es mayormente sintético); no presentar métricas de validación como generalización a clientes reales.

## Flujo de trabajo

Una rama por funcionalidad (`feature/...`) y PR a `main`; commits con prefijo convencional en español (`feat(nlp): ...`, `docs: ...`). Mantener [CONTRIBUCIONES.md](CONTRIBUCIONES.md) alineado con la historia real de GitHub (no atribuir aportes inexistentes).
