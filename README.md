# Clasificador de tickets — NLP y Machine Learning

Proyecto académico para clasificar solicitudes de **clientes**, escritas en español, según el departamento de atención inicial: **Soporte TI, Finanzas, Ventas, Marketing, Atención al Cliente o Logística**. El objetivo es implementar y explicar el procesamiento de lenguaje natural y comparar modelos de clasificación; el modelo no diagnostica ni resuelve la incidencia.

El proyecto prepara datos, analiza su distribución, entrena y compara modelos y guarda el pipeline completo para reutilizar el mismo procesamiento en inferencia. Los resultados actuales son **exploratorios**: SID es mayormente sintético y las muestras humanas son pequeñas.

## Datos

La fuente pública es [SID — Synthetic Intent Dataset](https://huggingface.co/datasets/luigicfilho/sid), filtrada a español y mapeada a las seis áreas con `data/sid_intent_map.json`. El mapeo de 44 intenciones fue revisado por el usuario. Los archivos originales, su licencia y las decisiones metodológicas se describen en [Dataset SID](docs/DATASET_SID.md).

La primera preparación aceptó **17.500 registros SID**. Marketing es la clase menos representada, con 343 registros. Este desbalance, la similitud entre textos sintéticos y la cobertura limitada de situaciones reales deben tenerse en cuenta al interpretar las métricas. Ver [resultados SID](docs/RESULTADOS_SID.md).

También se utilizan ejemplos redactados para el proyecto:

- `data/templates/custom_seed_draft.csv`: 130 borradores de IA; los 90 de Marketing fueron revisados y se incorporan a entrenamiento. Los 40 restantes siguen pendientes de revisión.
- `data/templates/internal_support_draft.csv`: 30 borradores de IA orientados a clientes, revisados y agregados a una versión posterior del entrenamiento. El nombre del archivo es histórico.
- `data/templates/customer_simulated_eval.csv`: 24 ejemplos simulados con IA, revisados por el usuario y usados para una prueba exploratoria. **No son mensajes humanos reales.**
- `data/raw/custom/`: datos locales no versionados, incluidos mensajes escritos por personas y sus conjuntos de evaluación. La [guía de datos propios](docs/GUIA_DATASET_PROPIO.md) explica el formato, la procedencia y la revisión de etiquetas.

La validación automática comprueba formato y algunas coincidencias, pero no garantiza que una etiqueta sea correcta ni detecta todas las paráfrasis. Mantener la procedencia `borrador_ia` o `propio_humano` según corresponda.

## Instalación

Desde la raíz del proyecto, con **Python 3.12**:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps 'https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl'
```

Las versiones de dependencias están fijadas en `requirements.txt`. Para ejecutar el notebook EDA se necesitan además `pandas`, `matplotlib` e `ipykernel` en el entorno utilizado por Jupyter.

## Preparación y particiones

```bash
python src/data/prepare_sid.py
python src/data/validate_custom.py data/templates/custom_seed_draft.csv --compare-sid
python src/data/build_splits.py
python -m src.data.build_customer_splits
```

`prepare_sid.py` descarga los archivos originales si no están disponibles, filtra el español y genera `data/processed/sid/`. La fuente está fijada a la revisión `43fad0f7d60bd1dfb0402c8632deb2cbc13f4731`; `data/sid_source.json` registra las huellas SHA-256 esperadas de los cinco archivos. También admite `--offline` si los archivos ya existen en `data/raw/sid/`. En ambos modos verifica todas las huellas antes de generar datos procesados. Si un archivo no coincide, conserva o muévelo fuera del directorio y vuelve a descargarlo; el script no sustituye automáticamente datos existentes.

`build_splits.py` crea las particiones base y añade los 90 borradores revisados de Marketing al entrenamiento. Después, `build_customer_splits.py` agrega los 30 borradores orientados a clientes en una versión separada. Sus resultados están en `data/processed/model_customers/`:

| Partición | Registros |
|---|---:|
| Entrenamiento | 15.560 |
| Validación | 194 |
| Validación adicional (`validation_real`) | 448 |
| Prueba (`test`) | 707 |

Las particiones priorizan el conjunto de prueba y apartan posibles similitudes mediante trigramas de caracteres. Es una heurística: puede excluir falsos positivos y no garantiza detectar todas las paráfrasis. `test` permanece reservado y no se usa para seleccionar configuraciones. El generador de la versión para clientes **no sobrescribe** un directorio de salida existente; conserva o mueve los resultados anteriores antes de regenerarlo.

Los archivos crudos, procesados y artefactos están excluidos de Git. Cada integrante debe regenerarlos o conservar una copia apropiada de los archivos locales necesarios.

## Pipeline NLP

El pipeline utiliza normalización, tokenización, stop words en español, **stemming o lematización** y TF-IDF con **unigramas o unigramas y bigramas**. La normalización aplica Unicode NFKC y minúsculas, elimina URLs, puntuación y emojis, y conserva tildes y números. Las negaciones `no`, `ni`, `nunca`, `jamás`, `jamas`, `tampoco` y `sin` se conservan al retirar stop words.

Ejemplo con «No puedo aplicar el cupón de descuento 😊»:

| Etapa | Resultado |
|---|---|
| Normalización | `no puedo aplicar el cupón de descuento` |
| Stop words retiradas | `puedo`, `el`, `de` |
| Tokens conservados | `no`, `aplicar`, `cupón`, `descuento` |
| Stemming | `no`, `aplic`, `cupon`, `descuent` |
| Lematización | `no`, `aplicar`, `cupón`, `descuento` |

El stemming puede eliminar tildes y agrupar palabras distintas bajo una misma raíz. Las decisiones de preprocesamiento deben justificarse con los experimentos y el análisis de errores.

## Entrenamiento y comparación

Para entrenar la referencia Naive Bayes con stemming y unigramas, usando las particiones base:

```bash
python -m src.models.train
```

La referencia lee `train.csv` y `validation.csv` y escribe el pipeline y sus métricas en `artifacts/baseline/`. También admite `--method lemmatization`, `--ngrams 2` y `--output-dir`. El comando interactivo `python -m src.models.predict_cli` carga esa referencia y permite probar mensajes en terminal.

Para comparar Naive Bayes y Regresión Logística con las particiones orientadas a clientes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.models.compare \
  --data-dir data/processed/model_customers \
  --output-dir artifacts/comparison_customers
```

La comparación prueba stemming y lematización, unigramas y bigramas, y dos valores de hiperparámetros por modelo. Usa validación cruzada estratificada de tres particiones sobre entrenamiento, con semilla 42; TF-IDF se ajusta dentro de cada partición. Guarda el pipeline seleccionado, parámetros, versiones, huellas, F1 por clase y matriz de confusión. Los artefactos locales existentes en el directorio de salida se pueden sobrescribir: usar otro `--output-dir` si se desea conservarlos.

## Evaluación y resultados actuales

En la comparación orientada a clientes, la configuración seleccionada fue **Naive Bayes con lematización, unigramas y `alpha=0,5`**. Obtuvo F1 macro medio de **0,9585** en validación cruzada y **0,9452** en la validación SID. La validación contiene solo cuatro ejemplos de Marketing y puede haber variantes similares dentro de entrenamiento, por lo que estas cifras no demuestran generalización a clientes reales.

Como pruebas adicionales, el modelo acertó **18 de 24** mensajes simulados con IA y **18 de 30** mensajes escritos por personas desde la perspectiva de clientes. El F1 macro de la evaluación humana fue **0,5952**; hay solo entre tres y siete mensajes por área. Estos conjuntos y la validación SID son distintos, de modo que sus cifras no deben compararse directamente como si midieran la misma población.

Para reproducir estas evaluaciones, si están disponibles sus archivos y artefactos locales:

```bash
python -m src.models.evaluate_simulated
python -m src.models.evaluate_customer_human
```

La evaluación humana comprueba las huellas del modelo y del conjunto congelado antes de ejecutarse. Para reproducirla, el equipo debe recibir por un canal privado `data/raw/custom/human_evaluation_clients_frozen.csv`, `data/raw/custom/human_evaluation_clients_manifest.json` y el `artifacts/comparison_customers/pipeline.joblib` correspondiente; las huellas del manifiesto deben coincidir. Estos archivos son locales y no se versionan. Si los resultados se usan para modificar o seleccionar un modelo, se necesitan otros mensajes humanos reservados para una evaluación independiente posterior. La evaluación final sobre `test` sigue pendiente. El detalle de los errores está en el informe local `artifacts/comparison_customers/human_clients_report.json`.

## Pruebas y estructura

```bash
python -m pytest -q
```

Las pruebas abarcan el procesamiento, la comparación, las particiones y la auditoría de datos. Algunas requieren los artefactos locales. La instalación actual puede emitir una advertencia de SciPy sobre `iprint` durante Regresión Logística.

- `src/data/`: preparación, validación y particiones.
- `src/nlp/`: normalización y tokenización compartidas.
- `src/models/`: entrenamiento, comparación e inferencia.
- `data/templates/`: plantillas y borradores versionados.
- `notebooks/`: análisis exploratorio.
- `docs/`: alcance, fuentes, requisitos y análisis.
- `tests/`: pruebas automatizadas del proyecto NLP.

## Pendientes

- Revisar más ejemplos por clase, especialmente Marketing, y documentar casos ambiguos.
- Ampliar y revisar datos de clientes con procedencia clara, sin duplicar textos entre entrenamiento y evaluación.
- Mantener una evaluación humana independiente para futuras decisiones y, una vez fijado el modelo final, evaluar el conjunto `test` reservado.
- Completar el informe técnico, el registro de contribuciones y la preparación de la defensa.
