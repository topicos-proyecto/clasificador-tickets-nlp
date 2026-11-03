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

### Clasificador de terminal

La mejora de [`src/models/predict_cli.py`](src/models/predict_cli.py) permite elegir uno o varios modelos, inspeccionar sus probabilidades por área y explicar los pasos de preprocesamiento del mensaje. Antes, el comando solo cargaba la referencia `baseline` y mostraba el área sugerida.

Para elegir otro pipeline ya entrenado, usar `--model`. En PowerShell, desde la carpeta `clasificador-tickets-nlp`, ejecutar con el entorno Python 3.12 del proyecto:

```powershell
.\.venv312\Scripts\python.exe -m src.models.predict_cli --model artifacts/comparison_customers/pipeline.joblib
```

Escribir `salir` para cerrar el clasificador. La ruta indicada en `--model` es relativa a la carpeta desde la que se ejecuta el comando.

Cada consulta muestra el área sugerida, las probabilidades de todas las áreas (de mayor a menor) y los pasos del pipeline guardado: normalización, tokens, stop words retiradas, tokens conservados, stemming o lematización y cantidad de términos reconocidos por TF-IDF. Si no reconoce vocabulario, solicita más detalles sin asignar un área. Los saludos y textos vacíos también solicitan detalles.

Para comparar varios modelos en una misma sesión, repetir `--model`. Cada mensaje se procesa con cada pipeline y al final aparece una tabla con el área y la probabilidad sugeridas por cada uno. Por ejemplo, **después de generar los finalistas v3** con la comparación del siguiente apartado:

```powershell
.\.venv312\Scripts\python.exe -m src.models.predict_cli `
  --model artifacts/comparison_customers_v3/finalists/naive_bayes.joblib `
  --model artifacts/comparison_customers_v3/finalists/logistic_regression.joblib `
  --model artifacts/comparison_customers_v3/finalists/knn.joblib
```

Los archivos `.joblib` deben existir localmente; Git no descarga los artefactos entrenados. La traza se muestra por modelo porque sus métodos y vocabularios pueden ser distintos. Las probabilidades son estimaciones no calibradas: una cifra mayor no demuestra que un modelo sea mejor. Esta comparación de mensajes es exploratoria; no sustituye la evaluación con etiquetas y no modifica los modelos.

#### Decisiones técnicas

- `--model` es opcional y repetible. Sin argumentos se conserva `artifacts/baseline/pipeline.joblib`; con varias rutas, los modelos se cargan una sola vez al iniciar y reciben el mismo mensaje en cada consulta.
- La traza procede de `tfidf.tokenizer.trace()` del artefacto. TF-IDF y el clasificador se reutilizan sin ejecutar `fit`, conservando el procesamiento y vocabulario aprendidos.
- Las probabilidades se asocian a las etiquetas mediante `classifier.classes_`, se muestran como porcentajes ordenados y el área sugerida corresponde a la probabilidad máxima. Se admiten los pipelines del proyecto con Naive Bayes, Regresión Logística y KNN.
- Cada modelo se identifica por su número, tipo de clasificador y ruta. La tabla final resume área y probabilidad e indica coincidencias, diferencias o modelos sin clasificación; no selecciona un ganador ni promociona artefactos al backend.
- Un archivo ausente, corrupto o incompatible impide iniciar la sesión y devuelve código de salida 1. Un error al procesar un mensaje con un modelo se muestra sin impedir que los demás se ejecuten.
- `salir`, fin de entrada y `Ctrl+C` cierran la sesión con código 0. Los saludos, entradas vacías y mensajes sin vocabulario reconocido solicitan más detalles.

#### Verificación de la mejora

Desde la raíz de este repositorio:

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_predict_cli.py tests/test_pipeline.py -q
```

Resultado registrado durante la implementación: **21 pruebas aprobadas** (17 casos del clasificador de terminal y 4 del pipeline). Se verificaron probabilidades y etiquetas contra los pipelines serializados, comparación de los tres clasificadores en una sesión, stemming y lematización, conservación del vocabulario, uso del modelo predeterminado, mensajes fuera de vocabulario, archivos inválidos, interrupciones y continuidad ante el fallo de un modelo.

Las pruebas de comparación entrenan modelos pequeños con ejemplos definidos en [`tests/test_predict_cli.py`](tests/test_predict_cli.py), guardados en directorios temporales. No usan datos privados, las particiones de evaluación ni los artefactos de producción. Se observaron advertencias de dependencias de spaCy/Click y SciPy (`iprint`), sin fallos de pruebas.

También se realizó una prueba manual con `artifacts/comparison_customers/pipeline.joblib`: «No puedo ingresar a mi cuenta» mostró el área Soporte TI, seis probabilidades y la traza de lematización; «zzxyzzxy» no recibió clasificación. Esta comprobación verifica el funcionamiento de la herramienta, no la calidad general del modelo. La comparación con los tres finalistas v3 requiere que sus archivos estén disponibles localmente; su integración en el comando se verificó con los modelos temporales de las pruebas.

#### Errores frecuentes

| Mensaje o situación | Acción |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Ejecutar desde `clasificador-tickets-nlp`, no desde el backend. Si estás en `clasificador-backend`, usar `cd ..\clasificador-tickets-nlp`. |
| Se ejecuta `C:\Python314\python.exe` u otro Python global | Usar explícitamente `.\.venv312\Scripts\python.exe` (Python 3.12). |
| No se encontró el modelo | Comprobar la ruta con `Test-Path` y generar u obtener el `.joblib` correspondiente. No basta con descargar el código del repositorio. |
| Se requiere un pipeline entrenado | Elegir un artefacto con los pasos `tfidf` y `classifier`, tokenizador con trazas y clasificador con `predict_proba`. |
| Falta `es_core_news_sm` | Instalar el modelo español indicado en el apartado Instalación, dentro del mismo entorno Python. |

### Comparación de modelos

Para comparar **Naive Bayes, Regresión Logística y KNN** con las particiones orientadas a clientes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.models.compare \
  --data-dir data/processed/model_customers \
  --output-dir artifacts/comparison_customers_v3 --grouped-cv
python -m src.data.dataset_stats            # distribución de clases y longitudes (solo agregados)
python -m src.models.evaluate_test          # evaluación ÚNICA en test, después de fijar la selección
python -m src.models.render_comparison_report   # genera docs/COMPARACION_MODELOS.md
```

La comparación prueba stemming y lematización, unigramas y bigramas, y tres modelos:

- Naive Bayes: `alpha ∈ {0,01; 0,03; 0,1; 0,3; 0,5; 1,0}`.
- Regresión Logística: `C ∈ {0,1; 1; 10; 30; 100}`, con y sin `class_weight='balanced'`.
- KNN con distancia coseno: `k ∈ {1; 3; 5; 7; 11; 15; 25}`, con pesos `uniform` o `distance`.

En total son 120 configuraciones. Se usa validación cruzada estratificada de tres particiones sobre entrenamiento, con semilla 42; TF-IDF se ajusta dentro de cada partición.

La selección sigue la **regla de una desviación estándar**: entre las configuraciones cuya media de F1 macro no queda por debajo de la mejor menos su desviación, se elige la más simple. El orden de simplicidad es:

1. unigramas;
2. Naive Bayes, luego Regresión Logística y luego KNN;
3. mayor regularización;
4. sin pesos;
5. stemming.

Para la mejor configuración de cada modelo, `report.json` guarda:

- las métricas fuera de partición del CV;
- validación, validación adicional y el conjunto simulado (exploratorio);
- métricas por clase y matriz de confusión;
- costo y diferencias pareadas.

Los pipelines finalistas quedan en `finalists/`. Opciones: `--tfidf-grid`, `--grouped-cv` y `--overwrite`. **Si `--output-dir` ya contiene `pipeline.joblib`, la ejecución se aborta** salvo con `--overwrite`. `evaluate_test.py` se niega a ejecutarse dos veces sobre el mismo directorio.

Para analizar los errores de un pipeline sin tocar `test`:

```bash
python -m src.models.analyze_errors --model artifacts/comparison_customers_v2/pipeline.joblib \
  --dataset validacion=data/processed/model_customers/validation.csv \
  --dataset simulado=data/templates/customer_simulated_eval.csv \
  --compare-with artifacts/comparison_customers/pipeline.joblib
```

Genera `error_analysis.json`, `error_analysis.md` y `error_review.csv` (para anotar a mano el motivo de cada error). `--redact-text` omite textos y términos; el comando rechaza cualquier archivo `test*.csv` o con la huella de `test`. Las causas son heurísticas. Resultados y lectura en [Análisis de errores](docs/ANALISIS_ERRORES.md).

## Evaluación y resultados actuales

**Comparación de tres modelos (v3, `artifacts/comparison_customers_v3/`, modelo que usa el backend).** La regla de una desviación estándar elige **Naive Bayes con lematización, unigramas y `alpha=0,1`**:

- **Naive Bayes:** F1 macro CV 0,9689 ± 0,0033; validación SID 0,9572.
- **Regresión Logística** (stemming, `C=30`, `balanced`): obtiene la mejor media, 0,9715. La diferencia con NB es menor que una desviación estándar.
- **KNN** (`k=5`, `distance`): queda claramente por debajo, con 0,9335. Es el que más cae con intenciones no vistas (CV agrupado 0,34).

En la evaluación única en `test`, posterior a la selección, el F1 macro fue: NB 0,9586, LR 0,9597 y KNN 0,9101. La justificación completa, con tablas por clase y matrices de confusión, está en [Comparación de modelos](docs/COMPARACION_MODELOS.md).

Las cifras de los párrafos siguientes corresponden al modelo anterior (`comparison_customers`, NB `alpha=0,5`). Ese modelo sigue disponible con `MODEL_DIR=artifacts/comparison_customers` en el backend.

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
