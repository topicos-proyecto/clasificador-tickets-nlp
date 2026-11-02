# Plan: rejilla ampliada de Regresión Logística (A) y análisis automático de errores (B)

**Estado:** implementado (A y B) y sin promover; ver «Desviaciones y resultados» al final. **Fecha del plan:** 1 de noviembre de 2026. **Plazo de entrega según `REQUISITOS.md`:** 2 de noviembre, 23:59.
**Orden:** primero A (cambia el modelo ganador y genera los artefactos), después B (los analiza). Cada una va en su propia rama y PR.

## 0. Motivación y evidencia previa

La comparación actual (`artifacts/comparison_customers/report.json`) prueba solo `C ∈ {0,1; 1,0}` para Regresión Logística (LR). Con `C=0,1` queda en 0,77–0,82 de F1 macro (CV), señal de subajuste por regularización excesiva, y con `C=1,0` en 0,92–0,93. Naive Bayes (NB) con `alpha=0,5` gana con 0,9585.

Un experimento exploratorio (stemming, unigramas, mismo CV de 3 particiones y semilla 42, solo sobre `train` de `model_customers`; `test` no se tocó) mostró:

| Configuración LR | F1 macro (CV) |
|---|---:|
| `C=1`, sin pesos (valor ya probado) | 0,9332 |
| `C=10` | 0,9664 |
| `C=30` | 0,9684 |
| `C=100` | 0,9687 |
| `C=1`, `class_weight='balanced'` | 0,9557 |
| `C=30`, `class_weight='balanced'` | **0,9715** |

La conclusión «gana Naive Bayes» del README podría ser un artefacto de una rejilla pobre. Estas cifras son orientativas: se obtuvieron con un script desechable y deben **reproducirse dentro de `compare.py`** antes de citarlas.

**Cautelas que el plan debe respetar** (de `CLAUDE.md`):
- `test` está reservado: ni A ni B lo leen.
- Los conjuntos de evaluación humana no se usan para ajustar ni seleccionar el modelo.
- SID es sintético y el CV puede estar inflado por variantes casi idénticas entre particiones.
- El backend carga `artifacts/comparison_customers/pipeline.joblib`: no se sobrescribe sin una decisión explícita.

---

## Parte A — Rejilla ampliada de Regresión Logística

### A.1 Objetivo y criterios de éxito
1. Ampliar el espacio de hiperparámetros de LR (y algo el de NB, para que la comparación sea justa) manteniendo reproducibilidad, semilla 42 y TF-IDF ajustado dentro de cada partición.
2. Re-seleccionar el modelo con una regla fijada **antes** de mirar resultados humanos.
3. Dejar un informe reproducible con la comparación NB vs LR, incluida la diferencia pareada por partición.
4. No romper el backend, el frontend ni las pruebas existentes.

Éxito: `compare.py` produce un `report.json` en un directorio nuevo, compatible con `/metrics`; las pruebas pasan; el README documenta el resultado, sea cual sea (si NB sigue ganando también se reporta).

### A.2 Diseño de la rejilla
Hoy `PARAMETERS` mapea cada modelo a una tupla de valores de un solo hiperparámetro. Pasa a ser una lista de configuraciones por modelo (dicts), de modo que se pueda variar más de un parámetro.

| Modelo | Hiperparámetros | Valores |
|---|---|---|
| Naive Bayes | `alpha` | 0,1; 0,3; 0,5; 1,0 |
| Regresión Logística | `C` | 0,1; 1; 10; 30; 100 |
| Regresión Logística | `class_weight` | `None`, `'balanced'` |

Total: 2 métodos × 2 n-gramas × (4 + 10) = **56 configuraciones** (hoy 16), 3 particiones cada una.

Decisiones y justificación:
- `C` hasta 100 porque el experimento mostró meseta en 30–100; ir más allá no aporta y se acerca a no regularizar.
- `class_weight='balanced'` por el desbalance de Marketing (≈2 % de los datos); lo recomienda `RESULTADOS_SID.md`.
- Se agregan valores de `alpha` menores: con `alpha=0,5` en el borde de la rejilla (el mejor valor era el mínimo probado), no se sabe si un valor menor sería mejor. Es la misma falla que se corrige en LR.
- Fuera de la rejilla principal: `solver`, `penalty` y `max_iter` (se fijan; ver A.4). Mantener el producto cartesiano acotado.
- **Riesgo de selección optimista:** más configuraciones aumentan la probabilidad de elegir por azar. Mitigación: reportar desviación estándar y diferencia pareada por partición (A.5) y aplicar la regla de desempate de A.3.

### A.3 Regla de selección (fijada de antemano)
1. Mayor F1 macro medio de CV.
2. Si dos configuraciones difieren en menos de **una desviación estándar de la mejor** (regla de 1-SE), preferir la más simple: unigramas antes que bigramas, NB antes que LR, mayor regularización (alpha mayor, C menor), `class_weight=None`, stemming antes que lematización (más barato en inferencia).
3. La regla y su justificación se escriben en `report['selection']`.

Se debe decidir con el equipo si se adopta 1-SE o el criterio actual (empate exacto). 1-SE es más defendible con 56 configuraciones, pero puede mantener NB aunque LR sea marginalmente mejor. Se recomienda 1-SE y reportar ambos resultados.

`ranking()` hoy ordena por `(-f1, ngrams, índice de modelo, índice de método, ±parámetro)`. Debe reescribirse para la regla 1-SE y para incluir `class_weight`; el test `test_stratified_comparison_is_complete_and_reproducible` (que fija `len(first) == 16`) cambia a calcular el tamaño esperado desde la rejilla.

### A.4 Cambios en el código (`src/models/compare.py`)
1. **Rejilla declarativa:** `GRID = {'naive_bayes': [{'alpha': a} ...], 'logistic_regression': [{'C': c, 'class_weight': w} ...]}` y una función `configurations()` que la expande. `key()` incluye todos los parámetros.
2. **Compatibilidad del informe con el backend.** `/metrics` en `clasificador-backend/api/main.py` lee de `selected`: `model, method, ngrams, parameter_name, parameter_value, f1_macro_mean, f1_macro_std`, y compara `parameter_value` con `classifier.C` o `classifier.alpha`. Se conservan esos campos (el hiperparámetro principal sigue en `parameter_name`/`parameter_value`) y se **añade** `class_weight` como campo adicional (`null` o `"balanced"`). El backend ignora campos extra, así que no hay cambio de contrato. Se añade un test que valida las claves.
3. **`classifier()`** acepta `class_weight`. Se sube `max_iter` a 5000 y se mantiene `random_state=SEED`. Los `ConvergenceWarning` siguen tratándose como error y marcando la configuración `failed` (no se ocultan); si hay fallos, se reportan y se documentan en vez de ocultarlos.
4. **Salida segura:** si `--output-dir` ya contiene `pipeline.joblib`, abortar salvo `--overwrite`. Esto protege al artefacto que consume el backend (hoy `CLAUDE.md` advierte que se puede sobrescribir en silencio). El valor por defecto de `--output-dir` es `artifacts/comparison`; la ejecución de A usa `artifacts/comparison_customers_v2`.
5. **Campos nuevos del informe:**
   - `grid` (la rejilla usada) y `n_configurations`.
   - `vocabulary_size` del modelo final.
   - `selection_rule` (texto de A.3) y `runner_up`: la mejor configuración del otro modelo, con la diferencia pareada.
   - `paired_comparison`: para el mejor NB vs mejor LR, diferencia de F1 por partición, media y desviación.
   - `elapsed_seconds` por método (para documentar el costo).
6. **Evaluación de validación** (`validation`) sin cambios; sigue sin leer `test` (`test_used: false` y la prueba existente que comprueba que solo se leen `train.csv` y `validation.csv`).

### A.5 Sub-etapa opcional: TF-IDF (`min_df`, `max_df`)
`REQUISITOS.md` pide documentar tamaño de vocabulario, `min_df` y `max_df`; hoy están fijos en 1 y 1,0. Para no multiplicar la rejilla, se hace un **segundo barrido pequeño solo sobre el mejor modelo** del barrido principal, con la bandera `--tfidf-grid`:
- `min_df ∈ {1, 2, 3}` y `max_df ∈ {1,0; 0,9}`; se reporta el tamaño del vocabulario por combinación.
- Si ninguna mejora más de una desviación estándar, se mantiene `min_df=1`, `max_df=1,0` y se documenta que se evaluó (cumple el requisito sin cambiar el modelo).
- `pretokenized_vectorizer()` recibe `min_df` y `max_df`; `vectorizer()` los usa igual (el test de equivalencia entre ambos debe seguir pasando con valores no triviales).

### A.6 Sensibilidad ante cuasi-duplicados (opcional, si hay tiempo)
SID tiene variantes casi idénticas; un `StratifiedKFold` puede poner variantes de un mismo mensaje en entrenamiento y validación. Verificación barata: repetir el CV de las dos mejores configuraciones con `StratifiedGroupKFold`, agrupando por texto normalizado sin errores ortográficos (por ejemplo, agrupando por trigramas de caracteres como ya hace `build_splits.py`). Si el orden NB/LR se mantiene, la conclusión es robusta; si cambia, se reporta. No cambia la selección: es un análisis de sensibilidad.

### A.7 Pruebas (`tests/test_compare.py`)
- La rejilla se expande a la cantidad esperada y no hay claves duplicadas.
- `classifier('logistic_regression', ...)` respeta `C` y `class_weight`.
- Regla de selección: casos sintéticos con empate dentro de 1-SE (gana la más simple), sin empate (gana la de mayor F1) y con configuraciones `failed` (se excluyen).
- Reproducibilidad: dos ejecuciones con la misma semilla dan resultados idénticos.
- Contrato con el backend: `selected` contiene las claves que `/metrics` necesita y `parameter_value` coincide con `classifier.C` o `.alpha` del pipeline guardado.
- `--output-dir` existente sin `--overwrite` aborta; con la bandera, continúa.
- La prueba actual de que no se lee `test` se conserva.

### A.8 Ejecución y verificación
```bash
source .venv312/bin/activate
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.models.compare \
  --data-dir data/processed/model_customers \
  --output-dir artifacts/comparison_customers_v2
```
Estimación de costo: ≈120 ajustes de LR (más NB). Los bigramas con `C=100` son los más lentos; se espera entre 10 y 40 minutos, sin paralelizar para mantener el determinismo. Si es demasiado, se ejecuta en segundo plano.

Verificación:
1. **Reproducción de la evidencia previa:** los valores de LR (stemming, unigramas) de la tabla de 0. deben coincidir con el nuevo informe (tolerancia 1e-4). Si no coinciden, investigar antes de continuar.
2. Comprobar que `report.json` tiene `test_used: false`, huellas de entrada iguales a las del manifiesto y `model_sha256` correcto.
3. Backend: probar `/metrics` contra el nuevo directorio apuntando temporalmente `ARTIFACTS` (ver A.9) y confirmar que la comparación muestra ambos modelos.

### A.9 Decisión de promoción (puerta explícita)
El nuevo modelo **no reemplaza** a `comparison_customers` automáticamente.
- **Puerta 1 (tras el barrido):** ¿LR supera al mejor NB por más de una desviación estándar en CV? Si no, se mantiene NB y se documenta que la rejilla ampliada no cambió la conclusión.
- **Puerta 2 (evaluación humana):** el conjunto humano de 30 mensajes ya se usó para evaluar el modelo v1. Evaluar v2 sobre ese mismo conjunto es válido solo como **informe**, nunca como criterio para elegir entre v1 y v2: si se usara para elegir, dejaría de ser independiente y haría falta un conjunto humano nuevo. La decisión de promover se toma con el CV y las validaciones SID/simulada, y se escribe en el PR antes de ver los resultados humanos.
- **Si se promueve:** PR separado en `clasificador-backend` que haga configurable `ARTIFACTS` (variable de entorno con el valor actual por defecto), más actualización de `human_clients_report.json` y de la huella en el manifiesto. Hoy `/metrics` exige que el informe de comparación y el modelo coincidan por SHA-256; la regeneración del artefacto invalida el informe humano del modelo anterior (el endpoint lo mostrará como `human_clients: null` hasta reevaluar).
- Hay que revisar antes por qué `test_health_and_metrics_without_exposing_example_text` falla en este equipo (`TypeError: 'NoneType' object is not subscriptable`): probablemente falta `data/raw/custom/human_evaluation_clients_*`, que no está en este equipo. Distinguir fallo ambiental de fallo real antes de tocar el backend.

### A.10 Documentación y entrega de A
- `README.md`: actualizar «Entrenamiento y comparación» y «Evaluación y resultados actuales» con los resultados reales, la nueva rejilla y la nota de la regla de selección.
- `CLAUDE.md`: actualizar la descripción de `compare.py` y el aviso de sobrescritura.
- `docs/ARQUITECTURA.md`: experimento 4 (hiperparámetros) y la nota sobre `min_df`/`max_df`.
- Rama `feature/rejilla-lr-ampliada`; commits en español con prefijo convencional (`feat(modelos): ...`, `test(modelos): ...`, `docs: ...`); PR a `main` revisado por otro integrante.

---

## Parte B — Análisis automático de errores

### B.1 Objetivo
`REQUISITOS.md` exige «analizar ejemplos de errores y posibles motivos». Hoy los informes guardan la lista cruda de errores (`errors`), sin explicación. B construye un comando reproducible que convierte un pipeline y un conjunto de evaluación en un análisis estructurado y un borrador de informe, sin tocar `test`.

**Principios:**
- Es una herramienta de **diagnóstico**, no de ajuste: no se usa para seleccionar hiperparámetros con conjuntos reservados.
- Las categorías automáticas son **heurísticas**; el motivo real de cada error lo confirma una persona (B.6).
- Los textos de la evaluación humana son privados: el resultado versionado en `docs/` solo contiene agregados y ejemplos de SID o simulados.

### B.2 Entradas y salida
Comando previsto:
```bash
python -m src.models.analyze_errors \
  --model artifacts/comparison_customers_v2/pipeline.joblib \
  --dataset validation=data/processed/model_customers/validation.csv \
  --dataset simulated=data/templates/customer_simulated_eval.csv \
  --dataset humano=data/raw/custom/human_evaluation_clients_frozen.csv \
  --compare-with artifacts/comparison_customers/pipeline.joblib \
  --output-dir artifacts/comparison_customers_v2/error_analysis
```
- Cada `--dataset nombre=ruta` es opcional y repetible; el humano se omite si el archivo no existe (mismo comportamiento que la API).
- **Protecciones obligatorias:** se rechaza cualquier ruta cuyo nombre sea `test.csv` o cuya huella SHA-256 coincida con la de `test` en `data/processed/*/manifest.json`; el informe registra la huella del modelo y de cada conjunto. Para el conjunto humano se reutiliza la verificación de huellas de `src/models/evaluate_human.py` (leerla antes de implementar para no duplicar lógica).
- Salidas en `--output-dir` (ignoradas por Git junto con `artifacts/`):
  - `error_analysis.json`: agregados y casos.
  - `error_analysis.md`: resumen legible, con opción `--redact-text` que omite los textos.
  - `error_review.csv`: una fila por error con columnas para anotación manual (B.6).

### B.3 Módulos de análisis
Cada uno es una función pura y probable por separado dentro de `src/models/analyze_errors.py`.

1. **Resumen por conjunto:** exactitud, F1 macro, precisión/recall/F1 por clase, soporte. Para clases con soporte < 30 se añaden intervalos de Wilson del 95 % para recall y exactitud, y una advertencia explícita (Marketing tiene 4 ejemplos en validación).
2. **Pares de confusión:** ranking de `real → predicho` con conteos y proporción respecto del soporte de la clase real; destaca los asimétricos (p. ej. Ventas → Finanzas o Soporte TI).
3. **Atribución de términos** (funciona para ambos modelos con el pipeline guardado, usando `tfidf.get_feature_names_out()` y el clasificador):
   - *Global:* términos más influyentes por clase. En LR, los coeficientes `coef_`; en NB, la razón de log-probabilidades `feature_log_prob_[c] − promedio de las demás`.
   - *Por error:* para cada mensaje mal clasificado, los términos que más empujaron hacia la clase predicha y los que más favorecían a la real. En LR, `coef_[c] · x`; en NB, `x · feature_log_prob_[c]`. Se usa la misma `trace()` del tokenizador para mostrar normalización, stop words y raíces o lemas, de modo que la explicación coincida con el pipeline real (regla central del proyecto: no duplicar la lógica NLP).
4. **Margen y calibración:** margen entre la probabilidad mayor y la segunda, distribuido por aciertos y errores; fracción de errores bajo cada margen; tabla de fiabilidad por intervalos (con advertencia de n pequeño) y ECE. Es **información** para el trabajo futuro del umbral de rechazo; no fija umbrales (`ARQUITECTURA.md`: «ningún umbral arbitrario constituye garantía»).
5. **Etiquetado heurístico de causas probables** (una etiqueta puede combinarse con otras):

   | Etiqueta | Regla |
   |---|---|
   | `sin_terminos_conocidos` | vector TF-IDF vacío (`nnz == 0`) |
   | `ruido_ortografico` | proporción alta de tokens fuera del vocabulario del TF-IDF |
   | `texto_corto` | ≤ 3 tokens tras retirar stop words |
   | `multi_intencion` | conjunción (`y`, `pero`, `además`, `también`) y términos distintivos de dos o más clases entre los 20 principales de cada clase |
   | `negacion` | contiene alguna de las negaciones conservadas (`NEGATIONS`) |
   | `termino_ambiguo` | el término de mayor aporte hacia la clase predicha es también uno de los principales de la real (palabras compartidas, p. ej. «factura», «pedido») |
   | `margen_bajo` | margen en el cuartil inferior |

   Los umbrales de estas reglas son constantes con nombre y se prueban con unidades; no se calibran con `test`.
6. **Comparación entre modelos (`--compare-with`):** sobre el mismo conjunto, errores corregidos, nuevos y comunes; prueba de McNemar exacta (`scipy.stats.binomtest`) sobre los desacuerdos, con la advertencia de que con conjuntos de 24–30 mensajes casi nunca habrá significación. Sirve para responder «¿LR corrige los errores de NB o solo cambia cuáles comete?».
7. **Casos diagnóstico:** repetir los `DIAGNOSTICS` de `compare.py` (p. ej. «no se descargó la factura» → Finanzas esperada, Soporte TI predicha) con su atribución, como ejemplos de multi-intención ya documentados.

### B.4 Estructura de `error_analysis.json`
```
{
  "model_sha256", "compared_model_sha256",
  "datasets": { "<nombre>": { "sha256", "records", "source" } },
  "per_dataset": { "<nombre>": {
      "summary": {...}, "per_class": {...}, "confusion_pairs": [...],
      "margins": {...}, "calibration": {...},
      "cause_tags": { "<etiqueta>": conteo },
      "errors": [ { "text", "expected", "predicted", "margin", "tags": [...],
                    "trace": {...}, "top_terms_predicted": [...], "top_terms_expected": [...] } ]
  } },
  "global_terms": { "<clase>": [...] },
  "model_comparison": {...},
  "limitations": "..."
}
```
Con `--redact-text`, `text` y `trace` se omiten en el `.json`/`.md` (los `tags` y los agregados se conservan).

### B.5 Pruebas (`tests/test_analyze_errors.py`)
Con un pipeline sintético diminuto (sin spaCy: stemming) para que sean rápidas:
- **Corrección de la atribución:** en LR, la suma de contribuciones por término más el intercepto es igual a `decision_function`; en NB, la suma más el log-prior de la clase es igual al log-joint (`predict_joint_log_proba` o equivalente) salvo la constante de normalización. Es la prueba más importante: valida que la explicación corresponde al modelo real.
- Pares de confusión y Wilson con casos calculados a mano.
- Cada regla de etiqueta con un ejemplo positivo y uno negativo.
- Protección: rechaza `test.csv` y un archivo cuya huella coincide con la de test en el manifiesto.
- `--redact-text` no deja textos en ninguna salida.
- Determinismo: dos ejecuciones producen JSON idéntico (excepto marcas de tiempo, que no se incluyen).
- McNemar con tabla de desacuerdos conocida.

### B.6 Revisión humana y documento final
1. Cada integrante revisa `error_review.csv` (columnas `motivo_manual`, `etiqueta_discutible`, `comentario`), con prioridad en errores de SID/simulados; los humanos se revisan solo localmente.
2. Se agrupan los motivos en la taxonomía final: ruido ortográfico del dataset, ambigüedad entre áreas por la definición operacional (`ALCANCE.md`), multi-intención, etiqueta discutible de SID, vocabulario ausente.
3. Los errores por etiqueta discutible se **anotan, no se corrigen**; si hay que cambiar etiquetas de entrenamiento, se hace en un cambio aparte con revisión (`GUIA_DATASET_PROPIO.md`).
4. Se escribe `docs/ANALISIS_ERRORES.md` (a mano, con las cifras de `error_analysis.md`): resumen cuantitativo, taxonomía con ejemplos de SID/simulados (sin textos humanos), comparación NB vs LR y limitaciones. Debe incluir el aviso de que las métricas no son generalización a clientes reales.

### B.7 Documentación y entrega de B
- `README.md`: sección breve «Análisis de errores» con el comando; `CLAUDE.md`: añadir el comando a la lista.
- `docs/REQUISITOS.md`: marcar «Analizar ejemplos de errores y posibles motivos» solo cuando `ANALISIS_ERRORES.md` esté escrito y revisado.
- Rama `feature/analisis-errores-automatico`, PR a `main` revisado por otro integrante.

---

## Cronograma sugerido (hasta el 2 de noviembre, 23:59)

| Orden | Tarea | Duración estimada |
|---|---|---|
| 1 | Rama de A, rejilla declarativa, regla de selección y pruebas (A.2–A.4, A.7) | 1,5–2 h |
| 2 | Ejecución de `compare.py` en segundo plano (A.8); entretanto, esqueleto de B (B.2, B.3 puntos 1–2) | 0,5–1 h de espera |
| 3 | Verificación de A, puerta 1 y 2, README y PR de A (A.8–A.10) | 1 h |
| 4 | B: atribución de términos y su prueba de corrección (B.3 punto 3, B.5) | 1,5 h |
| 5 | B: margen, etiquetas, comparación y salidas (B.3 puntos 4–7) | 1,5 h |
| 6 | Ejecución de B sobre el modelo final, revisión manual, `ANALISIS_ERRORES.md` y PR | 1,5–2 h |

**Corte mínimo si falta tiempo:** A completo sin A.5/A.6, y de B los módulos 1, 2, 3 y 5 con `error_review.csv` y el documento. Se posponen la calibración y McNemar.

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| El CV favorece a LR por filtración entre variantes de SID | Regla 1-SE; sensibilidad con `StratifiedGroupKFold` (A.6); conclusión basada también en validación SID y simulada, y reporte humano solo informativo |
| Elegir entre muchas configuraciones infla el resultado | Diferencia pareada por partición; la validación (194) no se usa para seleccionar |
| Sobrescribir el artefacto que usa el backend | Salida a `comparison_customers_v2` y protección `--overwrite` |
| Cambiar el modelo invalida `human_clients_report.json` | Aceptado y documentado: `/metrics` pasa a mostrar `human_clients: null` hasta reevaluar; no se seleccionan modelos con el conjunto humano |
| Filtrar textos humanos al repositorio | Artefactos ignorados por Git; `--redact-text`; `docs/` solo con ejemplos de SID o simulados |
| Falsa precisión en clases con pocos ejemplos | Intervalos de Wilson y advertencias de soporte bajo |
| Etiquetas heurísticas presentadas como causas | Se rotulan «causa probable» y se confirman manualmente (B.6) |
| LR no converge en algunas configuraciones | `max_iter=5000`; los fallos quedan registrados como `failed` y se reportan |
| Tiempo de ejecución de bigramas | Ejecutar en segundo plano; recorte de rejilla (quitar `C=100`) si excede el plazo |

## Registro de contribuciones
`CONTRIBUCIONES.md` se actualiza **después** de fusionar cada PR, con el usuario de GitHub real y los enlaces a commits y PR verificables. No se anotan aportes antes de que existan, y el trabajo asistido por IA se declara en el informe técnico.

## Preguntas abiertas para el equipo
1. ¿Se adopta la regla 1-SE o se mantiene «mayor media y desempate»? (recomendado: 1-SE)
2. ¿Quién revisa los PR y quién hará la revisión manual de errores humanos (datos privados)?
3. Si LR resulta ganador, ¿se promueve ahora o se espera a tener mensajes humanos nuevos para una evaluación independiente?
4. ¿Se amplía la rejilla de NB (`alpha` menores) o se limita estrictamente a LR?

## Desviaciones y resultados de la ejecución

- **Rejilla de NB:** pasó de `{0,1; 0,3; 0,5; 1,0}` a `{0,01; 0,03; 0,1; 0,3; 0,5; 1,0}` (64 configuraciones, no 56). La primera ejecución dejó el mejor `alpha` (0,1) en el borde de la rejilla; con la ampliación pasó a ser un óptimo interior. La rejilla de LR no cambió.
- **Resultado de la puerta 1:** LR no supera al mejor NB por más de una desviación (0,0019 frente a 0,0033), así que la selección es NB (`alpha=0,1`, lematización, unigramas), no LR. La hipótesis inicial de que LR ganaría solo se cumple de forma marginal en la media.
- **Sensibilidad agrupada (A.6):** se agrupó por la columna `intent` de SID (en lugar de por trigramas de caracteres, que habría sido demasiado costoso); es un escenario más estricto que el CV original (F1 ≈ 0,40–0,46).
- **Etiqueta renombrada:** `ruido_ortografico` pasó a `vocabulario_desconocido`, porque mide términos ausentes del entrenamiento y marcaba frases bien escritas.
- **Pendiente:** revisión manual de errores (B.6), decisión de promoción (puerta 2) con mensajes humanos nuevos, generalizar `evaluate_human.evaluate()` (fija NB `alpha=0,5`), y backend configurable (`ARTIFACTS`). La prueba del backend que falla localmente por falta de los archivos humanos no se investigó ni se modificó.
- No se marcaron casillas de `REQUISITOS.md` ni se tocó `CONTRIBUCIONES.md` (según el plan, solo tras fusionar los PR).
