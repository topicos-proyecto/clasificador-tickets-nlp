# Plan: comparación con KNN, panel de métricas y flujo de tickets con panel administrativo

**Estado:** implementado el 2 de noviembre de 2026. Ver «Resultado de la ejecución» al final. **Fecha del plan:** 2 de noviembre de 2026.
**Repositorios afectados:** `clasificador-tickets-nlp` (modelos y reporte), `clasificador-backend` (API y tickets), `clasificador-frontend` (interfaz).
**Plazo:** según `REQUISITOS.md`, el último commit es **hoy, 2 de noviembre, a las 23:59**. Por eso el plan está ordenado por prioridad (sección 9): si no alcanza el tiempo, lo que evalúa el docente (KNN, reporte y métricas) queda terminado primero.

---

## 0. Objetivos y criterios de éxito

| # | Objetivo | Criterio de éxito verificable |
|---|---|---|
| O1 | Incluir **KNN** en la comparación junto a Naive Bayes (NB) y Regresión Logística (LR) | `compare.py` evalúa los tres modelos con el mismo CV, la misma semilla y las mismas particiones; `report.json` contiene sus resultados |
| O2 | Dejar un **reporte en `docs/`** que justifique la elección | `docs/COMPARACION_MODELOS.md` con cifras **generadas por script** a partir de `report.json`, no copiadas a mano |
| O3 | **Visualizar métricas** en el frontend para la defensa | Vista «Modelo» con: comparación de 3 modelos, matriz de confusión real vs. predicho, tabla por clase (precisión, recall, F1, soporte), exactitud, distribución de clases, tamaños de entrenamiento/prueba y longitudes de texto |
| O4 | **Simular un flujo real** de soporte | Página pública `/soporte` → ticket → clasificación → notificación al área completa → asignación a un empleado desde el panel |
| O5 | **Panel administrativo** | El administrador gestiona áreas, empleados, tickets (asignar, reasignar, cambiar estado) y ve las notificaciones de cada área |

**Restricciones que se mantienen** (ya documentadas en el proyecto):
- `test` está reservado: no se usa para elegir modelo ni hiperparámetros (ver sección 1.6 para su única evaluación final).
- Los conjuntos de evaluación humana no se usan para seleccionar.
- El pipeline NLP no se duplica en el backend: la API carga el `pipeline.joblib`.
- No se sobrescribe el artefacto que consume el backend sin una decisión explícita.
- El bot **no conversa** con el cliente: solo clasifica y notifica.

---

## Parte 1 — KNN y comparación de los tres modelos (`clasificador-tickets-nlp`)

### 1.1 Datos disponibles (cifras reales de `data/processed/model_customers/`)

| Partición | Registros | Marketing | Palabras (media / mediana / máx.) | Uso |
|---|---:|---:|---|---|
| `train` | 15.560 | 387 (2,5 %) | 5,91 / 6 / 14 | CV 3 particiones y ajuste final |
| `validation` | 194 | 4 | 7,06 / 7 / 11 | Evaluación complementaria |
| `validation_real` | 448 | 8 | 5,27 / 5 / 8 | Evaluación complementaria |
| `test` | 707 | 19 | 5,90 / 6 / 11 | **Reservado**: una única evaluación final |

Distribución de `train`: Soporte TI 4.894 · Atención al Cliente 2.828 · Ventas 2.818 · Finanzas 2.655 · Logística 1.978 · Marketing 387.

Dos hechos que el reporte debe explicar al docente:
1. **Desbalance:** un modelo que nunca predijera Marketing seguiría teniendo ~97,5 % de exactitud. Por eso la exactitud (*accuracy*) se muestra, pero el criterio principal es **F1 macro**, que pondera igual a las seis áreas.
2. **Textos muy cortos (≈6 palabras):** pocas palabras por mensaje → vectores TF-IDF muy dispersos. Esto afecta especialmente a KNN (sección 1.2).

### 1.2 Por qué KNN y qué se espera de él

KNN clasifica un mensaje según las etiquetas de los *k* mensajes de entrenamiento más parecidos. Sobre TF-IDF:
- **Métrica:** distancia coseno (`metric='cosine'`, `algorithm='brute'`). Es la habitual en texto; como TF-IDF ya normaliza con L2, el orden de vecinos equivale al de la distancia euclidiana.
- **Hipótesis a verificar (no se asume el resultado):** KNN puede rendir bien en SID porque hay muchas variantes casi idénticas entre particiones, lo que también **infla** su resultado en CV. En mensajes reales, distintos de las plantillas, debería degradarse más que NB/LR. El reporte debe distinguir ambas cosas.
- **Costo:** KNN no aprende parámetros; guarda las 15.560 filas y en cada predicción calcula distancias contra todas. Se medirá tamaño del artefacto y latencia de predicción (sección 1.4).
- **Probabilidades:** `predict_proba` de KNN es la fracción de vecinos por clase (con *k*=5 solo toma valores 0, 0,2, 0,4…). Son aún menos calibradas que las de NB/LR; la API debe advertirlo si KNN llegara a seleccionarse.

### 1.3 Rejilla de KNN

| Hiperparámetro | Valores | Justificación |
|---|---|---|
| `n_neighbors` (principal) | 1, 3, 5, 7, 11, 15, 25 | De memorización pura (*k*=1) a suavizado fuerte; impares para reducir empates |
| `weights` | `uniform`, `distance` | `distance` da más peso a vecinos cercanos; útil con clases desbalanceadas |
| `metric` | `cosine` (fija) | Estándar en texto; fijarla mantiene la rejilla acotada |

Total nuevo: 2 métodos × 2 n-gramas × 14 = **56** configuraciones de KNN, que se suman a las 64 actuales → **120 configuraciones**.

### 1.4 Cambios en `src/models/compare.py`

1. **Modelo y rejilla**
   - `MODELS = ('naive_bayes', 'logistic_regression', 'knn')`.
   - `PRIMARY['knn'] = 'n_neighbors'`.
   - `GRID['knn'] = [{'n_neighbors': k, 'weights': w} for k in (1, 3, 5, 7, 11, 15, 25) for w in ('uniform', 'distance')]`.
   - `classifier()` crea `KNeighborsClassifier(n_neighbors=k, weights=w, metric='cosine', algorithm='brute', n_jobs=1)`.
   - `configurations()` y `key()` incluyen `weights` (hoy usan `class_weight`); se generaliza a un campo `secondary` o se añade `weights` como campo propio. **Recomendado:** añadir `weights` explícito para no romper el contrato con el backend (que lee `class_weight`).

2. **Regla de simplicidad** (`ranking()`): se extiende sin cambiar la regla de una desviación estándar.
   - Orden de modelos por simplicidad: NB < LR < KNN (KNN no comprime el conocimiento en parámetros y es el más costoso en inferencia).
   - Regularización de KNN: **mayor *k* = más simple** (frontera más suave); `uniform` antes que `distance`.
   - La regla se fija **antes** de ver resultados y se escribe en `SELECTION_RULE`.

3. **Métricas completas por finalista** (nuevo, necesario para el frontend): para la mejor configuración de **cada** modelo (por media de CV), entrenar sobre `train` y evaluar en `validation` y `validation_real`, guardando:
   - `accuracy`, `f1_macro`, `f1_weighted`, `balanced_accuracy`.
   - `classification_report` (precisión, recall, F1, soporte por clase).
   - `confusion_matrix` con `confusion_matrix_labels`.
   - `fit_seconds`, `predict_ms_per_1000`, `artifact_bytes` (tamaño de `joblib.dump` en memoria).
   - Además, las **predicciones fuera de partición** del CV (`cross_val_predict` con los mismos folds) para una matriz de confusión de CV sobre las 15.560 filas: es mucho más estable que la de `validation` (solo 4 Marketing).
   - Campo nuevo `report['finalists'] = {'naive_bayes': {...}, 'logistic_regression': {...}, 'knn': {...}}`.

4. **Comparación pareada** generalizada: `paired_comparison` pasa de NB-vs-LR a todos los pares (NB–LR, NB–KNN, LR–KNN), con diferencia por partición, media y desviación. Se conserva la clave antigua para no romper lectores existentes.

5. **Sensibilidad agrupada** (`--grouped-cv`): incluir el finalista KNN. Es la prueba clave de la hipótesis de 1.2: con CV agrupado por intención SID (intenciones nunca vistas), se espera que KNN caiga más que NB/LR.

6. **Rendimiento del CV:** con KNN, el cálculo de vecinos por fold es lo más caro. Optimización opcional: calcular una vez por fold los 25 vecinos más cercanos y derivar todos los *k* y ambos `weights` de esa misma matriz. Solo si el tiempo total supera ~10 minutos.

7. **Ejecución** en un directorio nuevo (no se toca el que usa el backend):
   ```bash
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.models.compare \
     --data-dir data/processed/model_customers \
     --output-dir artifacts/comparison_customers_v3 --grouped-cv
   ```

### 1.5 Estadísticas del dataset para el frontend (nuevo script)

`src/data/dataset_stats.py` → `artifacts/comparison_customers_v3/dataset_stats.json`, con:
- Conteo por clase y por partición (`train`, `validation`, `validation_real`, `test`). Contar filas de `test` **no** usa sus textos para decisiones; se lee solo la columna `department` y se documenta así.
- Longitud de textos por clase y partición: media, mediana, mínimo, máximo, percentiles 25/75 (en palabras y en caracteres) e **histograma** de palabras (bins 1–2, 3–4, …, 15+).
- Tokens tras preprocesamiento (media por mensaje) con el método seleccionado.
- Tamaño de vocabulario TF-IDF del modelo seleccionado.
- Huellas SHA-256 de cada CSV, para que la API verifique coherencia.

No incluye textos de mensajes, solo agregados.

### 1.6 Evaluación final en `test` (una sola vez)

Tras fijar la selección con CV (y no antes):
1. Congelar `report.json` (registrar su SHA-256).
2. Script `src/models/evaluate_test.py` que evalúa en `test` **los tres finalistas** y guarda `test_report.json` con las mismas métricas de 1.4.3.
3. El resultado de `test` **no cambia la selección**; se reporta tal cual, aunque otro modelo gane en `test`. Esto se explica en el reporte como buena práctica.
4. El script se niega a ejecutarse si `test_report.json` ya existe (evita «mirar varias veces»).

> Decisión para el equipo: si prefieren mantener `test` reservado hasta después de la defensa, se omite este paso y el frontend muestra «Evaluación en prueba: pendiente».

### 1.7 Pruebas (`tests/test_compare.py`)
- El número de configuraciones se calcula desde `GRID` (120).
- `classifier('knn', 5, weights='distance')` devuelve un `KNeighborsClassifier` con coseno.
- `ranking()`: entre candidatos equivalentes, NB antes que KNN y *k*=25 antes que *k*=1.
- `report['finalists']` tiene los tres modelos y cada uno sus claves obligatorias.
- `compare.py` sigue leyendo solo `train.csv` y `validation.csv` (+ `validation_real.csv`); `evaluate_test.py` es el único que lee `test.csv`.
- `dataset_stats.py` no escribe textos en su salida.

### 1.8 Reporte `docs/COMPARACION_MODELOS.md`

Se **genera** con `python -m src.models.render_comparison_report artifacts/comparison_customers_v3` para que las cifras coincidan con `report.json` (evita errores de transcripción). Una sección final de interpretación se redacta a mano. Estructura:

1. **Resumen de la decisión** (5 líneas): modelo elegido, F1 macro CV ± desviación, por qué.
2. **Datos usados:** tabla de 1.1, desbalance, longitudes, qué partición se usa para qué.
3. **Metodología común:** mismo preprocesamiento por variante, TF-IDF ajustado dentro de cada fold, CV estratificado de 3 particiones, semilla 42, F1 macro como criterio y por qué no la exactitud.
4. **Los tres modelos explicados en lenguaje simple:** cómo decide cada uno, sus hiperparámetros y su costo.
5. **Resultados de CV:** mejor configuración de cada modelo, media ± desviación, diferencias pareadas, efecto de stemming/lematización y de n-gramas.
6. **Métricas por clase y matrices de confusión** de los tres finalistas (CV fuera de partición y `validation`).
7. **Sensibilidad agrupada por intención:** cuánto cae cada modelo con intenciones no vistas.
8. **Costo:** tiempo de entrenamiento, latencia y tamaño del artefacto.
9. **Decisión y justificación:** aplicación literal de la regla de una desviación estándar; si dos modelos empatan estadísticamente, por qué se prefiere el más simple.
10. **Evaluación final en `test`** (si se ejecutó) y evaluación humana exploratoria del modelo elegido.
11. **Limitaciones:** SID sintético, posibles variantes cercanas entre folds, pocos ejemplos de Marketing en validación, probabilidades no calibradas.

Además, actualizar `README.md`, `ARQUITECTURA.md` («Naive Bayes / Regresión Logística / KNN»), `ALCANCE.md` («Tres modelos») y `REQUISITOS.md`.

### 1.9 Promoción del modelo al backend
- El backend pasa a leer el directorio del modelo desde la variable de entorno `MODEL_DIR` (por defecto el actual `artifacts/comparison_customers`), en vez de una ruta fija.
- Promover = arrancar la API con `MODEL_DIR=artifacts/comparison_customers_v3`. No se copia ni sobrescribe nada.
- `/metrics` debe reconocer `KNeighborsClassifier` → `'knn'` y comparar `n_neighbors`.
- La evaluación humana existente está ligada al SHA del modelo anterior; para el modelo promovido se vuelve a ejecutar `evaluate_customer_human` **después** de la selección (como evaluación, nunca para elegir).

---

## Parte 2 — Backend: métricas ampliadas y sistema de tickets (`clasificador-backend`)

### 2.1 Organización del código
`api/main.py` ya concentra todo; con tickets crecerá mucho. Se divide en:

```text
api/
  main.py            # crea la app, CORS, incluye routers
  config.py          # MODEL_DIR, DB_PATH, ADMIN_PASSWORD, SECRET_KEY (variables de entorno)
  model.py           # carga del pipeline + verificación SHA (código actual movido)
  db.py              # conexión SQLite, creación de tablas, datos semilla
  auth.py            # login del administrador y dependencia require_admin
  schemas.py         # modelos Pydantic de entrada/salida
  routers/
    classifier.py    # /health, /predict, /metrics (existentes)
    metrics.py       # /metrics/models, /metrics/dataset
    public.py        # POST /public/tickets
    admin.py         # áreas, empleados, tickets, notificaciones, dashboard
  services/
    routing.py       # clasificar → decidir área → crear notificación
```

**Persistencia:** SQLite con el módulo estándar `sqlite3` (sin dependencias nuevas; el archivo `data/tickets.db` va en `.gitignore`). Es suficiente para una simulación local y se puede explicar en la defensa.

### 2.2 Modelo de datos

```text
areas(id, nombre, descripcion, etiqueta_modelo UNIQUE NULL, activa, creada_en)
empleados(id, nombre, email UNIQUE, area_id → areas, cargo, activo, creado_en)
tickets(id, codigo UNIQUE, cliente_nombre, cliente_email NULL, mensaje,
        etiqueta_predicha NULL, confianza NULL, probabilidades JSON,
        area_id → areas NULL, empleado_id → empleados NULL,
        estado, requiere_revision, modelo_sha256, creado_en, actualizado_en)
notificaciones(id, ticket_id → tickets, area_id → areas, titulo, cuerpo, leida, creada_en)
eventos_ticket(id, ticket_id, tipo, detalle JSON, actor, creado_en)   # bitácora
```

- **Estados del ticket:** `nuevo` → `asignado` → `en_progreso` → `resuelto` → `cerrado`; transición aparte a `revision_manual`. Las transiciones válidas se validan en el backend.
- **Áreas y etiquetas del modelo:** las seis etiquetas del modelo son fijas. Al iniciar se crean las seis áreas semilla, cada una con su `etiqueta_modelo`. El administrador puede **crear** áreas nuevas (p. ej. «Legal»), pero el modelo **no** puede predecirlas: solo reciben tickets por reasignación manual. Esto se muestra en la interfaz para no prometer algo que el modelo no hace.
- Un área con `etiqueta_modelo` **no se elimina**, solo se desactiva. Si está inactiva, sus tickets van a `revision_manual`.
- Empleados de ejemplo: 2–3 por área (datos semilla ficticios).
- `eventos_ticket` registra creación, clasificación, notificación, asignación, reasignación y cambios de estado: sirve para mostrar el flujo completo en la defensa.
- Las **reasignaciones manuales** de área se guardan como corrección (`etiqueta_predicha` ≠ etiqueta del área final). El dashboard muestra la tasa de corrección: es una métrica realista del sistema, aunque no se use para reentrenar en esta versión.

### 2.3 Flujo de enrutamiento (`services/routing.py`)

```text
Cliente escribe en /soporte
  └─ POST /public/tickets {nombre, email?, mensaje}
       ├─ validar (mismas reglas que /predict: vacío, ilegible, > 3000 caracteres → 422)
       ├─ crear ticket (estado: nuevo, código p. ej. TK-2026-000123)
       ├─ clasificar con el pipeline cargado
       │    ├─ saludo / sin términos conocidos → requiere_revision, estado revision_manual
       │    └─ etiqueta → área con esa etiqueta_modelo
       │          ├─ área activa → asignar área
       │          └─ área inactiva → revision_manual
       ├─ notificar al ÁREA COMPLETA:
       │    título: «Nuevo ticket TK-2026-000123 · Finanzas»
       │    cuerpo: «Cliente: María López — Mensaje: “No me llegó la factura de octubre…”»
       ├─ registrar eventos en la bitácora
       └─ responder al cliente: {codigo, mensaje: "Su consulta será atendida a la brevedad."}
```

- **Al cliente no se le revela el área, la probabilidad ni el resultado del modelo**; solo la confirmación y el código de seguimiento. El bot no responde preguntas.
- La respuesta al cliente es la misma aunque el ticket quede en revisión manual (no se le pide aclaración, porque el bot no conversa; la aclaración la gestiona un humano).
- **Umbral de confianza:** no hay uno calibrado. Opción configurable `UMBRAL_REVISION` (desactivada por defecto); si se activa, documentar que es provisional, como exige `ARQUITECTURA.md`.
- **Notificación «al área completa»:** se guarda **una** notificación por área (bandeja compartida que ven todos sus empleados). Canal simulado dentro del sistema; queda un punto de extensión (`notificar_externo`) para correo o webhook, que no se implementa porque `ALCANCE.md` excluye integraciones reales.
- Protección básica del endpoint público: límite de longitud ya existente, límite simple por IP en memoria (p. ej. 10 tickets/minuto) y campo `nombre` obligatorio (1–120 caracteres).

### 2.4 Endpoints

**Públicos**
| Método | Ruta | Descripción |
|---|---|---|
| POST | `/public/tickets` | Crea ticket; responde `{codigo, mensaje}` |
| GET | `/health` | Existente |

**Clasificador y métricas** (protegidos por administrador, excepto `/health`)
| Método | Ruta | Descripción |
|---|---|---|
| POST | `/predict` | Existente (laboratorio con trazas NLP) |
| GET | `/metrics` | Existente (modelo seleccionado); se amplía con `accuracy` |
| GET | `/metrics/models` | `report['finalists']` + comparación pareada + regla de selección + mejor configuración por modelo y por variante (método × n-gramas) |
| GET | `/metrics/dataset` | `dataset_stats.json` (verificando huellas) |
| GET | `/metrics/test` | `test_report.json` si existe; si no, `{"estado": "pendiente"}` |

**Administración** (requieren token de administrador)
| Método | Ruta | Descripción |
|---|---|---|
| POST | `/admin/login` | `{password}` → token firmado (HMAC con `SECRET_KEY`, expira en 8 h) |
| GET | `/admin/dashboard` | Conteos por estado y por área, tickets en revisión, tasa de corrección, últimos tickets |
| GET/POST | `/admin/areas` | Listar / crear |
| PATCH | `/admin/areas/{id}` | Editar nombre, descripción, activar/desactivar |
| DELETE | `/admin/areas/{id}` | Solo áreas sin etiqueta del modelo y sin tickets abiertos (409 en otro caso) |
| GET/POST | `/admin/empleados` | Listar (filtro por área) / crear |
| PATCH/DELETE | `/admin/empleados/{id}` | Editar / desactivar (no se borra si tiene tickets) |
| GET | `/admin/tickets` | Filtros: estado, área, empleado, `requiere_revision`, texto; paginado |
| GET | `/admin/tickets/{id}` | Detalle + probabilidades + bitácora |
| PATCH | `/admin/tickets/{id}` | Asignar empleado (debe pertenecer al área), reasignar área, cambiar estado |
| GET | `/admin/notificaciones` | Por área; marcar como leída con `PATCH /admin/notificaciones/{id}` |

**Autenticación:** un único rol administrador, contraseña en la variable `ADMIN_PASSWORD` (sin valor por defecto en producción; en desarrollo el README indica cómo definirla). Token HMAC con la librería estándar (`hmac`, `hashlib`), sin dependencias nuevas. Es deliberadamente simple y se documenta como tal: no hay usuarios por empleado en esta versión.

**CORS:** añadir `PATCH` y `DELETE` y la cabecera `Authorization`.

### 2.5 Pruebas del backend (`tests/`)
- Base SQLite temporal por prueba (`tmp_path`).
- `POST /public/tickets`: crea ticket, notificación para el área predicha y responde solo `codigo` y `mensaje` (sin etiqueta).
- Mensaje «Hola» → `revision_manual` y sin notificación de área.
- Área inactiva → `revision_manual`.
- Endpoints `/admin/*` sin token → 401.
- Asignar un empleado de otra área → 422; transición de estado inválida → 409.
- Eliminar área con etiqueta del modelo → 409.
- `/metrics/models` devuelve los tres finalistas; 503 si el informe no corresponde al modelo.
- Las pruebas existentes de `/predict`, `/metrics` y `/health` siguen pasando.

---

## Parte 3 — Frontend (`clasificador-frontend`)

### 3.1 Estructura
Hoy todo está en `src/main.jsx` (117 líneas). Se reorganiza:

```text
src/
  main.jsx                 # router
  api.js                   # request() actual + token de administrador
  components/              # Icon, MetricCard, Tokens, Table, Badge, Modal, EmptyState
  components/charts/       # ConfusionMatrix, BarChart, Histogram, BoxStat (SVG/CSS propios)
  pages/public/Soporte.jsx
  pages/admin/Login.jsx
  pages/admin/Layout.jsx   # barra lateral actual reutilizada
  pages/admin/Dashboard.jsx
  pages/admin/Tickets.jsx  (+ TicketDetail.jsx)
  pages/admin/Areas.jsx
  pages/admin/Empleados.jsx
  pages/admin/Notificaciones.jsx
  pages/admin/Modelo.jsx   # métricas y decisión
  pages/admin/Laboratorio.jsx  # el clasificador actual con pasos NLP
```

- **Router:** `react-router-dom` (versión exacta fijada en `package.json`). Alternativa sin dependencias: enrutador mínimo con `window.location.pathname`; se recomienda la librería por claridad.
- **Gráficos:** SVG y CSS propios, sin librería. Son pocos tipos (barras, histograma, mapa de calor) y así se controlan colores, accesibilidad y modo oscuro. Si falta tiempo, `recharts` con versión fijada.
- `vite.config.js`: el proxy `/api` ya existe; se mantiene.

### 3.2 Página pública `/soporte`
Sin barra lateral ni datos del modelo. Formulario simple:
- Nombre completo (obligatorio), correo (opcional), mensaje (obligatorio, contador 0/3000).
- Botón «Enviar consulta». Al enviar: estado de carga, después tarjeta de confirmación:
  > **Su consulta será atendida.** Nuestro equipo la revisará a la brevedad. Código de seguimiento: **TK-2026-000123**.
- Botón «Enviar otra consulta». Errores de validación en lenguaje claro.
- No muestra área, probabilidades ni pasos NLP.

### 3.3 Panel administrativo `/admin/*`

**Login** (`/admin/login`): contraseña → token en `sessionStorage`; las rutas `/admin/*` redirigen al login si no hay token o la API responde 401.

**Dashboard** (`/admin`):
- Tarjetas: tickets nuevos, en revisión manual, en progreso, resueltos hoy.
- Barras de tickets por área; tasa de corrección manual del modelo.
- Lista de últimos tickets con área, estado y empleado.

**Tickets** (`/admin/tickets`):
- Tabla con filtros (estado, área, empleado, «requiere revisión», búsqueda).
- Detalle en panel lateral: datos del cliente, mensaje, área sugerida con confianza y probabilidades, bitácora, y acciones:
  - **Asignar a empleado** (desplegable con empleados activos del área).
  - **Reasignar área** (marca la corrección).
  - **Cambiar estado** (solo transiciones válidas).
- Cola destacada de «Revisión manual».

**Áreas** (`/admin/areas`): tabla y formulario crear/editar; insignia «Asignable por el modelo» en las seis áreas con etiqueta; activar/desactivar; eliminar solo áreas propias sin tickets abiertos. Aviso: «Las áreas nuevas solo reciben tickets por reasignación manual; el modelo conoce seis áreas».

**Empleados** (`/admin/empleados`): tabla filtrable por área; crear/editar/desactivar; número de tickets abiertos por empleado (carga de trabajo).

**Notificaciones** (`/admin/notificaciones`): bandeja por área (pestañas Soporte TI, Finanzas, …) con lo que «recibe» el área: código, nombre del cliente y mensaje; marcar como leída; contador de no leídas en la barra lateral. Es la forma visible de demostrar que el bot notificó al área completa.

**Laboratorio** (`/admin/laboratorio`): el clasificador actual con sus pasos de normalización, tokens, stop words y raíces/lemas. Se conserva tal cual porque cubre requisitos de la consigna.

### 3.4 Vista «Modelo» (`/admin/modelo`): métricas para la defensa

Pensada para recorrerse de arriba abajo explicando la decisión. Selector de modelo (NB / LR / KNN) que cambia las secciones 3–4; el seleccionado aparece marcado como «En producción».

1. **Decisión en una frase:** «Se eligió X porque…», generada desde `selection_rule` y los números (sin texto fijo que pueda quedar desactualizado).
2. **Comparación de los tres modelos:**
   - Tabla: modelo, mejor configuración, F1 macro CV (media ± desviación), exactitud, F1 en `validation`, latencia, tamaño del artefacto.
   - Barras de F1 macro con barra de error (± desviación) y la línea del umbral «mejor − 1 desviación»: se ve qué modelos quedan «empatados» y por qué gana el más simple.
   - Mini tabla de variantes: stemming vs. lematización × unigramas vs. bigramas, por modelo.
   - Sensibilidad agrupada por intención (si existe), para mostrar la caída de KNN.
3. **Matriz de confusión real vs. predicho:**
   - Mapa de calor 6×6, filas = real, columnas = predicho; diagonal resaltada; conmutador «conteos / % por fila».
   - Fuente seleccionable: CV fuera de partición (recomendada, 15.560 filas), `validation`, `validation_real` y, si existe, `test`.
   - Al pasar el cursor: «12 mensajes de Marketing fueron clasificados como Ventas».
4. **Métricas por clase:** tabla con **Área, Precisión, Recall, F1, Soporte**, más filas de *macro avg*, *weighted avg* y exactitud. Celdas con barra de fondo proporcional; resaltar la peor clase. Comparación lado a lado de los tres modelos en F1 por clase.
5. **Datos:**
   - Tarjetas: **entrenamiento 15.560 · validación 194 · validación adicional 448 · prueba 707**, con barra apilada de proporciones.
   - Distribución de clases por partición (barras agrupadas o apiladas) con el porcentaje de Marketing destacado.
   - **Longitud de los textos:** histograma de palabras por mensaje, tabla por clase (media, mediana, mín., máx., P25, P75) y comparación entre particiones.
6. **Notas de interpretación:** por qué F1 macro y no exactitud; datos sintéticos; probabilidades no calibradas; `test` evaluado una sola vez (o pendiente).

Accesibilidad: cada gráfico tiene título, descripción y una tabla equivalente (o `aria-label` con los valores); colores que funcionen en modo claro y oscuro.

---

## 4. Contratos de datos clave

`GET /metrics/models` (forma resumida):
```json
{
  "model_sha256": "…",
  "selection_rule": "…",
  "selected": {"model": "naive_bayes", "method": "lemmatization", "ngrams": 1,
               "parameter_name": "alpha", "parameter_value": 0.1,
               "f1_macro_mean": 0.0, "f1_macro_std": 0.0},
  "threshold": 0.0,
  "finalists": {
    "knn": {
      "config": {"n_neighbors": 0, "weights": "distance", "method": "…", "ngrams": 1},
      "cv": {"f1_macro_mean": 0.0, "f1_macro_std": 0.0, "fold_f1_macro": [],
             "oof": {"accuracy": 0.0, "classification_report": {}, "confusion_matrix": [[]]}},
      "validation": {"accuracy": 0.0, "f1_macro": 0.0, "classification_report": {}, "confusion_matrix": [[]]},
      "validation_real": {"…": "…"},
      "cost": {"fit_seconds": 0.0, "predict_ms_per_1000": 0.0, "artifact_bytes": 0}
    }
  },
  "variants": [{"model": "knn", "method": "stemming", "ngrams": 2, "f1_macro_mean": 0.0}],
  "paired": [{"a": "naive_bayes", "b": "knn", "mean_difference": 0.0, "std_difference": 0.0}],
  "grouped_sensitivity": {},
  "labels": ["Atención al Cliente", "Finanzas", "Logística", "Marketing", "Soporte TI", "Ventas"]
}
```
(Los ceros son marcadores de tipo, no resultados.)

---

## 5. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| KNN parece excelente en CV por las variantes casi idénticas de SID | Mostrar CV agrupado por intención y evaluación humana; explicarlo en el reporte |
| Más configuraciones → selección optimista | Regla de una desviación estándar fijada de antemano; diferencias pareadas |
| Promover un modelo rompe `/metrics` (verificación por SHA) | `MODEL_DIR` configurable; pruebas con el informe nuevo; el modelo anterior queda intacto |
| Tiempo de CV con KNN | Medir primero con un método; optimización de vecinos compartidos (1.4.6) |
| Cifras del reporte desactualizadas | Reporte generado desde `report.json`; el frontend lee de la API |
| Exponer datos de evaluación | `/metrics/*` solo devuelve agregados; nunca textos |
| Endpoint público abusado | Validación, límite por IP, sin exponer resultados del modelo |
| Plazo de hoy 23:59 | Prioridades de la sección 9 |

---

## 6. Ramas y PR

| Repo | Rama | Contenido |
|---|---|---|
| nlp | `feature/knn-comparacion` | 1.3–1.7, ejecución v3 |
| nlp | `docs/comparacion-modelos` | 1.8 y actualizaciones de docs |
| backend | `feature/metricas-modelos` | `MODEL_DIR`, `/metrics/models`, `/metrics/dataset`, `/metrics/test`, soporte KNN |
| backend | `feature/tickets-admin` | Parte 2 restante |
| frontend | `feature/vista-modelo` | Router, layout, vista «Modelo» |
| frontend | `feature/soporte-y-panel` | `/soporte` y panel administrativo |

Un PR por rama, revisado por otro integrante (requisito de la consigna). Cada integrante debe poder explicar KNN y la decisión.

---

## 7. Comandos de verificación

```bash
# NLP
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.models.compare \
  --data-dir data/processed/model_customers --output-dir artifacts/comparison_customers_v3 --grouped-cv
python -m src.data.dataset_stats --data-dir data/processed/model_customers --output-dir artifacts/comparison_customers_v3
python -m src.models.render_comparison_report artifacts/comparison_customers_v3   # → docs/COMPARACION_MODELOS.md

# Backend
cd ../clasificador-backend
MODEL_DIR=artifacts/comparison_customers_v3 ADMIN_PASSWORD=… SECRET_KEY=… \
  PYTHONPATH=../clasificador-tickets-nlp ../clasificador-tickets-nlp/.venv312/bin/python -m uvicorn api.main:app --port 8000
PYTHONPATH=../clasificador-tickets-nlp ../clasificador-tickets-nlp/.venv312/bin/python -m pytest -q

# Frontend
cd ../clasificador-frontend && npm install && npm run build && npm run dev
```

**Prueba de punta a punta (guion de la defensa):**
1. Abrir `/soporte`, escribir como «María López»: «No me llegó la factura de octubre» → ver «Su consulta será atendida» y el código.
2. Entrar a `/admin`, ver el ticket nuevo en el dashboard.
3. En Notificaciones → Finanzas, ver «Cliente: María López — Mensaje: …».
4. En Tickets, asignarlo a un empleado de Finanzas y pasarlo a «En progreso».
5. Enviar «Hola» desde `/soporte` → aparece en Revisión manual.
6. Abrir «Modelo» y explicar: comparación de los tres modelos, matriz de confusión, métricas por clase, datos y longitudes, y la regla de decisión.

---

## 8. Fuera de alcance (se dice explícitamente)
- Conversación del bot con el cliente o respuestas automáticas.
- Correo, WhatsApp o Slack reales (solo notificaciones simuladas dentro del sistema).
- Cuentas y permisos por empleado; solo existe el rol administrador.
- Reentrenamiento automático con las correcciones manuales.
- Despliegue en la nube.

---

## 9. Orden de ejecución y prioridades

| Prioridad | Tarea | Repo | Estimación |
|---|---|---|---|
| **P0** | KNN en `compare.py`, finalistas con métricas completas, pruebas | nlp | 1,5 h |
| **P0** | Ejecutar v3 + `dataset_stats.py` | nlp | 0,5 h (cómputo) |
| **P0** | Generador y `docs/COMPARACION_MODELOS.md` | nlp | 1 h |
| **P0** | `MODEL_DIR`, `/metrics/models`, `/metrics/dataset`, soporte KNN en `/metrics` | backend | 1 h |
| **P0** | Router + vista «Modelo» (comparación, matriz, tabla por clase, datos, longitudes) | frontend | 2,5 h |
| P1 | SQLite, áreas/empleados semilla, `POST /public/tickets`, enrutamiento y notificaciones | backend | 1,5 h |
| P1 | Página `/soporte` | frontend | 0,5 h |
| P1 | Login, dashboard, tickets (asignar, reasignar, estado), notificaciones | backend + frontend | 2,5 h |
| P2 | CRUD de áreas y empleados en la interfaz | frontend | 1 h |
| P2 | Evaluación única en `test` (si el equipo la aprueba) | nlp | 0,5 h |
| P3 | Límite por IP, umbral configurable, pulido visual, README de los tres repos | todos | 1 h |

Dependencias: P0-nlp → P0-backend → P0-frontend. La Parte 2 (tickets) puede avanzar en paralelo a la vista «Modelo» porque solo depende de `/predict`, que ya existe.

## 10. Decisiones pendientes del equipo
1. ¿Se evalúa `test` una vez antes de la defensa (1.6) o se deja pendiente?
2. ¿Se promueve al backend el modelo que resulte de v3, o se mantiene `comparison_customers` y solo se muestran las métricas comparativas?
3. ¿`react-router-dom` como dependencia nueva, o enrutador mínimo propio?
4. ¿El correo del cliente es obligatorio en `/soporte`? (El plan lo deja opcional.)

---

## Resultado de la ejecución (2 de noviembre de 2026)

- **Parte 1:** KNN integrado en `compare.py` (120 configuraciones). Comparación v3 ejecutada con `--grouped-cv`. Se generaron `dataset_stats.json`, la evaluación única en `test` y el reporte `docs/COMPARACION_MODELOS.md`.
  - Seleccionado: Naive Bayes, lematización, unigramas, `alpha=0,1`, con F1 CV 0,9689. Coincide con la rejilla v2.
  - Regresión Logística obtiene la mejor media (0,9715), dentro de una desviación estándar.
  - KNN queda fuera del umbral (0,9335).
- **Decisiones de la sección 10 tomadas al ejecutar:**
  1. `test` se evaluó una vez, después de congelar `report.json`.
  2. El backend usa v3 por defecto (`MODEL_DIR`). El modelo anterior sigue intacto.
  3. Se usó `react-router-dom` 7.6.0, fijado.
  4. El correo del cliente es opcional.
- **Desviaciones:**
  - Las tres partes se implementaron en una rama por repositorio, en lugar de seis ramas: `feature/knn-comparacion`, `feature/tickets-admin` y `feature/soporte-y-panel`.
  - Se agregó la columna `corregido` y `motivo_revision` a `tickets`.
  - La evaluación humana del modelo v3 no se ejecutó: sus archivos privados no están en este equipo.
  - El frontend mantiene el tema claro del diseño existente; no se añadió modo oscuro.
  - El costo medido incluye la tokenización. La lematización con spaCy hace que el pipeline de NB sea unas 20 veces más lento que el de LR, que usa stemming.
