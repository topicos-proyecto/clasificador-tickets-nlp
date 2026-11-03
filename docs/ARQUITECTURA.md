# Arquitectura inicial y flujo de aprendizaje

## Cinco capas obligatorias

```text
Dataset mixto (público + propio)
  └── EDA, definición y revisión de etiquetas
       └── NLP configurable
            ├── normalización + tokenización + stop words
            ├── stemming O lematización (ambos experimentados)
            └── TF-IDF (unigramas O unigramas+bigramas, comparados)
                 └── Naive Bayes / Regresión Logística / KNN
                      └── evaluación y selección
                           └── pipeline serializado
                                └── FastAPI (carga artefacto)
                                     └── React (/soporte público + panel /admin: tickets, áreas, métricas)
```

## Regla principal
El entrenamiento guarda un **pipeline completo de transformación y clasificación**; la API carga ese objeto y llama a `predict` / `predict_proba`. Prohibido duplicar la lógica de NLP manualmente en backend. La visualización didáctica puede exponer trazas de esas mismas transformaciones.

## Organización en GitHub

El workspace local separa `clasificador-tickets-nlp` (datos y modelos), `clasificador-backend` (API) y `clasificador-frontend` (interfaz). El backend importa el código NLP y carga sus artefactos desde `clasificador-tickets-nlp`, sin duplicar el pipeline. Las carpetas nuevas todavía no se inicializaron como repositorios Git.

La organización `topicos-proyecto` contempla **cuatro repositorios**. El reparto definitivo entre repositorios está pendiente de asignación; el diagrama siguiente representa **capas y rutas de código lógicas**, no una obligación de alojarlas todas en este repositorio. Se requiere una integración reproducible entre componentes y un pipeline NLP idéntico al de entrenamiento.

**Advertencia académica:** el PDF pide un solo repositorio por grupo; el uso de cuatro necesita validación del docente.

## Estructura lógica prevista
```text
data/raw/public/
data/raw/custom/
data/processed/
notebooks/
src/nlp/
src/models/
artifacts/
tests/
../clasificador-backend/api/
../clasificador-frontend/
docs/
README.md
CONTRIBUCIONES.md
```
Las carpetas se agregarán al comenzar su implementación; no se crean directorios vacíos porque Git no los rastrea.

## Experimentos
1. Separar datos de prueba antes de ajustar cualquier transformador.
2. Probar stemming frente a lematización manteniendo constante lo demás.
3. Comparar `ngram_range=(1,1)` frente a `(1,2)`.
4. Ajustar hiperparámetros en validación cruzada del entrenamiento (sin tocar prueba): `alpha` en Naive Bayes; `C` y `class_weight` en Regresión Logística; `k` y `weights` en KNN (coseno); barrido opcional de `min_df`/`max_df` (`--tfidf-grid`). La selección usa la regla de una desviación estándar. Detalle en [ANALISIS_ERRORES.md](ANALISIS_ERRORES.md).
5. Comparar Naive Bayes, Regresión Logística y KNN por F1-macro, métricas por clase, sensibilidad agrupada y costo.
6. Evaluar una sola vez la configuración elegida en prueba; registrar errores y reproducibilidad.

## API implementada

El backend (`clasificador-backend`) carga el pipeline de `MODEL_DIR` (por defecto `artifacts/comparison_customers_v3`) y verifica su huella SHA-256 en cada solicitud; si falta o cambió, responde 503.

- `GET /health`, `POST /public/tickets` (públicos).
- `POST /predict`, `GET /metrics`, `GET /metrics/models`, `GET /metrics/dataset`, `GET /metrics/test` (administrador).
- `/admin/*`: dashboard, áreas, empleados, tickets y notificaciones (administrador, token HMAC).

Flujo de un ticket:

1. El cliente envía su mensaje en `/soporte` y recibe solo «Su consulta será atendida» y un código.
2. El pipeline clasifica el mensaje.
3. Si hay un área activa, se crea una notificación para el área completa con el nombre del cliente y el mensaje.
4. Si no (saludo, sin términos conocidos, área inactiva o modelo no disponible), el ticket va a revisión manual.
5. El administrador asigna el ticket a un empleado del área y gestiona sus estados.

Persistencia: SQLite con el módulo estándar. Detalle en el README del backend.

Los mensajes sin solicitud suficiente no fuerzan una clasificación. Cualquier regla de rechazo o umbral (`UMBRAL_REVISION`) es provisional, porque las probabilidades no están calibradas.

## Dependencias
Python + scikit-learn + herramientas NLP de español (NLTK/spaCy), FastAPI y React. Versiones definitivas pendientes de fijar durante implementación. No se asumen prestaciones de latencia o exactitud sin medir.
