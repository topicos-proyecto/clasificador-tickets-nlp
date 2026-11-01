# Arquitectura inicial y flujo de aprendizaje

## Cinco capas obligatorias

```text
Dataset mixto (público + propio)
  └── EDA, definición y revisión de etiquetas
       └── NLP configurable
            ├── normalización + tokenización + stop words
            ├── stemming O lematización (ambos experimentados)
            └── TF-IDF (unigramas O unigramas+bigramas, comparados)
                 └── Naive Bayes / Regresión Logística
                      └── evaluación y selección
                           └── pipeline serializado
                                └── FastAPI (carga artefacto)
                                     └── React (predicción + pasos + métricas)
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
4. Ajustar hiperparámetros en validación cruzada del entrenamiento (sin tocar prueba).
5. Comparar Naive Bayes frente a Regresión Logística por F1-macro y métricas por clase.
6. Evaluar una sola vez la configuración elegida en prueba; registrar errores y reproducibilidad.

## API implementada
- `GET /health`: estado del modelo orientado a clientes y su SHA-256.
- `GET /metrics`: métricas reales de la comparación (incluida la mejor configuración por modelo) y, si existe y corresponde al modelo, la evaluación humana; no expone los textos de evaluación.
- `POST /predict`: recibe `{"text":"mensaje"}` y devuelve `label`, `probabilities`, `requires_clarification`, `reason` y `preprocessing` con las trazas del tokenizador del pipeline.

La API carga `artifacts/comparison_customers/pipeline.joblib` al primer uso y verifica su huella en solicitudes posteriores. Si faltan el modelo o las métricas, cambian o no coinciden, responde con estado HTTP 503. Las entradas vacías, sin caracteres legibles, no textuales o mayores de 3000 caracteres reciben 422. Los saludos simples y textos sin términos en el vocabulario solicitan aclaración. No hay umbral de confianza calibrado.

Los mensajes sin solicitud suficiente no deberían forzar una clasificación. Cualquier regla de rechazo, desambiguación o umbral es provisional y debe quedar probada y documentada.

## Dependencias
Python + scikit-learn + herramientas NLP de español (NLTK/spaCy), FastAPI y React. Versiones definitivas pendientes de fijar durante implementación. No se asumen prestaciones de latencia o exactitud sin medir.
