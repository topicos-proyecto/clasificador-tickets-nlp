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

## Estructura prevista
```text
data/raw/public/
data/raw/custom/
data/processed/
notebooks/
src/nlp/
src/models/
artifacts/
api/
frontend/
tests/
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

## Contrato propuesto de API (no implementado)
- `GET /health`: estado del servicio y versión del modelo.
- `GET /metrics`: métricas reales leídas de artefactos de evaluación.
- `POST /predict`: recibe `{"text":"mensaje"}` y devuelve `label`, `probabilities`, `requires_clarification` y trazas de preprocesamiento.

Los mensajes sin solicitud suficiente no deberían forzar una clasificación. Cualquier regla de rechazo, desambiguación o umbral es provisional y debe quedar probada y documentada.

## Dependencias
Python + scikit-learn + herramientas NLP de español (NLTK/spaCy), FastAPI y React. Versiones definitivas pendientes de fijar durante implementación. No se asumen prestaciones de latencia o exactitud sin medir.
