# Clasificador de Tickets NLP

**Proyecto Integrador — Tópicos y Programación Avanzada**

Aplicación web académica que clasifica mensajes de soporte en español según el **departamento de atención inicial**, usando un pipeline reproducible de procesamiento de lenguaje natural (NLP) y comparando **Multinomial Naive Bayes** con **Regresión Logística**.

> **Prioridad:** aprender, implementar, comparar y explicar las técnicas exigidas en la consigna. La aplicación web demuestra el pipeline; no busca reemplazar un sistema empresarial de mesa de ayuda.

## Alcance acordado

- **Organización:** empresa genérica.
- **Seis categorías:** Soporte TI, Finanzas, Ventas, Marketing, Atención al Cliente y Logística.
- **Dataset:** mixto (fuentes públicas compatibles + mensajes propios etiquetados); documentar origen, licencia y fecha. No se ha seleccionado todavía una fuente pública concreta.
- **Tarea:** clasificación multiclase de una solicitud principal; no diagnóstico de causas ni clasificación de prioridad.
- **Modelos a comparar:** Multinomial Naive Bayes y Regresión Logística.
- **Backend / frontend propuestos:** FastAPI y React.

## Pipeline académico obligatorio

1. Normalización: minúsculas, tildes, puntuación, números, URLs y emojis con decisiones justificadas.
2. Tokenización y casos difíciles del dominio.
3. Stop words en español, conservando las palabras que aporten significado.
4. **Implementar y comparar stemming vs. lematización.**
5. **Comparar unigramas vs. unigramas + bigramas.**
6. TF-IDF con parámetros documentados.
7. Entrenar, validar, comparar, analizar errores y seleccionar uno de los dos clasificadores.
8. Serializar el preprocesamiento, TF-IDF y modelo elegido como un pipeline reutilizable por la API.

En la interfaz se mostrará la clasificación con probabilidades y las transformaciones del texto **paso a paso**. Mensajes vacíos, saludos sin solicitud o textos insuficientes deben pedir aclaración, no forzar una categoría.

## Capas obligatorias

`data/` → `src/nlp/` → `src/models/` → `api/` → `frontend/`

El backend **carga el pipeline entrenado y no reimplementa el NLP**. Se entregan análisis exploratorio, experimentos, métricas, validación cruzada, pruebas, README y registro de contribuciones.

## Documentación de planificación

- [Alcance y límites](docs/ALCANCE.md)
- [Requisitos de la consigna](docs/REQUISITOS.md)
- [Arquitectura inicial](docs/ARQUITECTURA.md)
- [Contribuciones del equipo](CONTRIBUCIONES.md)

## Colaboración

Un único repositorio por grupo. Trabajar en ramas `feature/...`, abrir pull requests y solicitar revisión de otro integrante. Cada estudiante debe aportar al menos un cambio significativo en NLP o Machine Learning además de su rol principal. Nunca compartir cuentas de GitHub.

## Estado actual

**Inicio del proyecto:** documentación y planificación. No hay dataset público verificado, modelo entrenado, API ni métricas reales.

## Entrega

Según la consigna: presentación el **3 de noviembre** y último commit antes de las **23:59 del día anterior**; se planifica el **2 de noviembre de 2026**. Crear el tag `entrega-final` al finalizar.
