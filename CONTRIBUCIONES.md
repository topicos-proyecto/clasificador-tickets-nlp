# Registro de contribuciones

Registro basado en los autores reales de los commits y en los PR integrados de la organización `topicos-proyecto`. Los nombres se conservan como aparecen en Git; no se deducen nombres completos a partir de los usuarios. La organización en varios repositorios fue aprobada por el docente, según la confirmación del equipo.

| Nombre registrado en Git | Usuario de GitHub | Aportes y evidencia verificable | Aporte significativo NLP o ML |
|---|---|---|---|
| Alvaro | [alvaroSG34](https://github.com/alvaroSG34) | Preparación y validación de SID, documentación del dataset, notebook EDA y pipeline inicial: [PR #11](https://github.com/topicos-proyecto/clasificador-tickets-nlp/pull/11), [PR #12](https://github.com/topicos-proyecto/clasificador-tickets-nlp/pull/12), commits [284da77](https://github.com/topicos-proyecto/clasificador-tickets-nlp/commit/284da77) y [2d73a42](https://github.com/topicos-proyecto/clasificador-tickets-nlp/commit/2d73a42). Inicialización de [backend](https://github.com/topicos-proyecto/clasificador-backend/commit/f4adbc3) y [frontend](https://github.com/topicos-proyecto/clasificador-frontend/commit/bcbe2b1). | Preparación de datos en español, validación de ejemplos propios, particiones y pipeline NLP reproducible inicial. |
| AlexTiclla | [AlexTiclla](https://github.com/AlexTiclla) | Ampliación de hiperparámetros: [PR #13](https://github.com/topicos-proyecto/clasificador-tickets-nlp/pull/13), commit [c36fff6](https://github.com/topicos-proyecto/clasificador-tickets-nlp/commit/c36fff6). Análisis automático de errores: [PR #14](https://github.com/topicos-proyecto/clasificador-tickets-nlp/pull/14), commit [68a1c08](https://github.com/topicos-proyecto/clasificador-tickets-nlp/commit/68a1c08). Comparación NB/LR/KNN: [PR #15](https://github.com/topicos-proyecto/clasificador-tickets-nlp/pull/15), commit [8234274](https://github.com/topicos-proyecto/clasificador-tickets-nlp/commit/8234274). API y panel: [backend PR #1](https://github.com/topicos-proyecto/clasificador-backend/pull/1), [frontend PR #1](https://github.com/topicos-proyecto/clasificador-frontend/pull/1). | Modelado, ajuste y comparación de tres clasificadores, métricas, análisis de errores y justificación de selección. |
| juanse8202 | [juanse8202](https://github.com/juanse8202) | Clasificador de terminal con selección de uno o varios pipelines, probabilidades por área, trazas de preprocesamiento, tabla comparativa y pruebas: [PR #16](https://github.com/topicos-proyecto/clasificador-tickets-nlp/pull/16), commit [5d4c2f3](https://github.com/topicos-proyecto/clasificador-tickets-nlp/commit/5d4c2f3). | Inferencia y explicación del pipeline serializado, comparación interactiva de modelos y pruebas de consistencia entre probabilidades, etiquetas y trazas. |

## Criterio de atribución

Los aportes de código se atribuyen al autor de los commits, no a quien hizo el merge. En particular, los PR de modelado #13 y #14 contienen código de AlexTiclla y fueron integrados por Alvaro; esa integración no cambia la autoría. Los merges realizados por juanse8202 tampoco le atribuyen el código de los otros integrantes. Los commits de inicialización y el commit `prueba` del backend no se presentan como aportes sustanciales de NLP o ML.

Las responsabilidades se superponen: los roles no limitan qué parte puede explicar cada integrante. El registro del notebook acredita su creación; no afirma que sus celdas se hayan ejecutado. Los nuevos documentos de cierre deberán vincularse a su propio commit/PR después de publicarse.

## Repositorios y asistencia de IA

- [NLP, datos y modelos](https://github.com/topicos-proyecto/clasificador-tickets-nlp).
- [Backend FastAPI](https://github.com/topicos-proyecto/clasificador-backend).
- [Frontend React](https://github.com/topicos-proyecto/clasificador-frontend).

La declaración de asistencia de IA y sus límites está en [INFORME_TECNICO.md](docs/INFORME_TECNICO.md#asistencia-de-ia). La versión coordinada de los tres repositorios se registra siguiendo [ENTREGA.md](docs/ENTREGA.md).

## Reglas
- Cada estudiante utiliza su propia cuenta e identidad Git.
- Una rama por funcionalidad, PR revisado por otro integrante.
- Cada integrante debe contribuir significativamente a NLP o ML.
- Commits frecuentes con descripción precisa; trabajo en pareja documentado con `Co-authored-by:` cuando corresponda.
- No falsificar autoría ni reescribir la historia de `main`.
- Registrar decisiones técnicas, experimentos y evidencia para la defensa individual.
