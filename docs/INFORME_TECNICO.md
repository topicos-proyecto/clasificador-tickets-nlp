# Informe técnico: clasificador de solicitudes de clientes

## Problema y alcance

El sistema asigna una solicitud escrita en español al departamento de atención inicial: Soporte TI, Finanzas, Ventas, Marketing, Atención al Cliente o Logística. Es una simulación académica para explicar NLP y clasificación de texto. El resultado orienta la atención; no diagnostica ni resuelve la incidencia.

La página pública recibe el mensaje y entrega un código de seguimiento. El panel administrativo muestra los tickets, las áreas, los empleados, las notificaciones, las métricas y un laboratorio para inspeccionar el preprocesamiento. Los integrantes y sus aportes verificables se registran en [CONTRIBUCIONES.md](../CONTRIBUCIONES.md).

## Datos y procedencia

Se utiliza SID — Synthetic Intent Dataset, filtrado a español y mapeado a seis áreas. Su revisión está fijada en `data/sid_source.json`, junto con las huellas esperadas. La fuente declara licencia MIT; la fecha de selección y las decisiones del mapeo están en [DATASET_SID.md](DATASET_SID.md).

El entrenamiento incorpora también 90 borradores revisados de Marketing y 30 borradores revisados orientados a clientes. Su procedencia sigue siendo `borrador_ia`; no se presentan como mensajes de clientes reales. El conjunto simulado adicional contiene 24 mensajes, cuatro por área. Los archivos humanos privados pertenecen a evaluaciones históricas separadas y no se publican en Git.

| Partición orientada a clientes | Registros | Uso |
|---|---:|---|
| Entrenamiento | 15.560 | Validación cruzada y ajuste de los finalistas |
| Validación | 194 | Evaluación complementaria |
| Validación adicional | 448 | Evaluación complementaria |
| Test | 707 | Evaluación final posterior a la selección |

Marketing representa aproximadamente el 2,5 % del entrenamiento y tiene solo cuatro ejemplos en validación. Se prioriza F1 macro para dar el mismo peso a cada clase. Las particiones usan semillas y manifiestos; una heurística de trigramas aparta posibles similitudes entre ellas. Esta medida no garantiza detectar todas las paráfrasis.

## Decisiones de NLP

El pipeline completo se serializa con joblib. El backend importa el código NLP y carga ese artefacto, por lo que entrenamiento e inferencia comparten las transformaciones.

| Etapa | Decisión |
|---|---|
| Normalización | Unicode NFKC y minúsculas; eliminación de URLs, puntuación y emojis; conservación de tildes y números. |
| Tokenización | Separación de las palabras del texto normalizado; trazas didácticas del mismo tokenizador serializado. |
| Stop words | Lista española de spaCy, conservando negaciones como `no`, `nunca` y `sin`. |
| Reducción de palabras | Comparación de stemming Snowball y lematización con `es_core_news_sm` 3.8.0. |
| N-gramas | Comparación de unigramas y unigramas + bigramas. |
| TF-IDF | Ajuste dentro de cada fold; configuración elegida con `min_df=1`, `max_df=1.0` y 9.092 términos de vocabulario. |

Ejemplo con «No puedo aplicar el cupón de descuento 😊»: la normalización produce `no puedo aplicar el cupón de descuento`; tras retirar stop words quedan `no`, `aplicar`, `cupón`, `descuento`. Stemming produce raíces como `aplic` y `descuent`; la lematización conserva `aplicar` y `descuento`.

## Modelos, selección y resultados

Se compararon 120 configuraciones de Naive Bayes multinomial, Regresión Logística y KNN: dos métodos de reducción, dos rangos de n-gramas y sus hiperparámetros. Se utilizó CV estratificada de tres particiones, semilla 42 y las mismas particiones para todos los modelos. TF-IDF se ajustó dentro del entrenamiento de cada fold.

| Modelo finalista | F1 macro CV | F1 macro validación | F1 macro test |
|---|---:|---:|---:|
| Naive Bayes, lematización, unigramas, `alpha=0.1` | 0,9689 | 0,9572 | 0,9586 |
| Regresión Logística, stemming, unigramas, `C=30`, `balanced` | 0,9715 | 0,9735 | 0,9597 |
| KNN coseno, stemming, unigramas, `k=5`, `distance` | 0,9335 | 0,8676 | 0,9101 |

Se eligió Naive Bayes por la regla fijada de una desviación estándar: entre configuraciones próximas a la mejor media se prioriza la sencillez. La diferencia con LR en CV es aproximadamente 0,0027, menor que la desviación de la mejor configuración, aproximadamente 0,0033. Este criterio de selección no constituye una prueba de equivalencia estadística. La elección se mantiene después de evaluar `test`; no se vuelve a ajustar usando esos resultados.

Las métricas por clase, matrices de confusión, variantes y costos están en [COMPARACION_MODELOS.md](COMPARACION_MODELOS.md). Su sección de interpretación explica la elección para la defensa. Los resultados v3 locales se respaldan con `report.json` y `test_report.json`, cuyas huellas vinculan datos y modelos.

## Análisis de errores y límites

El modelo v3 comete 10 errores de 194 mensajes de validación y 8 de 24 del conjunto simulado. El vocabulario no visto, los errores ortográficos y la ambigüedad entre áreas dificultan la clasificación. El análisis automático propone causas heurísticas; su revisión humana debe distinguir una explicación plausible de una causa demostrada. [ANALISIS_ERRORES.md](ANALISIS_ERRORES.md) conserva el análisis de la versión v2 como antecedente, no como informe del modelo v3.

SID es mayormente sintético. Un F1 alto sobre ese corpus no demuestra el mismo rendimiento con clientes reales. Las probabilidades no están calibradas y los mensajes con varias incidencias exceden la clasificación de una sola etiqueta. Un umbral arbitrario no garantiza detectar todos los errores; algunos errores pueden tener alta probabilidad.

## Arquitectura e interfaz

1. `clasificador-tickets-nlp` prepara datos, tokeniza, compara modelos y guarda el pipeline y los informes.
2. `clasificador-backend` carga el pipeline, valida entradas, ofrece predicción/métricas/health y persiste tickets en SQLite.
3. `clasificador-frontend` consume la API desde React: `/soporte` para clientes y `/admin` para el panel.

Los endpoints de predicción y métricas del panel requieren autenticación. La página pública confirma la recepción sin mostrar al cliente el área predicha. El laboratorio administrativo sí muestra clase, probabilidades y trazas. Los saludos, mensajes insuficientes y casos sin vocabulario reconocido requieren detalles o revisión manual.

El clasificador de terminal también permite repetir `--model` para comparar varios pipelines, mostrar probabilidades por área y examinar el preprocesamiento de cada uno. Su comparación no reentrena ni sustituye la evaluación con etiquetas.

La separación en varios repositorios fue aprobada por el docente, según la confirmación del equipo. [ARQUITECTURA.md](ARQUITECTURA.md) describe los componentes y [ENTREGA.md](ENTREGA.md) explica cómo coordinar sus versiones.

## Reproducción y evidencia

El [README](../README.md) contiene la instalación con Python 3.12, dependencias fijadas y modelo español de spaCy, los comandos de preparación, comparación, evaluación y ejecución. Los datos privados y artefactos generados están excluidos de Git y deben regenerarse o compartirse por un canal apropiado. No deben publicarse contraseñas ni datos de clientes.

Durante la verificación de la mejora de terminal, la suite NLP registró **67 pruebas aprobadas, sin fallos ni omisiones**, una vez disponibles los artefactos v3. Este resultado acredita las pruebas ejecutadas en ese entorno; no sustituye el ensayo de la demostración completa desde las versiones finales. Los comandos para volver a verificar cada componente figuran en sus respectivos README.

## Asistencia de IA

Se utilizó Codex como asistente para revisar el proyecto, implementar y verificar la mejora del clasificador de terminal, generar artefactos mediante los scripts existentes y preparar documentación y diapositivas. Se empleó para proponer cambios y explicar resultados; las métricas del informe proceden de ejecuciones de código y de sus artefactos, no de cifras inventadas por el asistente.

Los datos creados con asistencia de IA conservan la procedencia `borrador_ia`, diferenciada de `propio_humano`. El historial de Git mantiene los autores reales de los commits; no se atribuye a un integrante el código de otro por haber integrado un PR. Cada integrante debe revisar, explicar y poder modificar el código y las decisiones que presenta. La asistencia de IA no reemplaza la revisión real del PR por otro integrante.

## Material de defensa

Las diapositivas están disponibles en [PDF](PRESENTACION.pdf) y en [HTML interactivo](PRESENTACION.html). Funcionan sin conexión. En HTML se navega con las flechas, se activa pantalla completa con el botón correspondiente y se puede imprimir o guardar como PDF. Incluyen problema, datos, pipeline, comparación, decisión, demostración, limitaciones y contribuciones.
