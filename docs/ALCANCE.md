# Alcance del proyecto

## Finalidad
Proyecto académico para **aprender y demostrar el pipeline NLP** exigido en el Proyecto Integrador de Tópicos y Programación Avanzada. El caso de uso es clasificar mensajes de soporte de una **empresa genérica** por departamento de atención inicial.

El alcance actual comprende **solicitudes de clientes**. Las tareas internas de empleados, como gestionar campañas o pedir reportes publicitarios, quedan fuera de esta versión; se conservan las evaluaciones anteriores como antecedentes y no como medidas específicas de este alcance.

## Decisiones aprobadas
- **Seis categorías:** Soporte TI, Finanzas, Ventas, Marketing, Atención al Cliente y Logística.
- **Dataset mixto:** registros de un dataset público, tras verificar licencia, idioma y compatibilidad, y ejemplos propios etiquetados por el equipo.
- **Tres modelos:** Multinomial Naive Bayes, Regresión Logística y KNN; el ganador se elige con una regla fijada antes de evaluarlos ([Comparación de modelos](COMPARACION_MODELOS.md)).
- **Salida:** departamento y probabilidad estimada, junto con visualización del preprocesamiento; no se promete resolver el incidente.
- **Arquitectura:** datos, NLP, modelos, backend REST y frontend web.

## Definición operacional preliminar de categorías
| Categoría | Atiende inicialmente | Ejemplo |
|---|---|---|
| Soporte TI | Fallos de aplicaciones, accesos o sistemas | No puedo iniciar sesión |
| Finanzas | Cobros, reembolsos y facturas | Solicito una devolución |
| Ventas | Contrataciones, precios y cotizaciones | Quiero cotizar diez licencias |
| Marketing | Promociones y campañas | ¿Qué promociones tienen? |
| Atención al Cliente | Consultas generales y orientación | ¿Cuál es el horario de atención? |
| Logística | Entregas, pedidos y envíos | Mi paquete todavía no llegó |

Estos criterios son **convenciones de enrutamiento** y no diagnósticos de causa raíz. Un cobro duplicado no prueba automáticamente un fallo técnico.

## Casos especiales
- Saludos aislados (p. ej. «Hola»), mensajes vacíos o consultas sin información suficiente: pedir detalles; no crear ticket categorizado.
- Mensajes con dos incidencias independientes: caso de limitación de la clasificación de una sola etiqueta; solicitar aclaración o derivar a revisión humana.
- Confianza insuficiente o probabilidades cercanas: opción de aclaración sujeta a experimentación y calibración; ningún umbral arbitrario constituye garantía.

## Incluido
- EDA: distribución de clases, longitudes, términos frecuentes y faltantes.
- Normalización, tokenización, stop words, stemming, lematización, n-gramas y TF-IDF.
- Experimentos comparativos de variantes y tres modelos; validación cruzada, métricas y análisis de errores.
- Pipeline reproducible serializado.
- API REST (predicción, métricas, estado) e interfaz React con explicación de procesamiento.
- Simulación del flujo de soporte: página pública `/soporte`, enrutamiento del ticket al área predicha con notificación al área completa y panel administrativo (áreas, empleados, asignación y estados). El bot no conversa con el cliente.
- Pruebas, documentación y Git colaborativo verificable.

## Excluido
- Clasificación de solicitudes internas de empleados.
- Integración real con WhatsApp, correo o herramientas empresariales.
- Diagnóstico de fallos, generación de reembolsos o ejecución de acciones externas.
- Clasificación automática de prioridad, múltiples etiquetas por ticket y NER avanzado.
- Chatbot conversacional complejo y deployment obligatorio en nube.

## Estado y revisiones posteriores

La fuente, licencia y fecha de selección están en [DATASET_SID.md](DATASET_SID.md). El protocolo y la procedencia de ejemplos propios están en [GUIA_DATASET_PROPIO.md](GUIA_DATASET_PROPIO.md). Los aportes verificables están en [CONTRIBUCIONES.md](../CONTRIBUCIONES.md), y los resultados v3 en [COMPARACION_MODELOS.md](COMPARACION_MODELOS.md).

El docente aprobó usar varios repositorios, según la confirmación del equipo. El cierre documental, la presentación y las instrucciones de etiquetado están en [ENTREGA.md](ENTREGA.md).

Las revisiones de etiquetas ambiguas y causas heurísticas de errores siguen requiriendo criterio humano. Los umbrales de confianza y las afirmaciones sobre rendimiento con clientes reales necesitan datos independientes; no se consideran resueltos por completar la documentación.
