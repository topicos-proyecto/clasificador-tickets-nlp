# Alcance del proyecto

## Finalidad
Proyecto académico para **aprender y demostrar el pipeline NLP** exigido en el Proyecto Integrador de Tópicos y Programación Avanzada. El caso de uso es clasificar mensajes de soporte de una **empresa genérica** por departamento de atención inicial.

El alcance actual comprende **solicitudes de clientes**. Las tareas internas de empleados, como gestionar campañas o pedir reportes publicitarios, quedan fuera de esta versión; se conservan las evaluaciones anteriores como antecedentes y no como medidas específicas de este alcance.

## Decisiones aprobadas
- **Seis categorías:** Soporte TI, Finanzas, Ventas, Marketing, Atención al Cliente y Logística.
- **Dataset mixto:** registros de un dataset público, tras verificar licencia, idioma y compatibilidad, y ejemplos propios etiquetados por el equipo.
- **Dos modelos:** Multinomial Naive Bayes y Regresión Logística; el ganador se elige después de evaluarlos.
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
- Experimentos comparativos de variantes y dos modelos; validación cruzada, métricas y análisis de errores.
- Pipeline reproducible serializado.
- API REST (predicción, métricas, estado) e interfaz React con explicación de procesamiento.
- Pruebas, documentación y Git colaborativo verificable.

## Excluido
- Clasificación de solicitudes internas de empleados.
- Integración real con WhatsApp, correo o herramientas empresariales.
- Diagnóstico de fallos, generación de reembolsos o ejecución de acciones externas.
- Clasificación automática de prioridad, múltiples etiquetas por ticket y NER avanzado.
- Chatbot conversacional complejo y deployment obligatorio en nube.

## Pendientes reales
- Elegir dataset(s) público(s) específico(s), registrar fuente, licencia y fecha; verificar cobertura de las seis etiquetas.
- Establecer protocolo de etiquetado, revisión de desacuerdos, anonimización y remoción de duplicados.
- Definir responsable y reparto del trabajo según integrantes efectivos.
- Validar resultados y parámetros reales antes de fijar umbrales o afirmar precisión.
