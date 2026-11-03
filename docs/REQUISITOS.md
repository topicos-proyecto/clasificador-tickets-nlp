# Requisitos verificables de la consigna

Lista basada en **Proyecto Integrador – Primer Parcial**, Tópicos y Programación Avanzada. Cada elemento deberá demostrarse en la entrega.

## Datos
- [ ] Documentar origen, licencia y fecha de los datasets público y propio.
- [ ] Analizar distribución de clases, longitud de textos, palabras frecuentes y faltantes.
- [ ] Definir seis etiquetas y guía de anotación consistente.
- [ ] Dividir entrenamiento/prueba de modo estratificado, con semilla fija, sin usar prueba para decisiones de entrenamiento.
- [ ] Analizar desbalance y justificar tratamiento, si aplica.
- [ ] Evitar filtración entre train/test, incluidos duplicados y transformaciones ajustadas con datos de prueba.

## Pipeline de NLP (mostrar antes/después y justificar cada paso)
- [ ] Normalización: minúsculas, tildes, puntuación, números, URLs y emojis; documentar qué se conserva.
- [ ] Tokenización: casos difíciles del dominio.
- [ ] Stop words en español: justificar modificaciones, incluidas negaciones relevantes.
- [ ] Implementar **stemming y lematización**; medir y comparar para elegir.
- [ ] Comparar **unigramas vs. unigramas + bigramas**.
- [ ] Vectorizar con **TF-IDF** documentando tamaño de vocabulario, min_df y max_df.
- [ ] Garantizar exactamente el mismo procesamiento en entrenamiento e inferencia.

## Modelos y análisis
- [ ] Entrenar Multinomial Naive Bayes.
- [ ] Entrenar Regresión Logística.
- [x] Entrenar KNN como tercer modelo de comparación (ver `COMPARACION_MODELOS.md`).
- [ ] Comparar los modelos y justificar la selección para uso en la API.
- [ ] Ejecutar ajuste de hiperparámetros con validación cruzada en al menos uno.
- [ ] Reportar métricas de clasificación por clase (incluyendo F1), matriz de confusión y comparación.
- [ ] Analizar ejemplos de errores y posibles motivos.
- [ ] Garantizar que cada integrante pueda explicar internamente el modelo.

## Reproducibilidad e integración
- [ ] Un comando ejecuta entrenamiento desde datos crudos hasta artefactos.
- [ ] Serializar vectorizador y modelos junto al procesamiento seleccionado.
- [ ] Fijar versiones de dependencias y semilla.
- [ ] API REST: predicción con clase/probabilidad, métricas y health.
- [ ] Validar entradas vacías, excesivamente largas e inválidas, con errores claros.
- [ ] Documentar API y crear pruebas automatizadas.
- [ ] Frontend: ingresar texto, mostrar clase/probabilidad.
- [ ] Frontend: mostrar normalización, tokens, stop words retiradas, raíces o lemas.
- [ ] Frontend: panel de métricas y comparación de modelos.
- [ ] README completo para instalar y levantar el sistema desde cero.

## Colaboración y entrega
- [x] **Organización de repositorios aprobada:** el equipo confirmó que el docente permite varios repositorios. Se documentan los tres componentes actuales de `topicos-proyecto` en [ARQUITECTURA.md](ARQUITECTURA.md).
- [ ] Asegurar que el docente tenga acceso a todos los repositorios relevantes desde el inicio.
- [ ] Cuenta individual por integrante y configuración de identidad Git.
- [ ] Ramas por funcionalidad y PR revisado por otro integrante.
- [ ] Commits frecuentes, descriptivos y auténticos; no reescribir historia de main.
- [ ] Cada integrante realiza al menos un commit significativo en NLP o ML.
- [x] Archivo [CONTRIBUCIONES.md](../CONTRIBUCIONES.md) con autores reales y enlaces a commits y PR de los repositorios utilizados.
- [ ] Código fuente, artefactos o script regenerador, notebooks, informe en docs/, presentación.
- [ ] Publicar `entrega-final` sobre los commits finales coordinados de `main`, una vez revisado e integrado el cierre documental; procedimiento en [ENTREGA.md](ENTREGA.md).
- [ ] Poder reproducir desde las versiones finales coordinadas el sistema completo durante la defensa, explicar aportes y modificar código en vivo.
- [x] Declarar el uso de asistentes de IA en [INFORME_TECNICO.md](INFORME_TECNICO.md#asistencia-de-ia).

Informe técnico y diapositivas disponibles en [INFORME_TECNICO.md](INFORME_TECNICO.md), [PRESENTACION.pdf](PRESENTACION.pdf) y [PRESENTACION.html](PRESENTACION.html). Las casillas restantes conservan su estado hasta que el equipo verifique cada requisito; esta actualización documental no certifica por sí sola toda la entrega.

**Fecha indicada en el PDF:** presentación 3 de noviembre; último commit del día anterior a las 23:59. Para 2026: 2 de noviembre, 23:59.

**Nota:** el PDF presenta el mínimo sugerido del dataset como marcador editable, no una cifra vinculante; no se fija un mínimo definitivo sin confirmación docente.
