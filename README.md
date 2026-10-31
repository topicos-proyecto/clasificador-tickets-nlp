# Clasificador de Tickets — NLP y Machine Learning

**Proyecto Integrador · Tópicos y Programación Avanzada**  
**Estado registrado:** 30 de octubre de 2026 — Fase de datos/EDA completada parcialmente; pendiente de revisión.

> **Objetivo académico:** aprender, implementar y justificar el pipeline NLP obligatorio de la consigna. La clasificación de tickets es el caso de estudio. **Todavía no hay modelos entrenados, API ni frontend funcionando.**

## 1. Alcance del proyecto

- **Empresa simulada:** genérica.
- **Tarea:** clasificar una solicitud textual por el departamento de **atención inicial**, no diagnosticar la causa del problema.
- **Seis categorías:** Soporte TI, Finanzas, Ventas, Marketing, Atención al Cliente y Logística.
- **Dataset mixto:** [SID — Synthetic Intent Dataset](https://huggingface.co/datasets/luigicfilho/sid) filtrado a español + mensajes propios o redactados para el proyecto.
- **Modelos a comparar:** Multinomial Naive Bayes y Regresión Logística.
- **Tecnologías previstas:** Python, scikit-learn, herramientas NLP en español, FastAPI y React.
- **Fuera de alcance inicial:** chatbot avanzado, NER complejo, acciones empresariales reales, diagnóstico automático de fallos y clasificación multietiqueta.

## 2. ¿Qué se hizo hasta ahora?

- [x] Se definieron el problema, las seis áreas y el objetivo académico.
- [x] Se creó documentación de [alcance](docs/ALCANCE.md), [requisitos](docs/REQUISITOS.md), [arquitectura](docs/ARQUITECTURA.md) y [contribuciones](CONTRIBUCIONES.md).
- [x] Se seleccionó SID como fuente pública inicial y se diseñó un mapeo **preliminar** de 44 intenciones hacia seis departamentos.
- [x] Se creó y ejecutó **localmente** el script `src/data/prepare_sid.py` para descargar SID, filtrar español, generar CSV separados y `summary.json`.
- [x] Se creó y ejecutó localmente `notebooks/01_eda_sid.ipynb`; se visualizaron ejemplos de las clases. **No se completó una auditoría de todas las etiquetas.**
- [x] Se creó una guía para datos propios y un validador `src/data/validate_custom.py`.
- [x] Se prepararon **70 borradores de IA**: 30 Marketing y 8 para cada una de las otras cinco clases.
- [x] Se ejecutó el validador sobre el CSV local: **70 filas, 0 errores y 0 advertencias**, con `source=borrador_ia`.

**Importante:** la validación anterior comprueba formato y reglas básicas, **no verifica que las etiquetas sean semánticamente correctas**. Los 70 ejemplos no deben presentarse como mensajes humanos reales, ni incorporarse al entrenamiento sin revisión.

### Resultados de preparación de SID

| Departamento | Registros aceptados |
|---|---:|
| Soporte TI | 5.529 |
| Ventas | 3.227 |
| Atención al Cliente | 3.195 |
| Finanzas | 2.969 |
| Logística | 2.237 |
| Marketing | 343 |
| **Total** | **17.500** |

Por partición: `train` **16.114**, `validation` **208**, `validation_real` **471**, `test` **707**. Se excluyeron 20.416 filas con intenciones no mapeadas y 493 duplicados exactos. **Marketing solo representa el 1,96 % de los datos aceptados**. Sigue pendiente revisar paráfrasis casi duplicadas, calidad de textos e intención correcta por área.

Fuente de estas cifras: salida local `data/processed/sid/summary.json` compartida por el equipo. Más detalle en el [informe de resultados de la rama de datos](https://github.com/topicos-proyecto/clasificador-tickets-nlp/blob/feature/data-sid-inicial/docs/RESULTADOS_SID.md).

## 3. ¿En qué rama quedó el trabajo?

- **`main`:** documentación general y este registro del estado actual.
- **`feature/data-sid-inicial`:** script de preparación, mapeo, notebook EDA, guía de datos propios, borradores y validador.
- **[Pull Request #11](https://github.com/topicos-proyecto/clasificador-tickets-nlp/pull/11):** **abierto en borrador; todavía no integrado a `main`**. Requiere revisión por otro integrante, conforme a la consigna.

Los archivos descargados y generados en `data/raw/` y `data/processed/` **solo están en las máquinas donde se ejecutó el script** y quedan excluidos por `.gitignore` en la rama de datos. `data/raw/custom/messages.csv` también es local y actualmente contiene los 70 borradores de IA. Guardar una copia privada si se desea conservar su contenido; no subir accidentalmente datos crudos o sensibles.

## 4. Cómo retomar el trabajo

Desde la raíz del repositorio local:

```bash
git fetch origin
git switch feature/data-sid-inicial
git pull origin feature/data-sid-inicial
```

Si los CSV de SID no están presentes, generarlos:

```bash
python src/data/prepare_sid.py
```

Validar el CSV local de borradores:

```bash
python src/data/validate_custom.py data/raw/custom/messages.csv --compare-sid
```

Abrir en VS Code `notebooks/01_eda_sid.ipynb` con el kernel del entorno Python y ejecutar las celdas. Instalar `pandas`, `matplotlib` e `ipykernel` en ese entorno si faltan.

### Próximos pasos, en orden

1. **Revisar el PR #11** con otro integrante y fusionarlo a `main` cuando el equipo lo apruebe; no hace falta esperar a tener un dataset perfecto, pero las limitaciones deben quedar documentadas.
2. Revisar manualmente el mapeo de las 44 intenciones SID y una muestra mayor de mensajes, sin usar el conjunto `test` para tomar decisiones de diseño.
3. Revisar y, si corresponde, corregir los 70 borradores de IA sin cambiar falsamente su procedencia; crear mensajes humanos originales, sobre todo de Marketing.
4. Definir y congelar divisiones de entrenamiento, validación y prueba, evitando que textos iguales o muy similares aparezcan a ambos lados.
5. **Implementar y explicar el pipeline NLP exigido en el PDF**: normalización, tokenización, stop words, **stemming vs. lematización**, **unigramas vs. unigramas+bigramas** y TF-IDF. Documentar ejemplos antes/después.
6. Entrenar y comparar Naive Bayes y Regresión Logística; validación cruzada, ajuste de hiperparámetros, F1 por clase, matriz de confusión y análisis de errores. Serializar el pipeline completo.
7. Implementar API REST (predicción/métricas/salud) y frontend que muestre el proceso y los resultados; realizar pruebas y preparar la defensa.

## 5. Requisitos académicos y colaboración

La consigna exige **el mismo pipeline en entrenamiento e inferencia**, entrenamiento reproducible, informe técnico, notebook, API, frontend, pruebas, README, contribuciones verificables, pull requests revisados y demostración con modificación en vivo.

La organización [`topicos-proyecto`](https://github.com/topicos-proyecto) contempla **cuatro repositorios**, pero el PDF solicita **uno por grupo**. Esto **requiere validación con el docente**. El equipo debe poder reproducir e identificar una entrega final coherente, aunque los componentes se distribuyan.

**Fechas según la consigna:** presentación 3 de noviembre; entrega hasta las 23:59 del día anterior (2 de noviembre de 2026, para este ciclo). Etiquetar la versión evaluada como `entrega-final` conforme a lo que autorice el docente.

**Documento de referencia:** `Proyecto_Integrador.pdf` proporcionado al equipo.
