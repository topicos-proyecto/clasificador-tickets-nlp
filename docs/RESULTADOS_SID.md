# Primer informe de cobertura del dataset SID (español)

**Estado:** resultado de la ejecución local del script `src/data/prepare_sid.py`, compartido por el equipo. **No es una evaluación de modelos.**

## Resumen
- Mensajes SID originales de idioma español en los cuatro splits: **38.409**.
- Mensajes aceptados tras mapeo y eliminación de duplicados exactos: **17.500**.
- Filas con intenciones no mapeadas: **20.416** (excluidas deliberadamente).
- Filas duplicadas eliminadas: **493** (471 en `validation`, 22 en `train`).
- Intenciones originales incluidas en el mapeo: **44**; **requieren revisión humana**.

## Distribución por departamento (todos los splits)

| Departamento | Registros | Porcentaje |
|---|---:|---:|
| Soporte TI | 5.529 | 31,59 % |
| Ventas | 3.227 | 18,44 % |
| Atención al Cliente | 3.195 | 18,26 % |
| Finanzas | 2.969 | 16,97 % |
| Logística | 2.237 | 12,78 % |
| **Marketing** | **343** | **1,96 %** |
| **Total** | **17.500** | **100 %** |

Marketing es el principal cuello de botella: `train` contiene 308 mensajes de Marketing frente a 5.079 de Soporte TI (relación aproximada de 16,5:1). El problema no se resuelve garantizadamente con el clasificador, ni conviene copiar frases idénticas para inflar el conjunto.

## Conteos por split

| Split | Aceptados | Marketing | Longitud media (palabras) |
|---|---:|---:|---:|
| `train` | 16.114 | 308 | 5,88 |
| `validation` | 208 | 6 | 7,04 |
| `validation_real` | 471 | 10 | 5,31 |
| `test` | 707 | 19 | 5,90 |
| **Total** | **17.500** | **343** | — |

**Precaución sobre duplicados:** 471 filas fueron eliminadas del split `validation` por coincidir exactamente, tras normalizar espacios y mayúsculas/minúsculas, con filas de particiones procesadas antes (`test` y `validation_real`). No se ha establecido cuántos duplicados se relacionan con cada partición anterior ni si existen paráfrasis casi idénticas. El script prioriza `test`, después `validation_real`, después `validation` y finalmente `train`. Las decisiones finales de evaluación deben explicitar ese orden.

## Decisión de uso

1. **No entrenar todavía:** validar muestras aleatorias e intenciones mapeadas con una guía de etiquetas; revisar en especial el caso de Marketing y las etiquetas de reembolso, cupones y asuntos mixtos.
2. **Mantener `test` reservado:** no usar sus etiquetas ni resultados para ajustar preprocesamiento, mapeos seleccionados por rendimiento o hiperparámetros.
3. **Diseñar la evaluación:** usar validación cruzada sobre `train` para comparar stemming/lematización, n-gramas y modelos; dejar `validation_real` y `validation` para evaluaciones complementarias, tras comprobar solapamientos.
4. **Datos propios:** como primer objetivo **orientativo**, redactar 150–250 mensajes diversos de Marketing y 30–50 por cada una de las otras cinco áreas; revisar manualmente su calidad y evitar paráfrasis triviales. Mantener una muestra independiente propia para evaluación; no entrenar y evaluar con las mismas plantillas.
5. **Desbalance:** comparar F1-macro, métricas por clase y matriz de confusión; considerar pesos de clase en Regresión Logística y, en su caso, estrategias de muestreo solo dentro de entrenamiento. No asegurar probabilidades calibradas sin pruebas.
6. **Evidencia del PDF:** registrar con notebook el EDA (distribución, longitud, palabras frecuentes, faltantes), y después las etapas NLP obligatorias con ejemplos antes/después.
7. **No sumar categorías solo para aumentar los conteos:** las 44 intenciones actuales siguen siendo un mapeo preliminar.

## Cómo continuar
- Ejecutar `python -m pip install pandas matplotlib jupyter`.
- Abrir `notebooks/01_eda_sid.ipynb` en VS Code o Jupyter.
- Ejecutar todas las celdas; los CSV se leen desde `data/processed/sid/`, generados localmente (no versionados).
- Compartir el resumen de muestras dudosas y la distribución resultante. Los resultados del notebook se consideran pendientes hasta que se ejecute realmente.

## Procedencia
Los conteos proceden del `summary.json` local suministrado por el equipo tras ejecutar `prepare_sid.py`. Dataset de origen: [SID en Hugging Face](https://huggingface.co/datasets/luigicfilho/sid), licencia declarada MIT. Los conteos excluyen intenciones sin mapeo.
