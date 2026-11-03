# Comparación de modelos: Naive Bayes, Regresión Logística y KNN

> Documento generado con `python -m src.models.render_comparison_report artifacts/comparison_customers_v3` a partir de `report.json` (SHA-256 `bc0945286289…`). No editar las cifras a mano; la sección 12 se redacta a mano y se conserva al regenerar.

## 1. Resumen de la decisión

- **Modelo elegido:** Naive Bayes (lematización, unigramas, alpha=0.1).
- **F1 macro en validación cruzada:** 0,9689 ± 0,0033 (3 particiones estratificadas, semilla 42).
- **Mejor media absoluta:** Regresión Logística (stemming, unigramas, C=30.0, class_weight=balanced) con 0,9715 ± 0,0033. Umbral de la regla de una desviación estándar: **0,9683**.
- **Modelos cuyo mejor resultado queda dentro del umbral:** Naive Bayes, Regresión Logística. Entre configuraciones estadísticamente equivalentes se elige la más simple.
- **Validación SID:** F1 macro 0,9572, exactitud 94,8 %. `test` se evaluó una sola vez después de la selección (sección 10).

## 2. Datos usados

| Partición | Registros | Marketing | Palabras (media / mediana / máx.) | Uso |
|---|---:|---:|---|---|
| `train` | 15.560 | 387 | 5,91 / 6 / 14 | Validación cruzada y ajuste final |
| `validation` | 194 | 4 | 7,06 / 7 / 11 | Evaluación complementaria |
| `validation_real` | 448 | 8 | 5,27 / 5 / 8 | Evaluación complementaria |
| `test` | 707 | 19 | — (no se procesan textos) | Reservado: evaluación única final |

Distribución de clases en entrenamiento:

| Área | Registros | Porcentaje |
|---|---:|---:|
| Atención al Cliente | 2.828 | 18,2 % |
| Finanzas | 2.655 | 17,1 % |
| Logística | 1.978 | 12,7 % |
| Marketing | 387 | 2,5 % |
| Soporte TI | 4.894 | 31,4 % |
| Ventas | 2.818 | 18,1 % |

**Desbalance:** Marketing representa 2,5 % del entrenamiento. Un clasificador que nunca predijera Marketing seguiría alcanzando hasta 97,5 % de exactitud. Por eso la exactitud se informa, pero el criterio de selección es **F1 macro**, que da el mismo peso a las seis áreas.

**Textos cortos:** la mediana es de 6 palabras por mensaje; tras el preprocesamiento quedan aún menos términos, de modo que los vectores TF-IDF son muy dispersos.

Con el modelo elegido (lematización), cada mensaje de entrenamiento conserva en promedio 4,05 términos y el vocabulario TF-IDF tiene 9.092 términos.

## 3. Metodología común

- Los tres modelos usan exactamente el mismo preprocesamiento por variante (normalización, tokenización, stop words en español con negaciones conservadas, stemming o lematización) y TF-IDF con unigramas o unigramas + bigramas.
- Validación cruzada estratificada de 3 particiones sobre `train`, semilla 42; las mismas particiones para todas las configuraciones, lo que permite comparar por partición (diferencias pareadas).
- TF-IDF se ajusta **dentro** de cada partición: el vocabulario de validación no se filtra al entrenamiento.
- Se evaluaron **120 configuraciones**: Naive Bayes 24, Regresión Logística 40, KNN 56.
- **Regla de selección (fijada antes de ver resultados):** Mayor media de F1 macro en CV. Se aplica la regla de una desviación estándar: entre las configuraciones cuya media no queda por debajo de (mejor media - desviación de la mejor) se elige la más simple: unigramas, luego el modelo más simple (Naive Bayes, después Regresión Logística y por último KNN, que guarda todo el entrenamiento y es el más costoso en inferencia), mayor regularización (alpha mayor, C menor, k mayor), sin pesos de clase ni ponderación por distancia y stemming; la última desambiguación es la mayor media.
- `test` no interviene en la selección. Los conjuntos humanos y simulados tampoco.

## 4. Los tres modelos en lenguaje simple

**Naive Bayes multinomial.** Estima, para cada área, con qué frecuencia aparece cada término y combina esas frecuencias suponiendo que los términos son independientes. `alpha` suaviza las frecuencias para que un término nunca visto en un área no anule su probabilidad. Entrena en una sola pasada por los datos y su modelo es una tabla de pesos por área y término.

**Regresión Logística.** Aprende un peso por término y área ajustándolos para separar las clases. `C` controla la regularización (menor `C` = pesos más pequeños = modelo más simple); `class_weight=balanced` da más importancia a las clases pequeñas como Marketing. Suele dar probabilidades más informativas que NB, pero su entrenamiento es iterativo y más costoso.

**KNN (k vecinos más cercanos).** No aprende parámetros: guarda todos los mensajes de entrenamiento y, para clasificar uno nuevo, busca los `k` más parecidos (distancia coseno entre vectores TF-IDF) y vota. `weights=distance` da más voto a los vecinos más cercanos. Su «probabilidad» es la fracción de vecinos de cada área. Es el más sensible a la presencia de frases casi idénticas entre entrenamiento y evaluación, y el más costoso al predecir porque compara contra todo el entrenamiento.

## 5. Resultados de validación cruzada

| Modelo | Mejor configuración por media | F1 macro CV | ¿Dentro del umbral? |
|---|---|---:|---|
| **Naive Bayes (elegido)** | stemming, uni+bigramas, alpha=0.1 | 0,9696 ± 0,0025 | sí |
| Regresión Logística | stemming, unigramas, C=30.0, class_weight=balanced | 0,9715 ± 0,0033 | sí |
| KNN | stemming, unigramas, n_neighbors=5, weights=distance | 0,9335 ± 0,0037 | no |

Configuración desplegada de Naive Bayes según la regla: lematización, unigramas, alpha=0.1, F1 macro 0,9689 ± 0,0033.

**Diferencias pareadas por partición** (mejor de cada modelo, mismas particiones):

| Par (A − B) | Diferencias por partición | Media | Desviación |
|---|---|---:|---:|
| Naive Bayes − Regresión Logística | 0,0010, -0,0017, -0,0051 | -0,0019 | 0,0025 |
| Naive Bayes − KNN | 0,0355, 0,0304, 0,0425 | 0,0361 | 0,0050 |
| Regresión Logística − KNN | 0,0345, 0,0321, 0,0475 | 0,0380 | 0,0068 |

**Efecto del preprocesamiento** (mejor F1 macro CV por variante):

| Modelo | Stemming · 1 | Stemming · 1-2 | Lematización · 1 | Lematización · 1-2 |
|---|---:|---:|---:|---:|
| Naive Bayes | 0,9658 | 0,9696 | 0,9689 | 0,9676 |
| Regresión Logística | 0,9715 | 0,9714 | 0,9694 | 0,9693 |
| KNN | 0,9335 | 0,9310 | 0,9316 | 0,9210 |

## 6. Métricas por clase y matrices de confusión

La matriz de confusión fuera de partición reúne las predicciones de las tres particiones del CV: cada mensaje de entrenamiento (15.560) se predice con un modelo que no lo vio. Es mucho más estable que la de validación, que tiene muy pocos ejemplos de Marketing. Filas = área real; columnas = área predicha; la diagonal (negrita) son aciertos.

| Modelo | F1 macro OOF | Exactitud OOF | Exactitud balanceada OOF | F1 macro val. | Exactitud val. | F1 macro val. adicional | F1 macro simulado |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive Bayes | 0,9689 | 96,6 % | 0,9681 | 0,9572 | 94,8 % | 0,9538 | 0,6540 |
| Regresión Logística | 0,9715 | 97,0 % | 0,9705 | 0,9735 | 96,4 % | 0,9629 | 0,7317 |
| KNN | 0,9335 | 92,8 % | 0,9315 | 0,8676 | 83,5 % | 0,8839 | 0,5692 |

### Naive Bayes — lematización, unigramas, alpha=0.1 (elegido)

**Por clase, fuera de partición (CV):**

| Área | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Atención al Cliente | 0,984 | 0,983 | 0,984 | 2.828 |
| Finanzas | 0,952 | 0,965 | 0,959 | 2.655 |
| Logística | 0,969 | 0,976 | 0,973 | 1.978 |
| Marketing | 0,987 | 0,987 | 0,987 | 387 |
| Soporte TI | 0,958 | 0,975 | 0,966 | 4.894 |
| Ventas | 0,968 | 0,922 | 0,944 | 2.818 |
| **Promedio macro** | 0,970 | 0,968 | 0,969 | 15.560 |
| Promedio ponderado | 0,966 | 0,966 | 0,966 | 15.560 |
| Exactitud |  |  | 0,966 | 15.560 |

**Matriz de confusión, fuera de partición (CV):**

| Real \ Predicho | At. Cliente | Finanzas | Logística | Marketing | Sop. TI | Ventas |
|---|---:|---:|---:|---:|---:|---:|
| **At. Cliente** | **2.781** | 8 | 13 | 0 | 16 | 10 |
| **Finanzas** | 9 | **2.563** | 3 | 0 | 54 | 26 |
| **Logística** | 4 | 4 | **1.931** | 0 | 25 | 14 |
| **Marketing** | 1 | 0 | 0 | **382** | 2 | 2 |
| **Sop. TI** | 14 | 54 | 16 | 5 | **4.771** | 34 |
| **Ventas** | 16 | 62 | 29 | 0 | 113 | **2.598** |

**Por clase, validación SID:**

| Área | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Atención al Cliente | 1,000 | 1,000 | 1,000 | 39 |
| Finanzas | 0,958 | 0,920 | 0,939 | 25 |
| Logística | 1,000 | 0,885 | 0,939 | 26 |
| Marketing | 1,000 | 1,000 | 1,000 | 4 |
| Soporte TI | 0,875 | 0,982 | 0,926 | 57 |
| Ventas | 0,975 | 0,907 | 0,940 | 43 |
| **Promedio macro** | 0,968 | 0,949 | 0,957 | 194 |
| Promedio ponderado | 0,952 | 0,948 | 0,949 | 194 |
| Exactitud |  |  | 0,948 | 194 |

### Regresión Logística — stemming, unigramas, C=30.0, class_weight=balanced

**Por clase, fuera de partición (CV):**

| Área | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Atención al Cliente | 0,984 | 0,974 | 0,979 | 2.828 |
| Finanzas | 0,960 | 0,969 | 0,965 | 2.655 |
| Logística | 0,984 | 0,978 | 0,981 | 1.978 |
| Marketing | 0,990 | 0,974 | 0,982 | 387 |
| Soporte TI | 0,977 | 0,974 | 0,976 | 4.894 |
| Ventas | 0,940 | 0,954 | 0,947 | 2.818 |
| **Promedio macro** | 0,973 | 0,971 | 0,972 | 15.560 |
| Promedio ponderado | 0,970 | 0,970 | 0,970 | 15.560 |
| Exactitud |  |  | 0,970 | 15.560 |

**Matriz de confusión, fuera de partición (CV):**

| Real \ Predicho | At. Cliente | Finanzas | Logística | Marketing | Sop. TI | Ventas |
|---|---:|---:|---:|---:|---:|---:|
| **At. Cliente** | **2.755** | 3 | 4 | 0 | 19 | 47 |
| **Finanzas** | 4 | **2.573** | 4 | 0 | 34 | 40 |
| **Logística** | 7 | 1 | **1.934** | 0 | 14 | 22 |
| **Marketing** | 1 | 1 | 1 | **377** | 1 | 6 |
| **Sop. TI** | 5 | 48 | 16 | 4 | **4.765** | 56 |
| **Ventas** | 28 | 53 | 6 | 0 | 42 | **2.689** |

**Por clase, validación SID:**

| Área | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Atención al Cliente | 1,000 | 1,000 | 1,000 | 39 |
| Finanzas | 1,000 | 0,960 | 0,980 | 25 |
| Logística | 1,000 | 0,962 | 0,980 | 26 |
| Marketing | 1,000 | 1,000 | 1,000 | 4 |
| Soporte TI | 0,903 | 0,982 | 0,941 | 57 |
| Ventas | 0,975 | 0,907 | 0,940 | 43 |
| **Promedio macro** | 0,980 | 0,968 | 0,973 | 194 |
| Promedio ponderado | 0,966 | 0,964 | 0,964 | 194 |
| Exactitud |  |  | 0,964 | 194 |

### KNN — stemming, unigramas, n_neighbors=5, weights=distance

**Por clase, fuera de partición (CV):**

| Área | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Atención al Cliente | 0,957 | 0,946 | 0,952 | 2.828 |
| Finanzas | 0,892 | 0,927 | 0,909 | 2.655 |
| Logística | 0,925 | 0,935 | 0,930 | 1.978 |
| Marketing | 0,987 | 0,961 | 0,974 | 387 |
| Soporte TI | 0,932 | 0,944 | 0,938 | 4.894 |
| Ventas | 0,923 | 0,876 | 0,899 | 2.818 |
| **Promedio macro** | 0,936 | 0,931 | 0,934 | 15.560 |
| Promedio ponderado | 0,929 | 0,928 | 0,928 | 15.560 |
| Exactitud |  |  | 0,928 | 15.560 |

**Matriz de confusión, fuera de partición (CV):**

| Real \ Predicho | At. Cliente | Finanzas | Logística | Marketing | Sop. TI | Ventas |
|---|---:|---:|---:|---:|---:|---:|
| **At. Cliente** | **2.676** | 31 | 25 | 1 | 50 | 45 |
| **Finanzas** | 25 | **2.460** | 21 | 1 | 87 | 61 |
| **Logística** | 12 | 24 | **1.849** | 0 | 66 | 27 |
| **Marketing** | 0 | 2 | 7 | **372** | 1 | 5 |
| **Sop. TI** | 37 | 124 | 44 | 3 | **4.619** | 67 |
| **Ventas** | 46 | 118 | 53 | 0 | 132 | **2.469** |

**Por clase, validación SID:**

| Área | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Atención al Cliente | 0,842 | 0,821 | 0,831 | 39 |
| Finanzas | 0,880 | 0,880 | 0,880 | 25 |
| Logística | 0,889 | 0,923 | 0,906 | 26 |
| Marketing | 1,000 | 1,000 | 1,000 | 4 |
| Soporte TI | 0,790 | 0,860 | 0,824 | 57 |
| Ventas | 0,816 | 0,721 | 0,765 | 43 |
| **Promedio macro** | 0,870 | 0,867 | 0,868 | 194 |
| Promedio ponderado | 0,835 | 0,835 | 0,834 | 194 |
| Exactitud |  |  | 0,835 | 194 |

## 7. Sensibilidad: CV agrupado por intención SID

Cada partición deja fuera intenciones completas (56 intenciones): el modelo debe clasificar formulaciones de solicitudes que nunca vio. Es una prueba mucho más exigente que el CV principal y sirve para detectar si un modelo depende de memorizar plantillas. No interviene en la selección.

| Modelo | Configuración | F1 macro CV | F1 macro agrupado | Diferencia |
|---|---|---:|---:|---:|
| Naive Bayes | stemming, uni+bigramas, alpha=0.1 | 0,9696 | 0,4018 | -0,5678 |
| Regresión Logística | stemming, unigramas, C=30.0, class_weight=balanced | 0,9715 | 0,4568 | -0,5147 |
| KNN | stemming, unigramas, n_neighbors=5, weights=distance | 0,9335 | 0,3408 | -0,5927 |

## 8. Costo computacional

| Modelo | Entrenamiento (s) | Predicción 1000 msg., pipeline (ms) | Predicción 1000 msg., solo clasificador (ms) | Tamaño del artefacto | Vocabulario |
|---|---:|---:|---:|---:|---:|
| Naive Bayes | 13,59 | 808,2 | 0,2 | 1,02 MB | 9.092 |
| Regresión Logística | 0,87 | 41,5 | 0,2 | 0,53 MB | 7.981 |
| KNN | 0,55 | 141,1 | 97,8 | 1,06 MB | 7.981 |

El tiempo del pipeline incluye la tokenización (con lematización, spaCy domina el costo). Las cifras se midieron en el equipo que ejecutó la comparación y sirven para comparar modelos entre sí, no como garantía.

## 9. Decisión y justificación

1. La mejor media de F1 macro es 0,9715 (Regresión Logística), con desviación 0,0033; el umbral de equivalencia es 0,9683.
2. Entre las configuraciones con media ≥ umbral, la regla elige la más simple: **Naive Bayes (lematización, unigramas, alpha=0.1)**, con 0,9689.
3. Regresión Logística: mejor media 0,9715 (diferencia con el elegido -0,0027); queda dentro del umbral, pero es un modelo más complejo según la regla.
3. KNN: mejor media 0,9335 (diferencia con el elegido 0,0354); queda por debajo del umbral.
4. Los criterios secundarios (sensibilidad agrupada, costo, tamaño) se informan para explicar la decisión, pero no la cambian: la regla se fijó antes de ver los resultados.

## 10. Evaluación final en `test` y evaluaciones exploratorias

Evaluación **única**, posterior a la selección (`test_report.json`, 707 mensajes). No se usó para elegir ni ajustar.

| Modelo | F1 macro | Exactitud | Exactitud balanceada | F1 Marketing |
|---|---:|---:|---:|---:|
| Naive Bayes (elegido) | 0,9586 | 95,3 % | 0,9563 | 1,0000 |
| Regresión Logística | 0,9597 | 95,9 % | 0,9565 | 0,9730 |
| KNN | 0,9101 | 89,8 % | 0,9122 | 1,0000 |

**Por clase, modelo elegido en `test`:**

| Área | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Atención al Cliente | 0,969 | 0,969 | 0,969 | 128 |
| Finanzas | 0,943 | 0,935 | 0,939 | 124 |
| Logística | 0,952 | 0,940 | 0,946 | 84 |
| Marketing | 1,000 | 1,000 | 1,000 | 19 |
| Soporte TI | 0,943 | 0,977 | 0,960 | 222 |
| Ventas | 0,960 | 0,915 | 0,937 | 130 |
| **Promedio macro** | 0,961 | 0,956 | 0,959 | 707 |
| Promedio ponderado | 0,953 | 0,953 | 0,953 | 707 |
| Exactitud |  |  | 0,953 | 707 |

**Matriz de confusión, modelo elegido en `test`:**

| Real \ Predicho | At. Cliente | Finanzas | Logística | Marketing | Sop. TI | Ventas |
|---|---:|---:|---:|---:|---:|---:|
| **At. Cliente** | **124** | 1 | 2 | 0 | 1 | 0 |
| **Finanzas** | 0 | **116** | 0 | 0 | 5 | 3 |
| **Logística** | 1 | 1 | **79** | 0 | 3 | 0 |
| **Marketing** | 0 | 0 | 0 | **19** | 0 | 0 |
| **Sop. TI** | 0 | 3 | 0 | 0 | **217** | 2 |
| **Ventas** | 3 | 2 | 2 | 0 | 4 | **119** |

**Conjunto simulado con IA (24 mensajes, exploratorio):** Naive Bayes acierta 16 de 24; Regresión Logística acierta 18 de 24; KNN acierta 15 de 24. Son pocos mensajes y no son humanos.

La evaluación humana de clientes requiere archivos privados que no están en este equipo; debe ejecutarse con el modelo elegido **después** de la selección y nunca para elegir.

## 11. Limitaciones

- SID es mayormente sintético: hay variantes casi idénticas entre particiones que inflan el CV, sobre todo para KNN.
- La validación SID tiene muy pocos ejemplos de Marketing; sus métricas por clase son inestables.
- Las probabilidades no están calibradas; en KNN son fracciones de vecinos (valores discretos).
- Las diferencias entre los mejores modelos son del orden de la desviación entre particiones: con otros datos el orden podría cambiar.

## 12. Interpretación del equipo

<!-- INTERPRETACION:INICIO -->
**Por qué se elige Naive Bayes.** La regla de selección se fijó antes de ver resultados: entre las configuraciones cuya media de F1 macro no es peor que la mejor menos una desviación estándar, se elige la más simple. La mejor Regresión Logística (0,9715) y el Naive Bayes elegido (0,9689) difieren en 0,0027, menos que la variación entre particiones (≈0,0033). Con estos datos no se puede afirmar que LR sea mejor que NB en CV, y NB es el modelo más simple: un solo hiperparámetro, entrenamiento en una pasada y una interpretación directa (frecuencias de términos por área).

**Por qué no KNN.** KNN queda claramente fuera del umbral (0,9335, unos 3,5 puntos por debajo), en todas las particiones (diferencias pareadas consistentes). En el CV agrupado por intención cae a 0,34, la caída más fuerte de los tres: depende de encontrar mensajes casi idénticos en el entrenamiento y generaliza peor a formulaciones nuevas. También es el más lento en el clasificador (compara cada mensaje con las 15.560 filas) y sus probabilidades son fracciones discretas de vecinos.

**Lo que el equipo debe saber de LR.** Regresión Logística obtiene cifras algo mayores que NB en casi todas las evaluaciones secundarias: validación SID, validación adicional, conjunto simulado (18 frente a 16 aciertos de 24), CV agrupado (0,46 frente a 0,40) y `test` (0,9597 frente a 0,9586). Su configuración usa stemming, mucho más rápido que la lematización con spaCy: el pipeline de LR predice 1.000 mensajes en unas decenas de milisegundos, frente a casi un segundo con NB. Ninguna de estas diferencias en F1 supera con claridad la variación esperada, y cambiar la regla **después** de mirar `test` sería seleccionar con `test`. Por eso se mantiene NB como modelo desplegado y estas observaciones quedan como argumento para una futura versión, que debería decidirse con datos humanos nuevos y reservados.

**Para la defensa.** La exactitud en `test` del modelo elegido es 95,3 %, pero el criterio es F1 macro (0,9586) porque con Marketing en 2,5 % la exactitud oculta los errores en las clases pequeñas. Las cifras sobre datos SID son optimistas (datos sintéticos con variantes cercanas); la caída en el CV agrupado y en el conjunto simulado es la mejor estimación disponible de cómo se comportarían los modelos con mensajes reales.
<!-- INTERPRETACION:FIN -->
