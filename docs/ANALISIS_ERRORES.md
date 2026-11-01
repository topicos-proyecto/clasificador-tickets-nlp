# Análisis de errores y comparación NB frente a Regresión Logística

**Estado:** borrador generado con `python -m src.models.analyze_errors`; las causas son **heurísticas** y **falta la revisión manual** (paso B.6 de [PLAN_RL_Y_ANALISIS_ERRORES.md](PLAN_RL_Y_ANALISIS_ERRORES.md)). Solo contiene agregados y ejemplos de SID o del conjunto simulado con IA; no incluye mensajes humanos. Las métricas son exploratorias y no demuestran generalización a clientes reales. `test` no se usó.

## 1. Rejilla ampliada (64 configuraciones)

`compare.py` pasó de 16 a 64 configuraciones: stemming/lematización × unigramas/bigramas × (Naive Bayes con `alpha ∈ {0,01; 0,03; 0,1; 0,3; 0,5; 1,0}` y Regresión Logística con `C ∈ {0,1; 1; 10; 30; 100}` × `class_weight ∈ {None, balanced}`). CV estratificada de 3 particiones, semilla 42, sin fallos de convergencia. Informe: `artifacts/comparison_customers_v2/report.json` (local, no versionado).

| Modelo | Configuración | F1 macro CV (media ± desv.) |
|---|---|---:|
| NB anterior (v1) | lematización, unigramas, `alpha=0,5` | 0,9585 ± 0,0033 |
| **NB seleccionado (v2)** | lematización, unigramas, `alpha=0,1` | 0,9689 ± 0,0033 |
| Mejor LR | stemming, unigramas, `C=30`, `balanced` | **0,9715** ± 0,0033 |

- **Regla de selección (una desviación estándar):** la mejor media es la LR, pero supera al mejor NB (stemming, bigramas, `alpha=0,1`: 0,9696) por solo **0,0019**, menos que la desviación (0,0033); por la regla se elige el NB, más simple.
- **Lo que la rejilla vieja ocultaba:** LR con `C ≤ 1` estaba subajustada (0,92–0,93 en su mejor caso), y NB estaba mejor con `alpha` menor. Ambos óptimos estaban en el borde de la rejilla anterior. Con la nueva, `alpha=0,1` es un óptimo interior (0,01 y 0,03 empeoran: 0,9614 y 0,9658).
- **`min_df` / `max_df`:** `min_df=2` baja el F1 a 0,9446 y `min_df=3` a 0,9216; `max_df=0,9` no cambia nada. Se conservan `min_df=1`, `max_df=1,0` (vocabulario de 9.092 términos en el modelo final). Quitar términos raros perjudica, probablemente porque en SID muchas palabras discriminativas aparecen con variantes mal escritas (hipótesis no verificada).

## 2. Comparación sobre conjuntos de evaluación (informativa)

| Modelo | Validación SID (194) | Simulado con IA (24) | Casos diagnóstico (4) |
|---|---:|---:|---:|
| NB v1 (`alpha=0,5`) | 182 (F1 0,9452) | 18 (F1 0,7249) | 3 |
| NB v2 (`alpha=0,1`) | 184 (F1 0,9572) | 16 (F1 0,6540) | 2 |
| LR (`C=30`, `balanced`) | 187 (F1 0,9735) | 18 (F1 0,7317) | 2 |

Las diferencias entre modelos en estos conjuntos son de 2 a 3 mensajes y **no son estadísticamente distinguibles** (McNemar exacto, NB v2 frente a LR: p = 0,25 en validación y 0,50 en el simulado). Aun así, el NB v2 mejora a v1 en validación y **empeora en el conjunto simulado y en un caso diagnóstico**: «la aplicación no me funciona» pasa de Soporte TI a Marketing. Con `alpha` bajo el NB se apoya más en términos raros.

**Sensibilidad (CV agrupado por intención SID, intenciones nunca vistas en entrenamiento):** F1 macro de 0,402 (NB) frente a 0,457 (LR); la LR gana en las tres particiones. Es un escenario muy estricto y desfavorable para ambos, y no interviene en la selección, pero muestra que el 0,96–0,97 del CV principal mide en buena parte la memorización de plantillas de SID. Se lee mejor como una señal de que **LR podría generalizar mejor a redacciones nuevas**, hipótesis que solo puede confirmar un conjunto humano independiente.

## 3. Errores del modelo NB v2

### Validación SID: 10 de 194 errores (exactitud 0,948; IC 95 % 0,908–0,972)
- **Destino dominante:** 8 de los 10 errores terminan en **Soporte TI**, la clase más grande (31,6 % de SID): Ventas → Soporte TI (4), Finanzas → Soporte TI (2), Logística → Soporte TI (2).
- **Patrón:** son mensajes genéricos («Necesito asistencia para ver espedificciones», «Dime, Necesito asistencia para fasrtear mmi enfrega») cuya palabra clave está mal escrita. El modelo no encuentra el término decisivo y se queda con los términos de relleno, que pesan más en Soporte TI.
- **Margen:** la probabilidad máxima menos la segunda es de 0,83 en promedio en aciertos y 0,28 en errores; 8 de 10 errores tienen margen < 0,5 y 9 de 10 caen en el cuartil inferior. Es evidencia de que un umbral de aclaración podría ayudar, pero **no se fija ningún umbral** aquí.
- **Calibración:** ECE 0,085 en validación y 0,303 en el simulado. Las probabilidades no están calibradas.

### Conjunto simulado con IA: 8 de 24 errores (exactitud 0,667; IC 95 % 0,467–0,820)

| Esperado → predicho | Mensaje | Margen | Causa probable |
|---|---|---:|---|
| Soporte TI → Marketing | «La web no acepta mi contraseña aunque la cambié ayer.» | 0,67 | negación |
| Finanzas → Marketing | «¿Cuándo recibiré el dinero de mi devolución?» | 0,62 | vocabulario nuevo, texto corto |
| Ventas → Finanzas | «Quiero saber cuánto cuesta el plan para mi negocio.» | 0,98 | vocabulario nuevo |
| Ventas → Finanzas | «¿Qué diferencia hay entre el plan básico y el premium?» | 0,99 | vocabulario nuevo |
| Ventas → Marketing | «¿Me pueden cotizar tres equipos con sus accesorios?» | 0,25 | vocabulario nuevo, texto corto, margen bajo |
| Atención al Cliente → Soporte TI | «¿Atienden por teléfono los domingos?» | 0,51 | vocabulario nuevo, texto corto |
| Atención al Cliente → Soporte TI | «Necesito saber dónde puedo presentar una consulta general.» | 0,50 | vocabulario nuevo |
| Logística → Marketing | «Mi pedido figura en camino desde hace varios días y no llega.» | 0,50 | multi-intención, negación, margen bajo |

- **Ventas es la clase más débil** (recall 0,25): «plan», «cotizar» y «negocio» no aparecen en SID asociados a Ventas con la misma redacción; dos errores tienen margen > 0,98, es decir, **seguros pero equivocados**, lo que un umbral de confianza no resolvería.
- La etiqueta `vocabulario_desconocido` (6 de 8) indica que los mensajes usan palabras poco o nada vistas en entrenamiento: el desfase está entre el estilo sintético de SID y una redacción natural, no en errores de escritura.
- Con 4 mensajes por clase, el conjunto es demasiado pequeño para conclusiones por clase; los intervalos son muy anchos.

## 4. Taxonomía provisional de causas (a confirmar)

| Causa | Evidencia en este análisis |
|---|---|
| Palabra clave dañada por el aumento de datos de SID | Validación: la mayoría de los errores |
| Sesgo hacia la clase mayoritaria (Soporte TI) | 8 de 10 errores de validación |
| Vocabulario y estilo fuera de SID | Simulado: 6 de 8 errores |
| Ambigüedad entre áreas por definición (devolución, plan) | Finanzas/Marketing, Ventas/Finanzas |
| Multi-intención y negaciones | Casos «no se descargó la factura», «no llega» |
| Etiqueta discutible | Pendiente de revisión humana |

## 5. Decisión de promoción: pendiente

El modelo v2 **no se ha promovido**: `artifacts/comparison_customers/` (el que usa el backend) sigue intacto. Elementos para decidir:

- Por la regla fijada de antemano, la selección es **NB v2** (+0,010 de F1 frente a v1 en CV y +0,012 en validación), pero empeora en el conjunto simulado y en un caso diagnóstico.
- La LR no supera al NB por más de una desviación en CV, pero es igual o mejor en validación, en el simulado y en la sensibilidad agrupada.
- Ninguna de las diferencias es significativa con estos tamaños. La decisión final debería apoyarse en mensajes humanos nuevos y reservados.
- `evaluate_human.evaluate()` tiene fijada la configuración NB `alpha=0,5` lematizada; habrá que generalizarla antes de evaluar cualquier otro modelo. Además, cambiar el artefacto invalida `human_clients_report.json` por huella.

## 6. Reproducción

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.models.compare \
  --data-dir data/processed/model_customers \
  --output-dir artifacts/comparison_customers_v2 --tfidf-grid --grouped-cv
python -m src.models.analyze_errors \
  --model artifacts/comparison_customers_v2/pipeline.joblib \
  --dataset validacion=data/processed/model_customers/validation.csv \
  --dataset simulado=data/templates/customer_simulated_eval.csv \
  --compare-with <otro pipeline.joblib>
```

La comparación completa tarda unos 70 segundos. El pipeline de la LR de la tabla del apartado 2 se entrenó con `train.csv` y la configuración indicada; no se guardó como artefacto del repositorio.
