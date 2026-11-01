# Guía para construir el dataset propio (piloto)

## Objetivo académico

Complementar SID **sin desviarse del pipeline obligatorio del PDF**. No necesitamos fabricar miles de mensajes ni aparentar que los ejemplos generados son casos reales. Buscamos textos naturales, etiquetas consistentes y una evaluación independiente.

## Archivos

- `data/templates/custom_messages.csv`: plantilla **vacía** para datos escritos por los integrantes.
- `data/templates/custom_seed_draft.csv`: **130 ejemplos redactados con asistencia de IA**; **NO son mensajes reales de clientes**. Distribución: Marketing 90, las otras 5 áreas 8 cada una. El usuario revisó los 90 de Marketing; los otros 40 aún requieren revisión. Mantener `source=borrador_ia` y apartar casos ambiguos hasta resolverlos.
- `data/templates/internal_support_draft.csv`: **30 borradores de IA** orientados a clientes, 15 de Marketing y 15 de Soporte TI. El nombre es histórico: sus contenidos se reemplazaron tras decidir que el alcance es de clientes. El usuario los aprobó y se incorporaron a una nueva versión del entrenamiento, sin alterar las particiones anteriores.
- `data/templates/customer_simulated_eval.csv`: **24 mensajes simulados con IA**, cuatro por área. El usuario revisó sus etiquetas; solo sirven para una prueba exploratoria y no representan clientes humanos ni una evaluación final independiente.
- `data/raw/custom/messages.csv`: archivo local no versionado; actualmente contiene los borradores de IA, no mensajes humanos.
- `data/raw/custom/human_evaluation.csv`: 55 candidatos humanos locales. La copia de 44 evaluada está en `human_evaluation_frozen.csv`; `human_evaluation_reviewed.csv` registra tres correcciones posteriores y ya no es una evaluación independiente nueva. No incorporar estos conjuntos automáticamente al entrenamiento.

Formato CSV: `text,department,source,intent,language`.

`source` debe ser **`propio_humano`** si un integrante escribió la solicitud original, o **`borrador_ia`** para ejemplos generados con asistencia de IA. No cambiar el origen a conveniencia; documentar y declarar el uso de IA en el informe. `intent` es una etiqueta fina escrita en snake_case (p. ej., `factura`, `promociones`, `error_inicio_sesion`); **department** es la etiqueta para el modelo.

## Criterios de asignación

| Departamento | Incluye | No asignar solamente porque aparece una palabra |
|---|---|---|
| Soporte TI | errores de inicio de sesión, fallos, indisponibilidad, problemas técnicos | no suponer un error informático por un cobro duplicado |
| Finanzas | facturas, pagos, cobros, reembolsos, recibos | no confundir «quiero comprar» con una reclamación de pago |
| Ventas | contratar, cotizar, adquirir, comparar planes o productos | no confundir promoción anunciada con solicitud de cotización |
| Marketing | campañas, ofertas, cupones, programas de referidos/fidelidad | no confundir el precio ordinario con un descuento |
| Atención al Cliente | horarios, canales de contacto, orientación o estado de consulta general | no usar como categoría de descarte de cualquier duda |
| Logística | rastreo, transporte, entrega, dirección del envío | no clasificar solo por la palabra «pedido» cuando solicita comprar |

Ejemplos límite:
- «Me cobraron dos veces» → Finanzas (no está demostrada la causa técnica).
- «La página devuelve error 500 al descargar la factura» → Soporte TI (incidencia técnica explícita).
- «El cupón anunciado no aplica el descuento» → Marketing si es una cuestión de condiciones/promoción; marcar para revisión si es una falla técnica claramente indicada.
- «¿Cuánto cuesta el plan empresarial?» → Ventas.
- «Hola» / «Ayuda» → **no etiquetar en una de las seis clases**; se abordará con validación previa al clasificador.
- Mensajes con dos incidencias independientes o intención incierta → **apartarlos para revisión**, no inventar una clasificación única.

Decisiones históricas acordadas al revisar errores de la evaluación anterior, que incluía solicitudes internas y ya no representa el alcance actual:

- Consulta por el presupuesto asignado a anuncios → **Finanzas**.
- Solicitud de cambiar el texto de un correo promocional antes de enviarlo → **Soporte TI**, según la decisión del usuario para ese caso.
- Consulta por palabras clave para mejorar el posicionamiento web → **Soporte TI**, según la decisión del usuario para ese caso.
- Solicitudes sobre segmentar publicidad, reportar resultados de campañas o preparar material publicitario → **Marketing**. Cuando el contexto no permita distinguir contenido de una tarea técnica, apartar para aclaración.

Estas son reglas de enrutamiento para este proyecto; no se deduce que deban aplicarse igual en todas las empresas.

Para nuevos datos, redactar solo solicitudes de clientes. Los ejemplos internos anteriores no forman parte del nuevo alcance; los criterios históricos para tareas internas no deben usarse para incorporar ese tipo de mensajes. Los casos de promociones, cupones y fidelidad de clientes se revisan como Marketing; las fallas técnicas explícitas de aplicaciones o páginas, como Soporte TI.

## Protocolo de redacción y revisión

1. Cada integrante escribe solicitudes **originales**, simulando distintas formas de expresarse: informal, formal, con faltas habituales y con distinta longitud. No copiar ni sustituir una sola palabra en las frases SID.
2. No introducir datos personales reales, contraseñas, números de cuenta, teléfonos ni direcciones reales.
3. Cada ejemplo recibe una única etiqueta principal y `intent`. Si dos responsables discrepan, marcar para revisión fuera del CSV de entrenamiento hasta consensuar la etiqueta.
4. Un compañero revisa al menos las muestras dudosas y una muestra aleatoria de cada categoría. Registrar criterios y desacuerdos en el informe.
5. Ejecutar el validador y revisar los mensajes demasiado cortos, duplicados exactos y coincidencias con SID.
6. Reservar el archivo humano independiente para evaluación. El script `build_splits.py` incorpora solo los 90 borradores revisados de Marketing al entrenamiento y comprueba similitudes con las particiones SID; no incorpora datos humanos. Antes de evaluar con mensajes humanos, revisar etiquetas, duplicados y similitudes con entrenamiento y congelar el archivo.

### Objetivo para la primera ronda

Comenzar por **40–60 mensajes humanos**: alrededor de 20–25 de Marketing y 4–8 de cada uno de los otros cinco departamentos. Son metas de trabajo, **no requisitos del PDF**. Después de medir cobertura y dificultades, ampliar donde sea necesario. Evitar completar una cuota duplicando variantes triviales.

## Pasos locales (desde la raíz del repositorio)

Crear el archivo humano copiando solamente la **plantilla vacía**, si todavía no existe (no sobrescribir mensajes previos):

```bash
mkdir -p data/raw/custom
test -e data/raw/custom/human_evaluation_clients.csv || cp data/templates/custom_messages.csv data/raw/custom/human_evaluation_clients.csv
```

Editar `human_evaluation_clients.csv` con un editor CSV, hoja de cálculo o VS Code. Por ejemplo:

```csv
text,department,source,intent,language
"¿Hay descuentos para estudiantes durante noviembre?",Marketing,propio_humano,descuentos,es
"Mi envío figura entregado pero todavía no lo recibí",Logística,propio_humano,entrega_extraviada,es
```

**Son ejemplos de formato**: no copiarlos literalmente para llenar el dataset. Conservar comillas CSV cuando el texto contenga comas o comillas.

Validar el borrador de IA como práctica:

```bash
python src/data/validate_custom.py data/templates/custom_seed_draft.csv
```

Validar el formato de los mensajes propios:

```bash
python src/data/validate_custom.py data/raw/custom/human_evaluation_clients.csv
```

El validador muestra distribución y errores, **no determina si la etiqueta semántica es correcta**. La revisión humana sigue siendo imprescindible.

Antes de congelar el conjunto humano, comprobar también similitudes con entrenamiento y las dos validaciones:

```bash
python -m src.data.audit_human_eval data/raw/custom/human_evaluation_clients.csv
```

El comando rechaza archivos vacíos o con fuentes que no sean `propio_humano`, señala líneas similares y áreas sin ejemplos, y calcula huellas SHA-256. No lee el test ni ejecuta el modelo. Una salida sin coincidencias no descarta todas las paráfrasis: revisar manualmente antes de congelar el CSV. Los mensajes ya usados para probar modelos no son evaluación independiente.

## Próximo paso del pipeline obligatorio

Una vez validada la fuente mixta, ejecutar EDA de ambas fuentes y continuar con la **normalización, tokenización, stop words, stemming vs. lematización, unigramas vs. bigramas y TF-IDF**. No seleccionar modelos ni hiperparámetros basándose en el test final.
