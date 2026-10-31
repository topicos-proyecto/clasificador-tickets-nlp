# Guía para construir el dataset propio (piloto)

## Objetivo académico

Complementar SID **sin desviarse del pipeline obligatorio del PDF**. No necesitamos fabricar miles de mensajes ni aparentar que los ejemplos generados son casos reales. Buscamos textos naturales, etiquetas consistentes y una evaluación independiente.

## Archivos

- `data/templates/custom_messages.csv`: plantilla **vacía** para datos escritos por los integrantes.
- `data/templates/custom_seed_draft.csv`: **70 ejemplos redactados con asistencia de IA**, sin revisión humana aún; **NO son mensajes reales de clientes**. Distribución: Marketing 30, las otras 5 áreas 8 cada una. Sirven para discutir la guía y practicar etiquetado. Mantener `source=borrador_ia` si se utilizan tras revisión.
- `data/raw/custom/messages.csv`: archivo local para los mensajes realmente redactados por personas, **no versionado** para facilitar privacidad, control de calidad y revisión.

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

## Protocolo de redacción y revisión

1. Cada integrante escribe solicitudes **originales**, simulando distintas formas de expresarse: informal, formal, con faltas habituales y con distinta longitud. No copiar ni sustituir una sola palabra en las frases SID.
2. No introducir datos personales reales, contraseñas, números de cuenta, teléfonos ni direcciones reales.
3. Cada ejemplo recibe una única etiqueta principal y `intent`. Si dos responsables discrepan, marcar para revisión fuera del CSV de entrenamiento hasta consensuar la etiqueta.
4. Un compañero revisa al menos las muestras dudosas y una muestra aleatoria de cada categoría. Registrar criterios y desacuerdos en el informe.
5. Ejecutar el validador y revisar los mensajes demasiado cortos, duplicados exactos y coincidencias con SID.
6. **No mezclar automáticamente** el conjunto propio con `train`, `validation` ni `test` de SID. Definir y congelar antes un conjunto independiente de evaluación con mensajes humanos no vistos en entrenamiento; controlar textos similares entre particiones.

### Objetivo para la primera ronda

Comenzar por **40–60 mensajes humanos**: alrededor de 20–25 de Marketing y 4–8 de cada uno de los otros cinco departamentos. Son metas de trabajo, **no requisitos del PDF**. Después de medir cobertura y dificultades, ampliar donde sea necesario. Evitar completar una cuota duplicando variantes triviales.

## Pasos locales (desde la raíz del repositorio)

Crear el archivo humano copiando solamente la **plantilla vacía**:

```bash
mkdir -p data/raw/custom
cp data/templates/custom_messages.csv data/raw/custom/messages.csv
```

Editar `messages.csv` con un editor CSV, hoja de cálculo o VS Code. Por ejemplo:

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

Validar los mensajes propios (requiere SID preparado si se usa `--compare-sid`):

```bash
python src/data/validate_custom.py data/raw/custom/messages.csv --compare-sid
```

El validador muestra distribución y errores, **no determina si la etiqueta semántica es correcta**. La revisión humana sigue siendo imprescindible.

## Próximo paso del pipeline obligatorio

Una vez validada la fuente mixta, ejecutar EDA de ambas fuentes y continuar con la **normalización, tokenización, stop words, stemming vs. lematización, unigramas vs. bigramas y TF-IDF**. No seleccionar modelos ni hiperparámetros basándose en el test final.
