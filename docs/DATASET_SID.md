# Dataset seleccionado: SID + datos propios

## Fuente original verificada

- **Nombre:** SID — Synthetic Intent Dataset
- **Autor:** Luigicfilho
- **Ficha:** https://huggingface.co/datasets/luigicfilho/sid
- **Revisión fijada:** `43fad0f7d60bd1dfb0402c8632deb2cbc13f4731`; las huellas SHA-256 de los cinco archivos se registran en `data/sid_source.json`.
- **Licencia declarada:** MIT; conservar referencia y verificar cualquier cambio antes de distribuir datos.
- **Formato:** CSV, campos `text`, `label`, `language`.
- **Idiomas originales:** en, es, pt, de, fr.
- **Particiones y archivos oficiales:**
  - `train.csv` (train)
  - `val.csv` (validation)
  - `val_real.csv` (validation_real)
  - `test.csv` (test)
  - `intents.json` (catálogo de intenciones)
- **Advertencia:** el dataset contiene textos sintéticos y aumentados con errores ortográficos; `validation_real` es el nombre del autor y no garantiza que cada registro provenga de una conversación auténtica.
- **Fecha de selección:** 30 de octubre de 2026.

## Decisiones metodológicas

1. Filtrar `language == "es"`. No traducir otros idiomas automáticamente.
2. Convertir **intenciones SID** a las **seis áreas** del proyecto con `data/sid_intent_map.json` (mapeo preliminar propio del equipo).
3. **Excluir y contabilizar** las intenciones no mapeadas, incluidos saludos. No usar una clase forzada para mensajes como «Hola».
4. Mantener el split original en los archivos de salida para facilitar las evaluaciones; no seleccionar hiperparámetros usando `test`.
5. Deduplicar por texto normalizado de manera conservadora. Se priorizan particiones de evaluación para reducir duplicación exacta; esto **no detecta paráfrasis** ni elimina todas las posibles fugas entre ejemplos sintéticos.
6. Evaluar cuidadosamente la cobertura desigual: Marketing solo tiene `apply_discount_coupon` como intención claramente aprovechable; requiere mensajes propios con campañas, ofertas, publicidad y descuentos.
7. Recopilar datos propios con consentimiento cuando corresponda; no almacenar nombres, contraseñas, cuentas ni otros datos personales reales. Crear textos nuevos y naturales, no simples copias mínimamente modificadas.
8. Documentar quién revisó cada etiqueta y cómo resolvió desacuerdos; reservar datos propios independientes para una evaluación complementaria.
9. Registrar licencia, fecha, URL y huellas SHA-256 de los archivos originales. No versionar grandes archivos crudos ni información privada.

## Etiquetas de trabajo

- **Soporte TI:** fallos técnicos y de acceso.
- **Finanzas:** facturas, pagos, reembolsos.
- **Ventas:** cotizaciones, adquisición, precios y características.
- **Marketing:** promociones, cupones y campañas.
- **Atención al Cliente:** consultas generales, horarios y orientación.
- **Logística:** entregas, seguimiento, envíos.

`request_refund` se propone inicialmente como Finanzas. `report_delivery_issue` se propone como Logística. Intenciones con múltiples posibles dueños quedan sin mapear hasta revisión. El mapeo **no identifica causas técnicas** de los problemas.

## Primer paso de ejecución

Desde la raíz de este repositorio, con Python 3.10+:

```bash
python src/data/prepare_sid.py
```

Sin dependencias externas: usa la biblioteca estándar. Necesita conexión a internet solo para descargar los CSV originales de Hugging Face; también permite `--offline` si esos archivos ya están en `data/raw/sid/`. Antes de preparar las particiones, verifica las huellas de todos los archivos descargados o reutilizados. Si un archivo no coincide, hay que conservarlo o moverlo fuera del directorio y descargar la revisión fijada.

Genera:
- `data/raw/sid/` CSV originales y `intents.json`.
- `data/processed/sid/` CSV por split en español y mapeados.
- `data/processed/sid/summary.json` con EDA preliminar por clase, volumen, faltantes, duplicados y cobertura de intenciones.
- `data/processed/sid/manifest.json` con SHA-256 de los archivos originales.

Los archivos en `data/raw/` y `data/processed/` están excluidos por `.gitignore`, por lo que cada integrante puede regenerarlos.

## Para datos propios

Copiar `data/templates/custom_messages.csv` a `data/raw/custom/messages.csv` y añadir una fila por mensaje, conservando la cabecera `text,department,source,intent,language`. Ejemplo de fila **ilustrativa**:

```csv
"¿Qué ofertas tienen para nuevos clientes?",Marketing,propio,promociones,es
```

No se mezclará esta fuente con los splits SID hasta definir el protocolo de separación y evitar filtraciones entre entrenamiento y prueba.

## Calidad antes de entrenar

- Revisar `summary.json` y comparar los tamaños reales por clase.
- Abrir y revisar al menos varias muestras por departamento, en especial Marketing.
- Detectar near-duplicates, ruido sintético y mensajes fuera de dominio.
- Aprobar formalmente el mapeo de intenciones con el equipo.
- Construir un dataset propio para compensar categorías débiles.
- Documentar tratamiento de desbalance, separación fija y semilla; reservar `test`.

Esto **no** sustituye los experimentos exigidos después: normalización, tokenización, stop words, stemming vs. lematización, unigramas vs. bigramas y TF-IDF.
