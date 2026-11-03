"""Genera docs/COMPARACION_MODELOS.md a partir de report.json, dataset_stats.json y test_report.json.

Las cifras se leen de los artefactos para evitar errores de transcripción. La sección de
interpretación escrita a mano se conserva entre los marcadores INTERPRETACION al regenerar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.train import LABELS, ROOT, digest

NAMES = {'naive_bayes': 'Naive Bayes', 'logistic_regression': 'Regresión Logística', 'knn': 'KNN'}
METHODS = {'stemming': 'stemming', 'lemmatization': 'lematización'}
SETS = {'validation': 'validación SID', 'validation_real': 'validación adicional SID',
        'simulated': 'simulado con IA (24)'}
START, END = '<!-- INTERPRETACION:INICIO -->', '<!-- INTERPRETACION:FIN -->'
DEFAULT_INTERPRETATION = '_Pendiente de redactar por el equipo._'


def num(value, digits: int = 4) -> str:
    if value is None:
        return '—'
    return f'{value:,.{digits}f}'.replace(',', ' ').replace('.', ',').replace(' ', '.')


def pct(value) -> str:
    return '—' if value is None else num(value * 100, 1) + ' %'


def integer(value) -> str:
    return f'{value:,}'.replace(',', '.')


def config_text(c: dict) -> str:
    extra = []
    if c.get('class_weight'):
        extra.append(f"class_weight={c['class_weight']}")
    if c.get('weights'):
        extra.append(f"weights={c['weights']}")
    value = c['parameter_value']
    value = int(value) if c['parameter_name'] == 'n_neighbors' else value
    params = ', '.join([f"{c['parameter_name']}={value}"] + extra)
    return f"{METHODS[c['method']]}, {'unigramas' if c['ngrams'] == 1 else 'uni+bigramas'}, {params}"


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or 'l' + 'r' * (len(header) - 1)
    marks = ['---:' if a == 'r' else '---' for a in align]
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(marks) + '|']
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(lines)


def per_class(metrics: dict) -> str:
    rep = metrics['classification_report']
    rows = [[label, num(rep[label]['precision'], 3), num(rep[label]['recall'], 3),
             num(rep[label]['f1-score'], 3), integer(int(rep[label]['support']))] for label in LABELS]
    for key, name in (('macro avg', '**Promedio macro**'), ('weighted avg', 'Promedio ponderado')):
        rows.append([name, num(rep[key]['precision'], 3), num(rep[key]['recall'], 3),
                     num(rep[key]['f1-score'], 3), integer(int(rep[key]['support']))])
    rows.append(['Exactitud', '', '', num(metrics['accuracy'], 3), integer(metrics['records'])])
    return table(['Área', 'Precisión', 'Recall', 'F1', 'Soporte'], rows)


def matrix(metrics: dict) -> str:
    short = {'Atención al Cliente': 'At. Cliente', 'Soporte TI': 'Sop. TI'}
    header = ['Real \\ Predicho'] + [short.get(l, l) for l in LABELS]
    rows = []
    for label, row in zip(LABELS, metrics['confusion_matrix']):
        rows.append([f'**{short.get(label, label)}**'] +
                    [f'**{integer(v)}**' if i == j else integer(v)
                     for j, v in enumerate(row) for i in [LABELS.index(label)]])
    return table(header, rows)


def render(artifacts: Path, previous: str | None) -> str:
    report = json.loads((artifacts / 'report.json').read_text(encoding='utf-8'))
    stats_path, test_path = artifacts / 'dataset_stats.json', artifacts / 'test_report.json'
    stats = json.loads(stats_path.read_text(encoding='utf-8')) if stats_path.is_file() else None
    test = json.loads(test_path.read_text(encoding='utf-8')) if test_path.is_file() else None
    finalists = report['finalists']
    selected = report['selected']
    best = report['best_by_mean']
    chosen = finalists[selected['model']]
    order = [m for m in ('naive_bayes', 'logistic_regression', 'knn') if m in finalists]
    out = []
    add = out.append

    add('# Comparación de modelos: Naive Bayes, Regresión Logística y KNN\n')
    add(f'> Documento generado con `python -m src.models.render_comparison_report {artifacts.relative_to(ROOT)}` '
        f'a partir de `report.json` (SHA-256 `{digest(artifacts / "report.json")[:12]}…`). '
        'No editar las cifras a mano; la sección 12 se redacta a mano y se conserva al regenerar.\n')

    # 1. Resumen
    add('## 1. Resumen de la decisión\n')
    within = [m for m in order if finalists[m]['best_by_mean']['f1_macro_mean'] >= report['threshold'] - 1e-12]
    add(f"- **Modelo elegido:** {NAMES[selected['model']]} ({config_text(selected)}).")
    add(f"- **F1 macro en validación cruzada:** {num(selected['f1_macro_mean'])} ± {num(selected['f1_macro_std'])} "
        f"(3 particiones estratificadas, semilla {report['seed']}).")
    add(f"- **Mejor media absoluta:** {NAMES[best['model']]} ({config_text(best)}) con {num(best['f1_macro_mean'])} "
        f"± {num(best['f1_macro_std'])}. Umbral de la regla de una desviación estándar: **{num(report['threshold'])}**.")
    add(f"- **Modelos cuyo mejor resultado queda dentro del umbral:** {', '.join(NAMES[m] for m in within) or 'ninguno'}. "
        'Entre configuraciones estadísticamente equivalentes se elige la más simple.')
    add(f"- **Validación SID:** F1 macro {num(chosen['validation']['f1_macro'])}, exactitud "
        f"{pct(chosen['validation']['accuracy'])}. `test` {'se evaluó una sola vez después de la selección (sección 10)' if test else 'no se usó'}.\n")

    # 2. Datos
    add('## 2. Datos usados\n')
    if stats:
        splits = stats['splits']
        rows = []
        use = {'train': 'Validación cruzada y ajuste final', 'validation': 'Evaluación complementaria',
               'validation_real': 'Evaluación complementaria', 'test': 'Reservado: evaluación única final'}
        for name, split in splits.items():
            words = split.get('words')
            rows.append([f'`{name}`', integer(split['records']), integer(split['class_counts']['Marketing']),
                         f"{num(words['mean'], 2)} / {num(words['median'], 0)} / {words['max']}" if words else '— (no se procesan textos)',
                         use.get(name, '')])
        add(table(['Partición', 'Registros', 'Marketing', 'Palabras (media / mediana / máx.)', 'Uso'], rows, 'lrrll'))
        train = splits['train']
        add('\nDistribución de clases en entrenamiento:\n')
        add(table(['Área', 'Registros', 'Porcentaje'],
                  [[l, integer(train['class_counts'][l]), pct(train['class_share'][l])] for l in LABELS]))
        marketing = train['class_share']['Marketing']
        add(f"\n**Desbalance:** Marketing representa {pct(marketing)} del entrenamiento. Un clasificador que "
            f"nunca predijera Marketing seguiría alcanzando hasta {pct(1 - marketing)} de exactitud. Por eso la "
            'exactitud se informa, pero el criterio de selección es **F1 macro**, que da el mismo peso a las seis áreas.\n')
        add(f"**Textos cortos:** la mediana es de {num(train['words']['median'], 0)} palabras por mensaje; tras el "
            'preprocesamiento quedan aún menos términos, de modo que los vectores TF-IDF son muy dispersos.\n')
        if 'preprocessing' in stats:
            p = stats['preprocessing']
            add(f"Con el modelo elegido ({METHODS[p['method']]}), cada mensaje de entrenamiento conserva en promedio "
                f"{num(p['tokens_per_message_train']['mean'], 2)} términos y el vocabulario TF-IDF tiene "
                f"{integer(p['vocabulary_size'])} términos.\n")
    else:
        add('_Falta `dataset_stats.json`; ejecutar `python -m src.data.dataset_stats`._\n')

    # 3. Metodología
    add('## 3. Metodología común\n')
    add('- Los tres modelos usan exactamente el mismo preprocesamiento por variante (normalización, tokenización, '
        'stop words en español con negaciones conservadas, stemming o lematización) y TF-IDF con unigramas o '
        'unigramas + bigramas.')
    add(f"- Validación cruzada estratificada de {report['folds']} particiones sobre `train`, semilla {report['seed']}; "
        'las mismas particiones para todas las configuraciones, lo que permite comparar por partición (diferencias pareadas).')
    add('- TF-IDF se ajusta **dentro** de cada partición: el vocabulario de validación no se filtra al entrenamiento.')
    add(f"- Se evaluaron **{report['n_configurations']} configuraciones**: " +
        ', '.join(f"{NAMES[m]} {len(report['grid'][m]) * 4}" for m in order) + '.')
    add(f"- **Regla de selección (fijada antes de ver resultados):** {report['selection']}")
    add('- `test` no interviene en la selección. Los conjuntos humanos y simulados tampoco.\n')

    # 4. Modelos explicados
    add('## 4. Los tres modelos en lenguaje simple\n')
    add('**Naive Bayes multinomial.** Estima, para cada área, con qué frecuencia aparece cada término y combina esas '
        'frecuencias suponiendo que los términos son independientes. `alpha` suaviza las frecuencias para que un término '
        'nunca visto en un área no anule su probabilidad. Entrena en una sola pasada por los datos y su modelo es una tabla '
        'de pesos por área y término.\n')
    add('**Regresión Logística.** Aprende un peso por término y área ajustándolos para separar las clases. `C` controla '
        'la regularización (menor `C` = pesos más pequeños = modelo más simple); `class_weight=balanced` da más '
        'importancia a las clases pequeñas como Marketing. Suele dar probabilidades más informativas que NB, pero su '
        'entrenamiento es iterativo y más costoso.\n')
    add('**KNN (k vecinos más cercanos).** No aprende parámetros: guarda todos los mensajes de entrenamiento y, para '
        'clasificar uno nuevo, busca los `k` más parecidos (distancia coseno entre vectores TF-IDF) y vota. '
        '`weights=distance` da más voto a los vecinos más cercanos. Su «probabilidad» es la fracción de vecinos de '
        'cada área. Es el más sensible a la presencia de frases casi idénticas entre entrenamiento y evaluación, y el '
        'más costoso al predecir porque compara contra todo el entrenamiento.\n')

    # 5. Resultados CV
    add('## 5. Resultados de validación cruzada\n')
    rows = []
    for m in order:
        f = finalists[m]
        b = f['best_by_mean']
        rows.append([f"{'**' if f['is_selected'] else ''}{NAMES[m]}{' (elegido)**' if f['is_selected'] else ''}",
                     config_text(b), f"{num(b['f1_macro_mean'])} ± {num(b['f1_macro_std'])}",
                     'sí' if b['f1_macro_mean'] >= report['threshold'] - 1e-12 else 'no'])
    add(table(['Modelo', 'Mejor configuración por media', 'F1 macro CV', '¿Dentro del umbral?'], rows, 'llrl'))
    if selected != best:
        add(f"\nConfiguración desplegada de {NAMES[selected['model']]} según la regla: {config_text(selected)}, "
            f"F1 macro {num(selected['f1_macro_mean'])} ± {num(selected['f1_macro_std'])}.")
    add('\n**Diferencias pareadas por partición** (mejor de cada modelo, mismas particiones):\n')
    add(table(['Par (A − B)', 'Diferencias por partición', 'Media', 'Desviación'],
              [[f"{NAMES[p['a']]} − {NAMES[p['b']]}", ', '.join(num(d) for d in p['fold_differences_a_minus_b']),
                num(p['mean_difference']), num(p['std_difference'])] for p in report['pairwise_comparisons']], 'llrr'))
    add('\n**Efecto del preprocesamiento** (mejor F1 macro CV por variante):\n')
    variant_rows = []
    for m in order:
        cells = {(v['method'], v['ngrams']): v['f1_macro_mean'] for v in report['variants'] if v['model'] == m}
        variant_rows.append([NAMES[m]] + [num(cells.get((meth, n))) for meth in ('stemming', 'lemmatization') for n in (1, 2)])
    add(table(['Modelo', 'Stemming · 1', 'Stemming · 1-2', 'Lematización · 1', 'Lematización · 1-2'], variant_rows))
    add('')

    # 6. Métricas por clase
    add('## 6. Métricas por clase y matrices de confusión\n')
    add('La matriz de confusión fuera de partición reúne las predicciones de las tres particiones del CV: cada mensaje '
        f"de entrenamiento ({integer(chosen['cv']['out_of_fold']['records'])}) se predice con un modelo que no lo vio. "
        'Es mucho más estable que la de validación, que tiene muy pocos ejemplos de Marketing. Filas = área real; '
        'columnas = área predicha; la diagonal (negrita) son aciertos.\n')
    summary = []
    for m in order:
        f = finalists[m]
        oof = f['cv']['out_of_fold']
        summary.append([NAMES[m], num(oof['f1_macro']), pct(oof['accuracy']), num(oof['balanced_accuracy']),
                        num(f['validation']['f1_macro']), pct(f['validation']['accuracy']),
                        num(f.get('validation_real', {}).get('f1_macro')), num(f.get('simulated', {}).get('f1_macro'))])
    add(table(['Modelo', 'F1 macro OOF', 'Exactitud OOF', 'Exactitud balanceada OOF', 'F1 macro val.',
               'Exactitud val.', 'F1 macro val. adicional', 'F1 macro simulado'], summary))
    add('')
    for m in order:
        f = finalists[m]
        add(f"### {NAMES[m]} — {config_text(f['config'])}{' (elegido)' if f['is_selected'] else ''}\n")
        add('**Por clase, fuera de partición (CV):**\n')
        add(per_class(f['cv']['out_of_fold']))
        add('\n**Matriz de confusión, fuera de partición (CV):**\n')
        add(matrix(f['cv']['out_of_fold']))
        add('\n**Por clase, validación SID:**\n')
        add(per_class(f['validation']))
        add('')

    # 7. Sensibilidad agrupada
    add('## 7. Sensibilidad: CV agrupado por intención SID\n')
    grouped = report.get('grouped_sensitivity')
    if grouped and 'finalists' in grouped:
        add(f"Cada partición deja fuera intenciones completas ({grouped['n_groups']} intenciones): el modelo debe "
            'clasificar formulaciones de solicitudes que nunca vio. Es una prueba mucho más exigente que el CV principal '
            'y sirve para detectar si un modelo depende de memorizar plantillas. No interviene en la selección.\n')
        rows = []
        for g in grouped['finalists']:
            normal = g['f1_macro_mean']
            grp = g['grouped_f1_macro_mean']
            rows.append([NAMES[g['model']], config_text(g), num(normal), num(grp),
                         num(grp - normal) if grp is not None else '—'])
        add(table(['Modelo', 'Configuración', 'F1 macro CV', 'F1 macro agrupado', 'Diferencia'], rows, 'llrrr'))
        add('')
    else:
        add('_No se ejecutó (`--grouped-cv`)._\n')

    # 8. Costo
    add('## 8. Costo computacional\n')
    add(table(['Modelo', 'Entrenamiento (s)', 'Predicción 1000 msg., pipeline (ms)', 'Predicción 1000 msg., solo clasificador (ms)',
               'Tamaño del artefacto', 'Vocabulario'],
              [[NAMES[m], num(finalists[m]['cost']['fit_seconds'], 2),
                num(finalists[m]['cost']['predict_ms_per_1000_pipeline'], 1),
                num(finalists[m]['cost']['predict_ms_per_1000_classifier'], 1),
                f"{num(finalists[m]['cost']['artifact_bytes'] / 1024 / 1024, 2)} MB",
                integer(finalists[m]['vocabulary_size'])] for m in order]))
    add('\nEl tiempo del pipeline incluye la tokenización (con lematización, spaCy domina el costo). Las cifras se '
        'midieron en el equipo que ejecutó la comparación y sirven para comparar modelos entre sí, no como garantía.\n')

    # 9. Decisión
    add('## 9. Decisión y justificación\n')
    add(f"1. La mejor media de F1 macro es {num(best['f1_macro_mean'])} ({NAMES[best['model']]}), con desviación "
        f"{num(best['f1_macro_std'])}; el umbral de equivalencia es {num(report['threshold'])}.")
    add(f"2. Entre las configuraciones con media ≥ umbral, la regla elige la más simple: **{NAMES[selected['model']]} "
        f"({config_text(selected)})**, con {num(selected['f1_macro_mean'])}.")
    for m in order:
        if m == selected['model']:
            continue
        b = finalists[m]['best_by_mean']
        gap = selected['f1_macro_mean'] - b['f1_macro_mean']
        reason = ('queda dentro del umbral, pero es un modelo más complejo según la regla' if b['f1_macro_mean'] >= report['threshold'] - 1e-12
                  else 'queda por debajo del umbral')
        add(f"3. {NAMES[m]}: mejor media {num(b['f1_macro_mean'])} (diferencia con el elegido {num(gap)}); {reason}.")
    add('4. Los criterios secundarios (sensibilidad agrupada, costo, tamaño) se informan para explicar la decisión, '
        'pero no la cambian: la regla se fijó antes de ver los resultados.\n')

    # 10. Test y otras evaluaciones
    add('## 10. Evaluación final en `test` y evaluaciones exploratorias\n')
    if test:
        add(f"Evaluación **única**, posterior a la selección (`test_report.json`, {integer(test['records'])} mensajes). "
            'No se usó para elegir ni ajustar.\n')
        add(table(['Modelo', 'F1 macro', 'Exactitud', 'Exactitud balanceada', 'F1 Marketing'],
                  [[NAMES[m] + (' (elegido)' if test['finalists'][m]['is_selected'] else ''),
                    num(test['finalists'][m]['f1_macro']), pct(test['finalists'][m]['accuracy']),
                    num(test['finalists'][m]['balanced_accuracy']),
                    num(test['finalists'][m]['classification_report']['Marketing']['f1-score'])] for m in order if m in test['finalists']]))
        add('\n**Por clase, modelo elegido en `test`:**\n')
        add(per_class(test['finalists'][selected['model']]))
        add('\n**Matriz de confusión, modelo elegido en `test`:**\n')
        add(matrix(test['finalists'][selected['model']]))
        add('')
    else:
        add('`test` sigue reservado: la evaluación final está pendiente.\n')
    if all('simulated' in finalists[m] for m in order):
        add('**Conjunto simulado con IA (24 mensajes, exploratorio):** ' + '; '.join(
            f"{NAMES[m]} acierta {integer(round(finalists[m]['simulated']['accuracy'] * finalists[m]['simulated']['records']))}"
            f" de {finalists[m]['simulated']['records']}" for m in order) + '. Son pocos mensajes y no son humanos.\n')
    add('La evaluación humana de clientes requiere archivos privados que no están en este equipo; debe ejecutarse '
        'con el modelo elegido **después** de la selección y nunca para elegir.\n')

    # 11. Limitaciones
    add('## 11. Limitaciones\n')
    add('- SID es mayormente sintético: hay variantes casi idénticas entre particiones que inflan el CV, sobre todo para KNN.')
    add('- La validación SID tiene muy pocos ejemplos de Marketing; sus métricas por clase son inestables.')
    add('- Las probabilidades no están calibradas; en KNN son fracciones de vecinos (valores discretos).')
    add('- Las diferencias entre los mejores modelos son del orden de la desviación entre particiones: con otros datos el orden podría cambiar.\n')

    # 12. Interpretación manual
    add('## 12. Interpretación del equipo\n')
    interpretation = DEFAULT_INTERPRETATION
    if previous and START in previous and END in previous:
        interpretation = previous.split(START, 1)[1].split(END, 1)[0].strip()
    add(START)
    add(interpretation)
    add(END + '\n')
    return '\n'.join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path, nargs='?', default=ROOT / 'artifacts/comparison_customers_v3')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/COMPARACION_MODELOS.md')
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    previous = args.output.read_text(encoding='utf-8') if args.output.is_file() else None
    args.output.write_text(render(artifacts, previous), encoding='utf-8')
    print(f'Reporte: {args.output}')


if __name__ == '__main__':
    main()
