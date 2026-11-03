import csv
import json

import pytest

from src.data.dataset_stats import build
from src.models.train import LABELS, ROOT

V3 = ROOT / 'artifacts/comparison_customers_v3'


def write(path, rows, fields=('text', 'department')):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_stats_contain_only_aggregates(tmp_path):
    secret = 'textosecretoquenodebeaparecer'
    rows = [{'text': f'{secret} {label} uno dos', 'department': label} for label in LABELS]
    for split in ('train', 'validation', 'test'):
        write(tmp_path / f'{split}.csv', rows)
    stats = build(tmp_path)
    dumped = json.dumps(stats, ensure_ascii=False)
    assert secret not in dumped
    assert stats['splits']['train']['records'] == 6
    assert stats['splits']['train']['words']['median'] == 4
    assert sum(b['count'] for b in stats['splits']['train']['words_histogram']) == 6
    assert 'words' not in stats['splits']['test']  # de test solo se cuentan etiquetas


def test_dataset_stats_reads_only_department_from_test(tmp_path):
    rows = [{'text': 'hola', 'department': label} for label in LABELS]
    write(tmp_path / 'train.csv', rows)
    write(tmp_path / 'test.csv', [{'department': label} for label in LABELS], fields=('department',))
    assert build(tmp_path)['splits']['test']['class_counts']['Marketing'] == 1


def test_evaluate_test_refuses_second_run(tmp_path):
    from src.models.evaluate_test import run
    (tmp_path / 'test_report.json').write_text('{}', encoding='utf-8')
    with pytest.raises(SystemExit, match='una sola vez'):
        run(tmp_path, tmp_path / 'test.csv')


@pytest.mark.skipif(not (V3 / 'report.json').is_file(), reason='requiere artefactos v3 locales')
def test_v3_report_has_three_finalists_and_renders():
    from src.models.render_comparison_report import START, END, render
    report = json.loads((V3 / 'report.json').read_text(encoding='utf-8'))
    assert set(report['finalists']) == {'naive_bayes', 'logistic_regression', 'knn'}
    assert report['test_used'] is False and report['n_configurations'] == 120
    text = render(V3, f'{START}\nnota del equipo\n{END}')
    assert 'nota del equipo' in text and 'KNN' in text
