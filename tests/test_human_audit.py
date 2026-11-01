import csv
import pytest

from src.data.audit_human_eval import audit
from src.data.validate_custom import COLUMNS


def write(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def record(text, source='propio_humano'):
    return dict(text=text, department='Logística', source=source, intent='envio', language='es')


def test_audit_detects_similarities_without_revealing_reference_text(tmp_path):
    ref = tmp_path / 'train.csv'
    human = tmp_path / 'human.csv'
    write(ref, ['text'], [{'text': 'Mi paquete se encuentra retrasado'}])
    write(human, COLUMNS, [record('Mi paquete se encuentra retrasado!')])
    result = audit(human, [ref])
    assert result['lineas_con_similitudes'] == [{'linea': 2, 'coincidencias': [
        {'particion': 'train.csv', 'tipo': 'exacta_normalizada'}]}]
    assert result['por_area']['Logística'] == 1
    assert 'Marketing' in result['areas_sin_ejemplos']
    assert 'Mi paquete' not in str(result)


def test_rejects_empty_and_nonhuman_inputs(tmp_path):
    ref = tmp_path / 'train.csv'
    human = tmp_path / 'human.csv'
    write(ref, ['text'], [{'text': 'otro texto'}])
    write(human, COLUMNS, [])
    with pytest.raises(ValueError, match='vacío'):
        audit(human, [ref])
    write(human, COLUMNS, [record('Mi paquete sigue retrasado', 'borrador_ia')])
    with pytest.raises(ValueError, match='propio_humano'):
        audit(human, [ref])
