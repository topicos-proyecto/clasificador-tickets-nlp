import csv
import hashlib

import pytest

import src.data.build_customer_splits as module
from src.data.build_splits import COLUMNS


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_new_split_adds_approved_drafts_and_preserves_evaluation(tmp_path, monkeypatch):
    base, out, draft = tmp_path / 'base', tmp_path / 'new', tmp_path / 'draft.csv'
    base.mkdir()
    for name in ('train', 'validation', 'validation_real', 'test'):
        row = dict(text='mensaje ' + hashlib.sha256(name.encode()).hexdigest(),
                   department='Ventas', source='SID', intent='consulta', language='es', original_split=name)
        write_csv(base / f'{name}.csv', COLUMNS, [row])
    rows = []
    for i in range(30):
        token = hashlib.sha256(str(i).encode()).hexdigest()
        rows.append(dict(text='consulta prueba ' + token, department='Marketing' if i < 15 else 'Soporte TI',
                         source='borrador_ia', intent='consulta', language='es'))
    write_csv(draft, COLUMNS[:-1], rows)
    monkeypatch.setattr(module, 'BASE', base)
    monkeypatch.setattr(module, 'DRAFT', draft)
    monkeypatch.setattr(module, 'OUT', out)
    module.main()
    with (out / 'train.csv').open(encoding='utf-8', newline='') as f:
        trained = list(csv.DictReader(f))
    assert len(trained) == 31
    assert sum(r['original_split'] == 'customer_draft_train' for r in trained) == 30
    for name in ('validation', 'validation_real', 'test'):
        with (base / f'{name}.csv').open(encoding='utf-8', newline='') as a, (out / f'{name}.csv').open(encoding='utf-8', newline='') as b:
            assert list(csv.DictReader(a)) == list(csv.DictReader(b))
    with pytest.raises(SystemExit, match='no se sobrescribió'):
        module.main()
