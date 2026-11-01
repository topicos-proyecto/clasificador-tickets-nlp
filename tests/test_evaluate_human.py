import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import src.models.evaluate_human as module
from src.data.validate_custom import COLUMNS
from src.models.train import LABELS


class MultinomialNB:
    alpha = 0.5


class FakeModel:
    named_steps = {'tfidf': SimpleNamespace(ngram_range=(1, 1),
                                            tokenizer=SimpleNamespace(method='lemmatization')),
                   'classifier': MultinomialNB()}

    def predict(self, texts):
        return np.asarray(LABELS[:len(texts)])


def test_evaluate_uses_frozen_hashes_and_all_labels(tmp_path, monkeypatch):
    dataset, model, manifest = (tmp_path / n for n in ('data.csv', 'model.joblib', 'manifest.json'))
    with dataset.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for label in LABELS:
            writer.writerow(dict(text='solicitud ejemplo', department=label, source='propio_humano',
                                 intent='ejemplo', language='es'))
    model.write_bytes(b'modelo')
    manifest.write_text(json.dumps({'dataset_sha256': module.sha256(dataset),
                                    'model_sha256': module.sha256(model), 'model': {}}))
    monkeypatch.setattr(module.joblib, 'load', lambda path: FakeModel())
    result = module.evaluate(dataset, model, manifest)
    assert result['records'] == 6 and result['f1_macro'] == 1
    assert set(result['confusion_matrix_labels']) == set(LABELS)
    assert result['errors'] == []
    assert result == module.evaluate(dataset, model, manifest)
    original = dataset.read_bytes()
    dataset.write_text('cambio', encoding='utf-8')
    with pytest.raises(ValueError, match='huella'):
        module.evaluate(dataset, model, manifest)
    dataset.write_bytes(original)
    model.write_bytes(b'otro modelo')
    with pytest.raises(ValueError, match='huella'):
        module.evaluate(dataset, model, manifest)
