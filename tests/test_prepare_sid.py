import hashlib
import io
import json
import sys

import pytest

import src.data.prepare_sid as sid


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def test_offline_file_present_missing_and_wrong_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(sid, 'RAW', tmp_path / 'raw')
    source = {'revision': 'abc123', 'source_url': 'https://example.invalid/dataset',
              'sha256': {'train.csv': hash_bytes(b'correcto')}}
    with pytest.raises(FileNotFoundError, match='No existe'):
        sid.download_file('train.csv', True, source)
    path = sid.RAW / 'train.csv'
    path.write_bytes(b'correcto')
    assert sid.download_file('train.csv', True, source) == path
    path.write_bytes(b'cambiado')
    with pytest.raises(ValueError, match='SHA-256 incorrecto'):
        sid.download_file('train.csv', True, source)
    assert path.read_bytes() == b'cambiado'


def test_download_uses_revision_and_verifies_before_install(tmp_path, monkeypatch):
    monkeypatch.setattr(sid, 'RAW', tmp_path / 'raw')
    source = {'revision': 'abc123', 'source_url': 'https://example.invalid/dataset',
              'sha256': {'train.csv': hash_bytes(b'correcto')}}
    calls = []
    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(b'correcto')
    monkeypatch.setattr(sid, 'urlopen', fake_urlopen)
    path = sid.download_file('train.csv', False, source)
    assert path.read_bytes() == b'correcto'
    assert calls == ['https://example.invalid/dataset/resolve/abc123/train.csv']
    path.unlink()
    monkeypatch.setattr(sid, 'urlopen', lambda request, timeout: io.BytesIO(b'alterado'))
    with pytest.raises(ValueError, match='SHA-256 incorrecto'):
        sid.download_file('train.csv', False, source)
    assert not path.exists()
    assert not path.with_suffix('.csv.part').exists()


def test_main_checks_all_sources_before_writing_processed(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    source = {'revision': 'abc123', 'source_url': 'https://example.invalid/dataset',
              'license_declared': 'MIT', 'sha256': {'train.csv': hash_bytes(b'correcto'),
                                                   'intents.json': hash_bytes(b'esperado')}}
    (raw / 'train.csv').write_bytes(b'correcto')
    (raw / 'intents.json').write_bytes(b'alterado')
    source_path = tmp_path / 'source.json'
    source_path.write_text(json.dumps(source), encoding='utf-8')
    processed = tmp_path / 'processed'
    processed.mkdir()
    sentinel = processed / 'existente.csv'
    sentinel.write_text('no modificar', encoding='utf-8')
    monkeypatch.setattr(sid, 'RAW', raw)
    monkeypatch.setattr(sid, 'OUT', processed)
    monkeypatch.setattr(sid, 'SOURCE_PATH', source_path)
    monkeypatch.setattr(sid, 'FILES', {'train': 'train.csv'})
    monkeypatch.setattr(sid, 'load_mapping', lambda: {})
    monkeypatch.setattr(sys, 'argv', ['prepare_sid.py', '--offline'])
    with pytest.raises(ValueError, match='SHA-256 incorrecto'):
        sid.main()
    assert list(processed.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding='utf-8') == 'no modificar'
