"""Run saved-artifact validation after relocation, without fitting any models."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import validate_outputs as validator


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('explicit_data', [False, True], ids=['repository-default', 'explicit-data'])
def test_relocated_validator(tmp_path, explicit_data):
    moved = tmp_path/'relocated repository'
    moved.mkdir()
    for name in ['validate_outputs.py', 'experiment.py']:
        shutil.copy2(ROOT/name, moved/name)
    for name in ['data', 'outputs_verified']:
        shutil.copytree(ROOT/name, moved/name)
    config = moved/'outputs_verified/research_configuration.json'
    provenance_before = config.read_bytes()
    original_path = json.loads(provenance_before)['dataset_path']
    args = [str(moved/'validate_outputs.py')]
    if explicit_data:
        alternate = tmp_path/'custom input.csv'
        (moved/'data/kc_house_data.csv').rename(alternate)
        args += ['--data', str(alternate)]
    before = hashes(moved/'outputs_verified')
    # Fail if the validator attempts to read the historical absolute dataset path,
    # even though that path happens to exist on the developer's machine.
    bootstrap = '''import pathlib, runpy, sys
import pandas as pd
original = sys.argv.pop(1)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(script).parent))
read_csv = pd.read_csv
def guarded_read(path, *args, **kwargs):
    assert str(pathlib.Path(path).resolve()) != original, 'Historical dataset path accessed'
    return read_csv(path, *args, **kwargs)
pd.read_csv = guarded_read
runpy.run_path(script, run_name='__main__')
'''
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run([sys.executable, '-c', bootstrap, original_path, *args],
                               cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'PASS: dataset SHA256' in completed.stdout
    assert 'saved-model predictions' in completed.stdout
    assert 'Winner: M3' in completed.stdout
    assert hashes(moved/'outputs_verified') == before
    assert config.read_bytes() == provenance_before


def test_checksum_mismatch_rejected_before_model_loading(tmp_path, monkeypatch):
    invalid = tmp_path/'wrong.csv'
    invalid.write_bytes((ROOT/'data/kc_house_data.csv').read_bytes() + b'\n')
    def forbidden_load(*args, **kwargs):
        pytest.fail('Model loaded before rejecting the dataset checksum')
    monkeypatch.setattr(validator.joblib, 'load', forbidden_load)
    with pytest.raises(ValueError, match='Dataset SHA256 mismatch'):
        validator.validate(ROOT/'outputs_verified', invalid)
