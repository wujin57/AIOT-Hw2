"""Package source, approved data, tests and real outputs; exclude environment/caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root=Path(__file__).resolve().parent
archive=root/'7115056123_hw2.zip'
excluded={'.venv','__pycache__','.pytest_cache','.git','.DS_Store','outputs'}
with ZipFile(archive,'w',compression=ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if p.is_file() and not any(s in excluded for s in rel.parts) and p.suffix!='.zip':
            z.write(p,Path('hw2')/rel)
with ZipFile(archive) as z:
    assert z.testzip() is None
    print(f'{archive}: {len(z.namelist())} files, {archive.stat().st_size:,} bytes; CRC check passed')
