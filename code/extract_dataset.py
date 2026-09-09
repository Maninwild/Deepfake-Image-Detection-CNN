"""Extract the bundled dataset archives into the project folder."""
from pathlib import Path
from zipfile import ZipFile

root = Path(__file__).resolve().parent.parent
parts = sorted((root / 'dataset_parts').glob('*.zip'))
if not parts:
    raise SystemExit('Missing dataset_parts ZIP files')
for part in parts:
    with ZipFile(part) as archive:
        for name in archive.namelist():
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError('Invalid archive path')
        archive.extractall(root)
    print('Extracted', part.name)
