from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Extract each new public archive with one full extraction comparison.'
from pathlib import Path
import zipfile, json
BASE = DATA_CORE
receipts = []
for name in ('concrete', 'superconductivity'):
    out = BASE / name
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    with zipfile.ZipFile(BASE / (name + '.zip')) as z:
        names = z.namelist()
        for member in names:
            destination = (out / member).resolve()
            assert destination.is_relative_to(out.resolve())
        z.extractall(out)
        comparisons = []
        for member in names:
            if member.endswith('/'):
                continue
            data = z.read(member)
            assert (out / member).read_bytes() == data
            comparisons.append({'name': member, 'bytes': len(data), 'same': True})
    receipts.append({'dataset': name, 'complete_extraction_check_once': True, 'files': comparisons})
(BASE / 'data_receipt.json').write_text(json.dumps({'status': 'PASS', 'archives': receipts}, indent=2), encoding='utf-8')
print(json.dumps({'status': 'PASS', 'datasets': receipts}))
