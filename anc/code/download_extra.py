from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Official UCI files; one comparison for each new transfer/member.'
from pathlib import Path
import urllib.request, zipfile, io, json, traceback
B = DATA_EXTRA
B.mkdir(exist_ok=True)
SOURCES = {'airfoil': ('291', 'airfoil+self+noise'), 'powerplant': ('294', 'combined+cycle+power+plant')}
records = []
for name, (uid, slug) in SOURCES.items():
    target = B / (name + '.zip')
    url = f'https://archive.ics.uci.edu/static/public/{uid}/{slug}.zip'
    row = dict(dataset=name, url=url)
    try:
        if target.exists():
            row['download'] = 'REUSED'
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            target.write_bytes(payload)
            assert target.read_bytes() == payload
            row.update(download='OK', bytes=len(payload), transfer_comparison_count=1)
        with zipfile.ZipFile(target) as z:
            row['members'] = z.namelist()
            root = (B / name).resolve()
            root.mkdir(exist_ok=True)
            row['extracted'] = []
            for info in z.infolist():
                if info.is_dir():
                    continue
                out = (root / info.filename).resolve()
                if not out.is_relative_to(root):
                    raise ValueError('Unsafe ZIP path')
                if out.exists():
                    row['extracted'].append(dict(name=info.filename, status='REUSED'))
                    continue
                out.parent.mkdir(exist_ok=True, parents=True)
                payload = z.read(info)
                out.write_bytes(payload)
                assert out.read_bytes() == payload
                row['extracted'].append(dict(name=info.filename, status='OK', transfer_comparison_count=1))
        row['status'] = 'PASS'
    except Exception as e:
        row.update(status='ERROR', error=str(e), traceback=traceback.format_exc())
    records.append(row)
    (B / 'download_receipts.json').write_text(json.dumps(records, indent=2))
    print(name, row['status'], row.get('members', row.get('error')), flush=True)
