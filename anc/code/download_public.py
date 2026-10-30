from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Reproducible official downloads; existing local archives are reused.'
from pathlib import Path
import urllib.request, json
BASE = DATA_CORE
BASE.mkdir(exist_ok=True)
urls = {'concrete': 'https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip', 'superconductivity': 'https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip'}
receipts = []
for name, url in urls.items():
    target = BASE / (name + '.zip')
    if target.exists():
        receipts.append({'dataset': name, 'status': 'REUSED', 'url': url})
        continue
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = response.read()
    target.write_bytes(payload)
    assert target.read_bytes() == payload
    receipts.append({'dataset': name, 'status': 'DOWNLOADED', 'url': url, 'bytes': len(payload), 'complete_comparison_once': True})
(BASE / 'download_receipt.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')
print(json.dumps(receipts))
