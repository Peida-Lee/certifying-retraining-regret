"""Official public inputs for the four real-world contraction experiments."""
from pathlib import Path
import urllib.request,zipfile,json,traceback
from project_paths import MODULE_DIR
B=MODULE_DIR/'data'
B.mkdir(exist_ok=True)
sources=[('parkinsons',189,'parkinsons+telemonitoring'),('wine',186,'wine+quality'),('yacht',243,'yacht+hydrodynamics')]
records=[]
for name,uid,slug in sources:
    row=dict(dataset=name,url=f'https://archive.ics.uci.edu/static/public/{uid}/{slug}.zip')
    try:
        archive=B/(name+'.zip')
        if archive.exists():row['download']='REUSED'
        else:
            with urllib.request.urlopen(row['url'],timeout=60) as response:payload=response.read()
            archive.write_bytes(payload);assert archive.read_bytes()==payload
            row.update(download='OK',complete_transfer_checks=1,bytes=len(payload))
        target=(B/name).resolve();target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():continue
                path=(target/info.filename).resolve();assert path.is_relative_to(target)
                if path.exists():continue
                path.parent.mkdir(parents=True,exist_ok=True);payload=z.read(info)
                path.write_bytes(payload);assert path.read_bytes()==payload
        row['status']='PASS'
    except Exception as e:row.update(status='ERROR',error=str(e),traceback=traceback.format_exc())
    records.append(row)
    (B/'contraction_download_receipts.json').write_text(json.dumps(records,indent=2))
    print(name,row['status'],flush=True)
