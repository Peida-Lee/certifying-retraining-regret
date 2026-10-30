"""Fixed rank-three baseline for the six component-control inputs."""
from pathlib import Path
import json,time,traceback
from public_problems import problem
from observed_anchor import ObservedAnchor
from moment_tail import MomentSearch
from dual_certificate import reserve_cores

def main():
    reserve_cores();out=Path(__file__).resolve().parent/'component_base_results.json'
    assert not out.exists();records=[]
    for name in ('diabetes','concrete','superconductivity'):
      for pattern in ('concentrated','dispersed'):
        r={'dataset':name,'spread':pattern}
        try:
            args,meta=problem(name=name,spread=pattern);tick=time.monotonic()
            a=ObservedAnchor(*args,anchor='observed',preconditioner='midpoint')
            result=MomentSearch(a,3,kind='gradient',search_bound='adaptive').run(time_limit=1,absolute_tolerance=1e-6)
            r.update(meta,status='PASS',result=result,total_seconds=time.monotonic()-tick)
        except Exception as e:r.update(status='ERROR',error=str(e),traceback=traceback.format_exc())
        records.append(r)
    out.write_text(json.dumps({'status':'PASS' if all(r['status']=='PASS' for r in records) else 'COMPLETED_WITH_ERRORS','results':records},indent=2))

if __name__=='__main__':main()
