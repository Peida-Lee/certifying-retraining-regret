from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Independent exact equality of original loss and compressed SCIP objective.'
from pathlib import Path
from fractions import Fraction as F
import sys, json, itertools, time
import numpy as np, psutil
B = Path(__file__).resolve().parent
OLD = MODULE_DIR
from audit_real_world import read, mq, vq, dot, tr, mm, quad, exact_original_loss

def main():
    psutil.Process().cpu_affinity([12])
    source = json.loads((OLD / 'confirmation_results.json').read_text())
    reports = []
    start = time.monotonic()
    for name in ('concrete', 'superconductivity', 'airfoil', 'powerplant'):
        r = next((r for r in source['results'] if r['dataset'] == name and r['seed'] == 84111 and (r['method'] == 'energy')))
        cert = json.loads((OLD / r['certificate_file']).read_text())['certificate']
        D = read(cert)
        T = mq(cert['T'])
        K = mm(tr(T), mm(D['A'], T))
        h = [dot(col, [dot(row, D['beta']) - v for row, v in zip(D['A'], D['r'])]) for col in tr(T)]
        errors = []
        for case in range(2):
            z = [F((-1) ** (j + case), 1024 * (j + 1)) for j in range(D['p'])]
            x = [row.copy() for row in D['c']]
            for i in D['unknown']:
                for j, w in enumerate(D['w'][i]):
                    if w:
                        x[i][j] += w * F((-1) ** (i + j + case), 2)
            theta = [v - dot(row, z) for v, row in zip(D['beta'], T)]
            direct = exact_original_loss(x, D['y'], D['beta'], D['ridge']) - exact_original_loss(x, D['y'], theta, D['ridge'])
            compressed = 2 * dot(h, z) - quad(z, K) + sum(((dot(x[i], D['beta']) - D['y'][i]) ** 2 - (dot(x[i], theta) - D['y'][i]) ** 2 for i in D['unknown']), F(0)) / D['n']
            assert direct == compressed
            tf = np.array(cert['T'])
            Af = np.array(D['A'], float)
            Kf = tf.T @ Af @ tf
            hf = np.array(h, float)
            zf = np.array(z, float)
            thf = np.array(theta, float)
            bf = np.array(D['beta'], float)
            numeric = 2 * hf @ zf - zf @ Kf @ zf + sum((float(np.array(x[i], float) @ bf - D['y'][i]) ** 2 - float(np.array(x[i], float) @ thf - D['y'][i]) ** 2 for i in D['unknown'])) / D['n']
            errors.append(abs(float(direct) - numeric))
        assert max(errors) < 1e-12
        reports.append(dict(dataset=name, exact_original_identity=True, points=2, max_numeric_coefficient_objective_difference=max(errors)))
    out = dict(status='PASS', source_inputs='four preselected v84first masks', imports_scip_builder=False, points=8, reports=reports, seconds=time.monotonic() - start)
    (B / 'scip_objective_audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
if __name__ == '__main__':
    main()
