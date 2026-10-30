from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Exact original-row mechanism comparison at fixed ambient dimension.'
from pathlib import Path
from fractions import Fraction as F
import json, csv
B = Path(__file__).resolve().parent

def main():
    out = B / 'nuisance_scale_results.json'
    assert not out.exists()
    m = 64
    n = 128
    alpha = F(1, 32)
    lam = F(1, 2)
    weight = F(1, n)
    rows = []
    for scale in (1, 2, 4, 16, 64, 256):
        u = F(3, 4)
        t = F(16 * scale)
        beta0 = F(1)
        g0 = lam * beta0 + sum((weight * u * (u - 2) for _ in range(m)))
        assert g0 == F(1, 32)
        m00 = lam + sum((weight * u * u for _ in range(m))) + sum((weight * alpha * alpha for _ in range(m)))
        diag = lam + weight * t * t
        cross = weight * alpha * t
        schur = m00 - sum((cross * cross / diag for _ in range(m)))
        actual_h = g0 * g0 / schur
        actual_v = g0 * g0 / m00
        h = 1 / (800 + 512 * alpha * alpha / F(1 + 4 * scale * scale))
        v = 1 / (800 + 512 * alpha * alpha)
        assert actual_h == h and actual_v == v
        gsup = F(1, 32)
        signal_curvature_min = lam + sum((weight * u * u for _ in range(m)))
        assert signal_curvature_min == F(25, 32)
        cross_squared = sum(((weight * alpha * t) ** 2 for _ in range(m)))
        assert cross_squared == alpha * alpha * scale * scale
        a2 = sum((weight * alpha * alpha for _ in range(m)))
        assert a2 == alpha * alpha / 2
        root = gsup / signal_curvature_min
        assert root == F(1, 25)
        energy = a2 * root * root
        scalar = cross_squared * root * root / lam
        u0 = gsup * gsup / lam
        moment_root_squared = u0 / lam
        moment = cross_squared * moment_root_squared / lam
        assert moment_root_squared == F(1, 256) and energy == F(1, 1280000)
        assert h - v <= energy and scalar == 4 * scale * scale * energy
        rows.append(dict(scale=scale, p=65, n=128, alpha=str(alpha), lambda_=str(lam), full_H=str(h), retained_V=str(v), exact_omitted=str(h - v), energy_tail=str(energy), scalar_same_radius_tail=str(scalar), moment_exact_constants_tail=str(moment), retained_radius=str(root), moment_radius='1/16', cross_squared=str(cross_squared), energy_float=float(energy), scalar_float=float(scalar), moment_float=float(moment), omitted_float=float(h - v)))
    payload = dict(status='PASS', protocol='nuisance_scale_protocol.md', arithmetic='exact Fraction', dimension_fixed=True, normal_equation_attainment_checks=6, scales=rows)
    out.write_text(json.dumps(payload, indent=2))
    with (B / 'nuisance_scale_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(dict(status='PASS', conditions=len(rows), energy_tail=float(energy), last_scalar=float(scalar), last_moment=float(moment))))
if __name__ == '__main__':
    main()
