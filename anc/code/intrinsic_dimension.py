from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Exact two-completion proof check, generated from original fractional rows.'
from pathlib import Path
from fractions import Fraction as F
import json
B = Path(__file__).resolve().parent

def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))

def solve(M, b):
    a = [row.copy() + [v] for row, v in zip(M, b)]
    for j in range(len(b)):
        pivot = a[j][j]
        assert pivot > 0
        a[j] = [v / pivot for v in a[j]]
        for i in range(len(b)):
            if i != j:
                q = a[i][j]
                a[i] = [v - q * w for v, w in zip(a[i], a[j])]
    return [row[-1] for row in a]

def main():
    import psutil
    psutil.Process().cpu_affinity([10])
    lam = F(1, 4)
    alpha = F(1, 32)
    n, p = (64, 34)
    beta = [F(1), F(1)] + [F(0)] * 32
    vectors = []
    for active in (0, 1):
        x = []
        y = []
        for j in (0, 1):
            for i in range(16):
                row = [F(0)] * p
                row[j] = 1 - F(1, 2 ** (j + 2)) if j == active else F(1)
                x.append(row)
                y.append(F(2))
        for j in (0, 1):
            for i in range(16):
                row = [F(0)] * p
                row[j] = alpha
                row[2 + 16 * j + i] = F(8)
                x.append(row)
                y.append(alpha)
        M = [[lam * int(i == j) + sum((row[i] * row[j] for row in x), F(0)) / n for j in range(p)] for i in range(p)]
        g = [lam * beta[j] + sum((row[j] * (dot(row, beta) - yy) for row, yy in zip(x, y)), F(0)) / n for j in range(p)]
        u = solve(M, g)
        assert all((dot(row, u) == v for row, v in zip(M, g)))
        vectors.append(u)
    assert dot(*vectors) == 0
    s0, s1 = (F(320, 8001), F(80, 9041))
    assert vectors[0][0] == s0 and vectors[1][1] == s1
    A, C = [dot(v, v) for v in vectors]
    all_line_lower = lam * A * C / (A + C)
    conservative = lam * s0 * s0 * s1 * s1 / (s0 * s0 + s1 * s1)
    a2 = alpha * alpha / 2
    G2 = F(17, 65536)
    kappa = F(25, 64)
    R2 = G2 / kappa ** 2
    energy = a2 * R2
    assert energy == F(17, 20480000) and conservative == F(25600, 1371850897) and (all_line_lower == F(25604, 1371850897))
    assert energy < F(1, 100000) < conservative
    out = dict(status='PASS', original_normal_equations=True, orthogonal_displacements=True, active_components=[str(s0), str(s1)], squared_norms=[str(A), str(C)], a_squared=str(a2), G_squared=str(G2), kappa=str(kappa), radius_squared=str(R2), energy_upper=str(energy), conservative_all_line_lower=str(conservative), exact_two_point_lower=str(all_line_lower), epsilon='1/100000', intrinsic_dimension=2, solver_imported=False, source_public_data_opened=False)
    (B / 'intrinsic_dimension_audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
if __name__ == '__main__':
    main()
