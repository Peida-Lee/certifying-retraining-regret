"""Independent support-normal polygon oracle for exact dense-row audits."""
from fractions import Fraction as Q

def independent_rowmax(a, b, ga, gb):
    g = [(x, y) for x, y in zip(ga, gb) if x or y]
    if not g:
        return a * a - b * b
    vertices = set()
    for x, y in g:
        for side in (-1, 1):
            normal = (-side * y, side * x)
            for tie in (-1, 1):
                signs = []
                for u, v in g:
                    cross = normal[0] * u + normal[1] * v
                    if cross:
                        signs.append(1 if cross > 0 else -1)
                    else:
                        signs.append(tie * (1 if x * u + y * v > 0 else -1))
                vertices.add((a + sum((u * s for (u, v), s in zip(g, signs)), Q(0)), b + sum((v * s for (u, v), s in zip(g, signs)), Q(0))))
    points = sorted(vertices)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        out = []
        for point in seq:
            while len(out) > 1 and cross(out[-2], out[-1], point) <= 0:
                out.pop()
            out.append(point)
        return out
    hull = half(points)[:-1] + half(points[::-1])[:-1]
    if len(points) == 1:
        hull = points
    best = None
    for i, (u, v) in enumerate(hull):
        uu, vv = hull[(i + 1) % len(hull)]
        dx = uu - u
        dy = vv - v
        aa = dx * dx - dy * dy
        bb = 2 * (u * dx - v * dy)
        cc = u * u - v * v
        values = [cc, aa + bb + cc]
        if aa < 0 and 0 < -bb / (2 * aa) < 1:
            values.append(cc - bb * bb / (4 * aa))
        val = max(values)
        best = val if best is None else max(best, val)
    return best
