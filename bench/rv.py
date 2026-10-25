"""RV tables: {name: (dist 'LN'|'N', mean, cov, source)}.  draw / shift helpers."""
import numpy as np
def defaults(RV): return {k: v[1] for k, v in RV.items()}
def draw(RV, rng):
    out = {}
    for k, (d, m, c, _) in RV.items():
        z = rng.standard_normal()
        if d == 'LN':
            s = np.sqrt(np.log(1 + c * c)); out[k] = m * np.exp(-s * s / 2 + s * z)
        else: out[k] = m * (1 + c * z)
    return out
def shift(RV, signs):
    """+-1 sd on every RV (signs: list of +1/-1, same order as RV)."""
    return {k: v[1] * (1 + s * v[2]) for (k, v), s in zip(RV.items(), signs)}
