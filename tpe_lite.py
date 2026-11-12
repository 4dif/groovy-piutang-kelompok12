"""TPE (Tree-structured Parzen Estimator) ringan, meniru TPESampler default Optuna:
10 trial awal acak (startup), selanjutnya tiap parameter disampling univariat dengan
gamma(n) = min(ceil(0.1 n), 25) trial terbaik sebagai l(x), sisanya g(x), 24 kandidat,
dipilih kandidat dengan rasio l(x)/g(x) terbesar. Dipakai karena paket optuna tidak tersedia
di lingkungan eksekusi; skrip Optuna asli disertakan terpisah."""
import math
import numpy as np


class Int:
    def __init__(self, lo, hi, log=False): self.lo, self.hi, self.log = lo, hi, log
class Float:
    def __init__(self, lo, hi, log=False): self.lo, self.hi, self.log = lo, hi, log
class Cat:
    def __init__(self, choices): self.choices = list(choices)


def _to_u(sp, v):
    if sp.log: return math.log(v)
    return float(v)
def _bounds(sp):
    lo, hi = (math.log(sp.lo), math.log(sp.hi)) if sp.log else (float(sp.lo), float(sp.hi))
    if isinstance(sp, Int) and not sp.log: lo, hi = lo - 0.5, hi + 0.5
    return lo, hi
def _from_u(sp, u):
    v = math.exp(u) if sp.log else u
    if isinstance(sp, Int): return int(min(max(round(v), sp.lo), sp.hi))
    return float(min(max(v, sp.lo), sp.hi))


def _parzen(obs, lo, hi):
    """Mixture: prior (uniform-ish, center) + Gaussian per observasi; bandwidth ala Optuna (jarak tetangga)."""
    mus = np.array(sorted(list(obs) + [(lo + hi) / 2]))
    n = len(mus)
    sig = np.empty(n)
    for i in range(n):
        left = mus[i] - (mus[i - 1] if i > 0 else lo)
        right = (mus[i + 1] if i < n - 1 else hi) - mus[i]
        sig[i] = max(left, right)
    sig = np.clip(sig, (hi - lo) / min(100.0, 1.0 + n), hi - lo)
    w = np.ones(n) / n
    return mus, sig, w


def _logpdf(x, mus, sig, w, lo, hi):
    from scipy.stats import norm
    x = np.atleast_1d(x)[:, None]
    z = norm.cdf((hi - mus) / sig) - norm.cdf((lo - mus) / sig)
    p = (w / z) * norm.pdf((x - mus) / sig) / sig
    return np.log(p.sum(1) + 1e-300)


def _sample(rng, mus, sig, w, lo, hi, k):
    out = []
    while len(out) < k:
        i = rng.choice(len(mus), p=w); v = rng.normal(mus[i], sig[i])
        if lo <= v <= hi: out.append(v)
    return np.array(out)


class Study:
    def __init__(self, space, seed=42, n_startup=10, n_ei=24):
        self.space, self.rng, self.n_startup, self.n_ei = space, np.random.default_rng(seed), n_startup, n_ei
        self.trials = []  # (params, value)

    def _suggest(self):
        if len(self.trials) < self.n_startup:
            p = {}
            for k, sp in self.space.items():
                if isinstance(sp, Cat): p[k] = sp.choices[self.rng.integers(len(sp.choices))]
                else:
                    lo, hi = _bounds(sp); p[k] = _from_u(sp, self.rng.uniform(lo, hi))
            return p
        vals = np.array([v for _, v in self.trials]); n = len(vals)
        ng = min(math.ceil(0.1 * n), 25)
        order = np.argsort(-vals); good = set(order[:ng].tolist())
        p = {}
        for k, sp in self.space.items():
            G = [self.trials[i][0][k] for i in range(n) if i in good]
            Bd = [self.trials[i][0][k] for i in range(n) if i not in good]
            if isinstance(sp, Cat):
                cg = np.array([1 + G.count(c) for c in sp.choices], float); cg /= cg.sum()
                cb = np.array([1 + Bd.count(c) for c in sp.choices], float); cb /= cb.sum()
                cand = self.rng.choice(len(sp.choices), size=self.n_ei, p=cg)
                best = max(cand, key=lambda i: math.log(cg[i]) - math.log(cb[i]))
                p[k] = sp.choices[best]
            else:
                lo, hi = _bounds(sp)
                lg = _parzen([_to_u(sp, v) for v in G], lo, hi); lb = _parzen([_to_u(sp, v) for v in Bd], lo, hi)
                cand = _sample(self.rng, *lg, lo, hi, self.n_ei)
                score = _logpdf(cand, *lg, lo, hi) - _logpdf(cand, *lb, lo, hi)
                p[k] = _from_u(sp, cand[int(np.argmax(score))])
        return p

    def optimize(self, objective, n_trials, callback=None):
        for t in range(n_trials):
            p = self._suggest(); v = objective(p); self.trials.append((p, v))
            if callback: callback(t, p, v, self)

    @property
    def best(self):
        i = int(np.argmax([v for _, v in self.trials])); return i, self.trials[i]
