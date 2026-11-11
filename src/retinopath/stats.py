"""Subject-level statistics: paired tests, effect sizes, bootstrap intervals, multiple-comparison correction."""
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _clean(x, y=None):
    x = np.asarray(x, float)
    if y is None:
        return x[np.isfinite(x)], None
    y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def hodges_lehmann(d):
    """Pseudo-median of differences: median of all pairwise averages."""
    i, j = np.triu_indices(len(d)); return float(np.median((d[i] + d[j]) / 2)) if len(d) else np.nan


def rank_biserial(d):
    d = d[d != 0]
    if not len(d):
        return np.nan
    r = stats.rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bootstrap_ci(d, fn=np.median, n=10000, seed=0, alpha=0.05):
    if len(d) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), (n, len(d)))
    b = np.array([fn(d[i]) for i in idx]) if fn is not np.median else np.median(d[idx], axis=1)
    return tuple(float(v) for v in np.quantile(b, [alpha / 2, 1 - alpha / 2]))


def paired_test(x, y=None, alternative="two-sided", shift=0.0, n_boot=10000):
    """Wilcoxon signed-rank test of x - y (or of x against 0), with effect sizes and a bootstrap interval.
    `shift` is added to the differences (non-inferiority margin). Too few subjects -> p = NaN, never an error."""
    x, y = _clean(x, y); d = (x - y if y is not None else x) + shift
    out = dict(n=int(len(d)), median=float(np.median(d)) if len(d) else np.nan, hodges_lehmann=hodges_lehmann(d),
               rank_biserial=rank_biserial(d), p=np.nan)
    out["ci_lo"], out["ci_hi"] = bootstrap_ci(d, n=n_boot)
    if len(d) >= 6 and np.any(d != 0):
        out["p"] = float(stats.wilcoxon(d, alternative=alternative).pvalue)
    return out


def correct(p, method="holm"):
    """Holm ('holm') or Benjamini-Hochberg ('fdr_bh') adjusted p-values; NaNs stay NaN."""
    p = np.asarray(p, float); out = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method=method)[1]
    return out


def mixed_model(df, formula="rsi ~ C(method, Treatment('csd')) + C(protocol)", group="sid"):
    """Linear mixed model with a random intercept per subject; method effects are relative to classical CSD.
    Returns the fitted summary table or None."""
    import statsmodels.formula.api as smf
    try:
        return smf.mixedlm(formula, df.dropna(), groups=df.dropna()[group]).fit(reml=True).summary().tables[1]
    except Exception as e:                       # too few subjects (DEV_MODE) or a singular design
        print("mixed model not fitted:", type(e).__name__, e); return None
