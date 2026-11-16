"""Subject-level statistics: paired tests, effect sizes, bootstrap intervals, multiple-comparison correction."""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _clean(x, y=None):
    x = np.asarray(x, float)
    if y is None:
        return x[np.isfinite(x)], None
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def hodges_lehmann(differences):
    """Pseudo-median of differences: median of all pairwise averages."""
    first, second = np.triu_indices(len(differences))
    return float(np.median((differences[first] + differences[second]) / 2)) if len(differences) else np.nan


def rank_biserial(differences):
    differences = differences[differences != 0]
    if not len(differences):
        return np.nan
    ranks = stats.rankdata(np.abs(differences))
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / ranks.sum())


def bootstrap_ci(values, statistic=np.median, n_resamples=10000, seed=0, alpha=0.05):
    if len(values) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(values), (n_resamples, len(values)))
    estimates = (
        np.array([statistic(values[resample]) for resample in resamples])
        if statistic is not np.median
        else np.median(values[resamples], axis=1)
    )
    return tuple(float(bound) for bound in np.quantile(estimates, [alpha / 2, 1 - alpha / 2]))


def paired_test(x, y=None, alternative="two-sided", shift=0.0, n_boot=10000):
    """Wilcoxon signed-rank test of x - y (or of x against 0), with effect sizes and a bootstrap interval.
    `shift` is added to the differences for the test (non-inferiority margin); median, Hodges-Lehmann estimate and
    interval describe the differences themselves. Too few subjects -> p = NaN, never an error."""
    x, y = _clean(x, y)
    differences = x - y if y is not None else x
    shifted = differences + shift
    out = dict(
        n=int(len(shifted)),
        median=float(np.median(differences)) if len(shifted) else np.nan,
        hodges_lehmann=hodges_lehmann(differences),
        rank_biserial=rank_biserial(shifted),
        p=np.nan,
    )
    out["ci_lo"], out["ci_hi"] = bootstrap_ci(differences, n_resamples=n_boot)
    if len(shifted) >= 6 and np.any(shifted != 0):
        out["p"] = float(stats.wilcoxon(shifted, alternative=alternative).pvalue)
    return out


def correct(p_values, method="holm"):
    """Holm ('holm') or Benjamini-Hochberg ('fdr_bh') adjusted p-values; NaNs stay NaN."""
    p_values = np.asarray(p_values, float)
    adjusted = np.full(len(p_values), np.nan)
    finite = np.isfinite(p_values)
    if finite.any():
        adjusted[finite] = multipletests(p_values[finite], method=method)[1]
    return adjusted


def mixed_model(df, formula="rsi ~ C(method, Treatment('csd')) + C(protocol)", group="sid"):
    """Linear mixed model with a random intercept per subject; method effects are relative to classical CSD.
    Returns the fitted summary table or None."""
    import statsmodels.formula.api as smf

    try:
        return smf.mixedlm(formula, df.dropna(), groups=df.dropna()[group]).fit(reml=True).summary().tables[1]
    except Exception as e:  # too few subjects (DEV_MODE) or a singular design
        print("mixed model not fitted:", type(e).__name__, e)
        return None
