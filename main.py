"""gta6-trophy-rarity-forecast-01: forecast Exophase earn rates for GTA VI (PS5).

Model (written for this entry, standard library only):
  For each trophy, work on the logit scale against log-time x = ln(1 + hours
  since the game's release midnight). A straight line in logit-vs-log-time IS a
  saturating logistic curve in rate space (it levels off toward 100%), which is
  how trophy earn rates actually grow: fast early, then flattening.

  1. Collapse frozen reads. Exophase refreshes discretely, so the page sits
     unchanged for hours at a time. Consecutive reads with an identical rate
     carry no new information and would drag any fit flat, so each constant
     run collapses to a single point at the run's last time.
  2. Recency-weighted least squares. Fit logit(rate) = a + b*x per trophy,
     weights halving every 36 hours, so the fit follows the current regime
     rather than launch-night behaviour.
  3. Shrinkage. Each trophy's slope is shrunk toward the median slope of its
     tier (platinum/gold/silver/bronze), weight 4 distinct points. With
     hundreds of reads the data dominates; with thin data the tier pattern
     keeps wild slopes in check. Slopes are clipped to [-0.5, 2.5].
  4. Anchor at the last read. The forecast median starts from the freshest
     observed rate and adds slope * (x_target - x_last), instead of trusting
     the fitted intercept, so a refresh dip is never smoothed away.
  5. Honest spread. sigma combines (a) the fit's residual scatter, shrunk
     toward the cross-trophy median, (b) a random-walk term sigma_rw *
     sqrt(days) with sigma_rw calibrated from the history's own ~24h logit
     moves (robust MAD scale, floored at 0.35), and (c) slope uncertainty
     times the extrapolation distance. Floored at 0.45 logit units so a
     short-horizon forecast still admits a refresh jump; capped at 2.5.

  Quantiles come from the normal quantiles of the case's levels (Acklam's
  rational approximation), mapped back through the logistic. Outputs are
  clamped to [0.01, 99.99] and rounded to 4 decimals, half up.

  A trophy already at 100% (the opening mission) gets a near-degenerate
  band just under 100: it cannot rise, and it essentially cannot fall.

Deterministic: pure functions of the case file only. No randomness, no
wall-clock reads, no network. Reads every *.in in the input directory
($MAKEFUN_INPUT_DIR, else /input), in name order, and writes one
/output/<id>.out per case ($MAKEFUN_OUTPUT_DIR, else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Recency half-life for the weighted fit, in hours. 36h keeps the fit on the
# current regime (early history is dominated by launch-night dilution, while
# the target regime is steady growth) without letting short-term noise or a
# single refresh dominate: on the synthetic rising case 36h scored slightly
# better than 24h (pinball 0.230 vs 0.234) with more medians above the last
# rate (53/55 vs 51/55).
HALF_LIFE_HOURS = 36.0
# Shrinkage strength, in units of distinct (collapsed) points.
SHRINK_WEIGHT = 4.0
# Slope bounds (logit units per unit of log-time).
SLOPE_MIN, SLOPE_MAX = -0.5, 2.5
# Prior slope bounds (cross-trophy median slopes are clipped to this).
PRIOR_MIN, PRIOR_MAX = 0.0, 1.2
# Spread bounds (logit units).
SIGMA_FLOOR, SIGMA_CAP = 0.45, 2.5
# Random-walk sigma calibration bounds (logit units per sqrt-day).
RW_MIN, RW_MAX, RW_DEFAULT = 0.35, 1.0, 0.5
# Residual scatter floor / cap.
RESID_FLOOR, RESID_CAP = 0.12, 1.5


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up. Whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def collapse_series(points):
    """Collapse consecutive points with an identical rate to one point at the
    run's last time. `points` is a list of (x, y, hours) in time order."""
    out = []
    for x, y, h in points:
        if out and y == out[-1][1]:
            out[-1] = (x, y, h)
        else:
            out.append((x, y, h))
    return out


def weighted_line_fit(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (slope, resid_sd, se_slope)
    or None when the fit is degenerate (fewer than 2 points / no x spread)."""
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    if n > 2:
        var = sum(w * r * r for w, r in zip(ws, resid)) / sw * n / (n - 2)
        resid_sd = math.sqrt(max(0.0, var))
    else:
        resid_sd = 0.0
    se_slope = resid_sd / math.sqrt(sxx) if resid_sd > 0 else 0.0
    return slope, resid_sd, se_slope


def calibrate_random_walk(all_series):
    """Robust scale of ~24h logit moves across all trophies. Each trophy's
    collapsed series contributes the logit change between points about a day
    apart; the MAD-based SD of those changes is the per-day move scale."""
    deltas = []
    for pts in all_series:
        for i in range(len(pts)):
            x_i, y_i, h_i = pts[i]
            target_h = h_i - 24.0
            best = None
            for x_j, y_j, h_j in pts:
                if h_j >= h_i:
                    break
                if abs(h_j - target_h) <= 6.0:
                    if best is None or abs(h_j - target_h) < abs(best[2] - target_h):
                        best = (x_j, y_j, h_j)
            if best is not None:
                deltas.append(y_i - best[1])
    if len(deltas) < 10:
        return RW_DEFAULT
    med = median(deltas)
    mad = median([abs(d - med) for d in deltas])
    sigma = 1.4826 * mad
    if sigma <= 0:
        return RW_DEFAULT
    return min(RW_MAX, max(RW_MIN, sigma))


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    history = case["history"]
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [norm_ppf(min(0.999999, max(0.000001, lv))) for lv in levels]

    release = None
    game = case.get("game") or {}
    rd = game.get("release_date")
    if rd:
        try:
            release = datetime.fromisoformat(rd).replace(tzinfo=timezone.utc)
        except ValueError:
            release = None
    if release is None:
        release = parse_time(history[0]["at"]) if history else parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    def x_of(t: datetime) -> float:
        hours = (t - release).total_seconds() / 3600.0
        return math.log(1.0 + max(hours, -0.99))

    x_target = x_of(target)

    # Per-trophy collapsed series: (x, logit_rate, hours_since_release).
    all_series = []
    last_rates = []
    for i in range(len(trophies)):
        pts = []
        last_rate = None
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            rate = float(rates[i])
            last_rate = rate
            t = parse_time(read["at"])
            hours = (t - release).total_seconds() / 3600.0
            pts.append((x_of(t), logit(rate), hours))
        all_series.append(collapse_series(pts))
        last_rates.append(last_rate)

    # First pass: raw fits, for tier priors and the residual scale.
    raw = []
    for pts in all_series:
        if len(pts) < 2:
            raw.append(None)
            continue
        h_last = pts[-1][2]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ws = [0.5 ** ((h_last - p[2]) / HALF_LIFE_HOURS) for p in pts]
        raw.append(weighted_line_fit(xs, ys, ws))

    tier_slopes = {}
    for tr, fit in zip(trophies, raw):
        if fit is None:
            continue
        tier_slopes.setdefault(tr.get("tier", "bronze"), []).append(fit[0])
    all_slopes = [f[0] for f in raw if f is not None]
    overall_prior = median(all_slopes) if all_slopes else 0.0
    overall_prior = min(PRIOR_MAX, max(PRIOR_MIN, overall_prior))

    resids = [f[1] for f, pts in zip(raw, all_series)
              if f is not None and len(pts) >= 3]
    med_resid = median(resids) if resids else 0.25
    if med_resid <= 0:
        med_resid = 0.25

    sigma_rw = calibrate_random_walk(all_series)

    rows = []
    for i, tr in enumerate(trophies):
        pts = all_series[i]
        last_rate = last_rates[i]
        tid = tr["id"]

        # Trophy already at 100%: a near-degenerate band just under 100.
        if last_rate is not None and last_rate >= 99.99:
            rows.append({"id": tid, "q": [99.9, 99.97, 99.99, 99.99, 99.99]})
            continue

        if last_rate is None:
            # No observation at all: wide band around 1%.
            med_lg = logit(1.0)
            sigma = 1.5
            rows.append({"id": tid, "q": [
                round4(clamp_rate(sigmoid(med_lg + z * sigma))) for z in zs]})
            continue

        n = len(pts)
        fit = raw[i]
        if fit is None:
            slope_raw, resid, se_slope = 0.0, 0.0, 0.0
        else:
            slope_raw, resid, se_slope = fit

        tier = tr.get("tier", "bronze")
        tier_vals = tier_slopes.get(tier, [])
        prior = median(tier_vals) if len(tier_vals) >= 2 else overall_prior
        prior = min(PRIOR_MAX, max(PRIOR_MIN, prior))

        slope = (n * slope_raw + SHRINK_WEIGHT * prior) / (n + SHRINK_WEIGHT)
        slope = min(SLOPE_MAX, max(SLOPE_MIN, slope))

        x_last = pts[-1][0] if pts else x_target
        dx = x_target - x_last
        med_lg = logit(last_rate) + slope * dx

        resid_s = (n * resid + SHRINK_WEIGHT * med_resid) / (n + SHRINK_WEIGHT)
        resid_s = min(RESID_CAP, max(RESID_FLOOR, resid_s))
        se_s = se_slope * n / (n + SHRINK_WEIGHT)
        sigma = math.sqrt(resid_s ** 2
                          + (sigma_rw * math.sqrt(horizon_days)) ** 2
                          + (se_s * dx) ** 2)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

        rows.append({"id": tid, "q": [
            round4(clamp_rate(sigmoid(med_lg + z * sigma))) for z in zs]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast exceeds 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
