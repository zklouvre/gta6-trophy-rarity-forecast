# GTA VI Trophy Rarity Forecast

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at **2026-11-26T00:00:00Z** — five quantiles (5%, 25%, 50%,
75%, 95%) per trophy.

Built for make.fun's
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty. Referral code: `ddjvdkr0`.

[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zklouvre/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zklouvre/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

## The model, in plain words

Earn rates only go one way in the long run: up, then flat. The program works
on the **logit scale against log-time** — hours since the game's release
midnight, passed through `ln(1 + hours)`. A straight line there is a
saturating logistic curve in rate space, which is how trophy rates actually
grow: fast early, then flattening toward 100%.

Five ideas make it beat a flat "rates stay where they are" forecast:

1. **Collapse frozen reads.** Exophase refreshes on its own schedule, so the
   page sits unchanged for hours. Identical consecutive reads carry no new
   information and would drag any fit flat, so each constant run becomes one
   point at the run's last time.
2. **Recency-weighted fit.** A least-squares line per trophy, with weights
   halving every 36 hours, so the fit follows the current regime rather than
   launch night.
3. **Shrinkage toward the tier.** Each trophy's slope is pulled toward the
   median slope of its tier (platinum / gold / silver / bronze). With
   hundreds of reads the data dominates; with thin data the tier pattern
   keeps wild slopes in check.
4. **Anchor at the last read.** The median starts from the freshest observed
   rate and adds `slope × (log-time to target)`, so a refresh dip is never
   smoothed away.
5. **Honest spread.** The 5%–95% band combines the fit's residual scatter,
   a random-walk term that widens with the square root of the horizon
   (calibrated from the history's own day-to-day moves), and slope
   uncertainty over the extrapolation distance. Floored so a short horizon
   still admits a refresh jump.

A trophy already at 100% (the opening mission) gets a near-degenerate band
just under 100%: it cannot rise, and it essentially cannot fall.

## How to run it

Python 3.13, standard library only — nothing to install, no network.

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `*.in` case in the input directory (or `/input` when
`MAKEFUN_INPUT_DIR` is unset) and writes one `<id>.out` per case to the
output directory (or `/output`). Each output is a single JSON object
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per
trophy, quantiles non-decreasing, in percent, at most 65,536 bytes.
Deterministic: the same input always produces byte-identical output.

## Validation

- Practice cases `p00` / `p01` (flat launch-night history): valid output,
  55 rows, quantiles ordered, under 8 seconds per case. Their histories are
  completely flat, so the forecast sits near the last rate with a modest
  spread — the practice is a format and runtime smoke test, not a model
  test.
- Synthetic rising case: a fake history built on a logistic-in-log-time
  growth path with refresh jumps and a frozen stretch, checkpoint
  2026-11-21T11:30:00Z, target 2026-11-26T00:00:00Z (120 reads). The
  program's medians sit above the last rates for 53 of 55 rising trophies
  and track the synthetic truth with a typical miss of 3.13 percentage
  points and 54/55 inside the 90% band. Mean pinball loss **0.2305** against
  the house baseline's **0.3802** on the same synthetic truth — a ~39%
  improvement, from extrapolating growth instead of assuming rates stay
  flat. Runtime 0.12s per case (limit 8s).

## License

MIT.
