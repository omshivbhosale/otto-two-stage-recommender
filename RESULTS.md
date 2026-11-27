# Results

All scores are Recall@20 using OTTO's weighting:
score = 0.10 × clicks + 0.30 × carts + 0.60 × orders.
Every number below comes from a file in `reports/`, written by the pipeline scripts.

## Evaluation setup

- **Data:** OTTO e-commerce sessions (CC BY 4.0), 4 weeks, 216.7M events, 12.9M sessions, 1.86M items.
- **Test week:** sessions that started in the final 7 days (1,801,251 sessions).
  Each session is cut at a random point; the model sees the events before
  the cut and is scored on what came after (next click, carted items,
  ordered items).
- **Ranker week:** the same procedure one week earlier (2,237,925 sessions),
  built only from data before the test week, so no test information leaks
  into training. Every candidate source, feature and GRU for a week is built
  from events before that week's cutoff, and the scripts assert it.
- **Model selection:** rankers train on 300,000 ranker-week sessions. Every design
  choice was made on the other 1,937,925 ranker-week sessions (scored on two random
  samples of 200,000). The test week was scored once, after the final system was chosen.
- These scores are **not comparable to the Kaggle leaderboard**, which uses
  OTTO's hidden test set.

## Main result (test week, 1,801,251 sessions)

| System | Clicks | Carts | Orders | **Score** |
|---|---|---|---|---|
| Popular items only | 0.0053 | 0.0067 | 0.0060 | 0.0061 |
| Session history + popular | 0.3259 | 0.3122 | 0.5736 | 0.4704 |
| History + bought-together + popular | 0.4244 | 0.3831 | 0.6170 | 0.5276 |
| v1: two-stage, 4 sources, one blended list | 0.4730 | 0.4004 | 0.6282 | 0.5443 |
| **v2: + viewed-together source, separate list per type** | **0.4968** | **0.4087** | **0.6327** | **0.5519** |

v2 improves the best baseline by **+0.0244 (+4.6%)** and v1 by +0.0076, with gains
on every event type (clicks +0.0725, carts +0.0256, orders +0.0157 over the baseline).
Sources: `reports/baselines_test_week.json`, `reports/evaluation_test_week_v1_v2.json`.

## What changed from v1 to v2, one step at a time

| Test week | Clicks | Carts | Orders | Score |
|---|---|---|---|---|
| v1, one blended list | 0.4730 | 0.4004 | 0.6282 | 0.5443 |
| v1, separate list per type | 0.4833 | 0.4009 | 0.6282 | 0.5455 |
| v2, one blended list | 0.4964 | 0.4080 | 0.6327 | 0.5517 |
| v2, separate list per type | 0.4968 | 0.4087 | 0.6327 | 0.5519 |

1. **Viewed-together candidates** (+0.0064 with separate lists, +0.0073 with one list).
   For each item the shopper touched, add the items other shoppers viewed within 24 hours
   of it (last 14 days before the cutoff). This raises the pool from 74 to 83 candidates per
   session, and `clk_rank` becomes the click ranker's second most important feature
   (10.5% of gain, after recency).
2. **Separate lists per event type** (+0.0012 on v1, +0.0003 on v2). OTTO scores clicks,
   carts and orders separately; merging three rankers into one list leaves little room for
   click-only items, so almost all of the v1 gain is on clicks (+0.0103). Once the
   viewed-together source adds click candidates, the separate lists add almost nothing.

### The same order held on held-out ranker-week sessions

These were scored before the test week, and decided the final system:

| Score | Seed 42 | Seed 7 | Test week |
|---|---|---|---|
| v1, one blended list | 0.6033 | 0.6062 | 0.5443 |
| v1, separate list per type | 0.6040 | 0.6071 | 0.5455 |
| v2, one blended list | 0.6102 | 0.6137 | 0.5517 |
| **v2, separate list per type** | **0.6103** | **0.6142** | **0.5519** |

v2 beat v1 on both samples (+0.0064 and +0.0071 with separate lists). Ranker-week scores
are higher than test-week scores for every system; they are only used to compare systems
within the same week. Sources: `reports/evaluation_ranker_week_v1_v2_seed{42,7}.json`.

## What the data shows

From `notebooks/01_data_exploration.ipynb` (test week):

| Answer type | Answers | Already touched in the session |
|---|---|---|
| Next click | 1,755,489 | 32.4% |
| Carted items | 581,121 | 31.2% |
| Ordered items | 313,045 | 59.5% |

Most orders are for items the shopper already looked at, which is why session history is
the first candidate source and recency the strongest ranking feature. Events are 89.9%
clicks, 7.8% carts and 2.4% orders; the top 1% of items get 38.2% of all events.

## What the v2 rankers rely on (LightGBM gain share)

| Feature | Clicks | Carts | Orders |
|---|---|---|---|
| `secs_since` (time since the shopper last touched the item) | 0.684 | 0.283 | 0.829 |
| `hist_rank` (position in the shopper's recent history) | 0.010 | 0.553 | 0.013 |
| `clk_rank` (viewed-together rank, new in v2) | 0.105 | 0.023 | 0.007 |
| `n_carts` (times the shopper carted the item) | 0.000 | 0.044 | 0.042 |
| `gru_rank` (GRU retrieval rank) | 0.021 | 0.014 | 0.006 |

Full shares for both versions: `reports/rankers_v1.json`, `reports/rankers_v2.json`.

## Serving

The API (`src/serve.py`) recomputes the v2 candidates and features for one session at a time in NumPy/pandas.
`scripts/check_serving.py` sends 200 random test-week sessions through it and compares the results with the
Spark pipeline's saved features, scored by the same rankers:

| Check (200 sessions) | Result |
|---|---|
| Same candidate items | 200 / 200 |
| Feature mismatches | 0 (GRU similarities agree to 6 decimal places) |
| Same top 20 for clicks / carts / orders | 200 / 200 / 200 |
| Latency, median / p95 | 14.8 ms / 15.4 ms |

Source: `reports/serving_check.json`. The lookup tables the API loads are rebuilt from the pipeline's data by
`scripts/export_serving.py`, with the same cutoff as the test week.

## Reproducibility

`run_pipeline.sh` rebuilds every table, model and report from the raw data. A clean rerun
scored 0.5519 on the test week; the original exploratory run of the same system scored 0.5522.
The scripts stop with an error if any session is missing from a saved feature table, so an
incomplete write cannot silently change a score.

## Earlier experiments

These come from the exploratory notebooks during development (kept in the git history)
and are not rerun by the pipeline scripts.

- **Room left in the v1 pool.** A perfect ranker over the v1 candidates would have scored
  0.5717. The v1 ranker captured about 40% of the possible gain over the best baseline, and
  most remaining misses never reached the pool, which motivated the viewed-together source.
- **GRU features help every ranker.** Share of reachable answers ranked in the top 20
  (fixed rule → ranker without GRU features → ranker with them): clicks
  0.8856 → 0.9293 → 0.9487, carts 0.9268 → 0.9373 → 0.9497, orders 0.9527 → 0.9641 → 0.9688.
- **The GRU alone does not beat bought-together.** In place of bought-together it scored
  0.5222 vs 0.5276: better on clicks (0.4507 vs 0.4244), worse on carts and orders.
  Combining the two only helped once a ranker chose between them item by item.
- **A ranker is only useful with diverse candidates.** Without GRU candidates, a fixed order
  already caught about 98.5% of the reachable answers, leaving the ranker almost nothing to fix.
- **Shrinking the item catalog for faster search did not work.** Almost every item had activity
  in the previous 2 weeks; halving the catalog would have made 7.6% of next clicks impossible
  to recommend.
- **The search bottleneck was top-k on the Mac GPU.** Moving the top-k step to the CPU made
  the full search 3.6× faster (about 2 hours → about 40 minutes for 1.8M sessions) with
  identical results.

## Findings

- **More candidates beat a better merge.** Adding one candidate source gained +0.0064;
  separate lists per type gained +0.0012, about a fifth as much.
- **Recency dominates.** `secs_since` carries 68–83% of the click and order rankers' gain.

## Limitations

- The v2 click ranker is the weakest of the three: on both ranker-week samples its own list
  scored slightly lower on clicks than the blended list (0.5427 vs 0.5429, 0.5417 vs 0.5431),
  while on the test week it was slightly ahead (0.4968 vs 0.4964).
- Rankers were trained on a 300K-session sample of the ranker week.
- The GRU is small (64-dimensional, 1 epoch) and was still improving when training stopped.
- Latency was measured on an Apple M5 laptop, one session at a time, outside Docker.
