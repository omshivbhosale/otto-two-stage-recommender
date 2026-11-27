# Results

All scores are Recall@20 on a held-out final week of the OTTO dataset,
using OTTO's weighting: score = 0.10 × clicks + 0.30 × carts + 0.60 × orders.

## Evaluation setup

- **Data:** OTTO e-commerce sessions (CC BY 4.0), 4 weeks, 216.7M events.
- **Test week:** sessions that started in the final 7 days (1.80M sessions).
  Each session is cut at a random point; the model sees the events before
  the cut and is scored on what came after (next click, carted items,
  ordered items).
- **Ranker training week:** the same procedure one week earlier (2.24M sessions),
  built only from data before the test week, so no test information leaks
  into training.
- **Model selection:** every design choice was made on ranker-week sessions the
  rankers never trained on (1.94M held out; scored on two random samples of 200K).
  The test week was scored once, after the final system was chosen.
- These scores are **not comparable to the Kaggle leaderboard**, which uses
  OTTO's hidden test set.

## Main result (test week, 1,801,251 sessions)

| System | Clicks | Carts | Orders | **Score** |
|---|---|---|---|---|
| Popular items only | 0.0053 | 0.0067 | 0.0060 | 0.0061 |
| Session history + popular | 0.3259 | 0.3122 | 0.5736 | 0.4704 |
| History + bought-together + popular | 0.4244 | 0.3831 | 0.6170 | 0.5276 |
| v1: two-stage, 4 sources, one blended list | 0.4727 | 0.4006 | 0.6285 | 0.5446 |
| **v2: + viewed-together source, separate list per type** | **0.4964** | **0.4089** | **0.6332** | **0.5522** |

v2 improves the best baseline by **+0.025 (+4.7%)** and v1 by +0.008, with gains
on every event type. Source: `reports/evaluation_test_week_v1_v2.json`.

## What changed from v1 to v2, one step at a time

| Test week | Clicks | Carts | Orders | Score |
|---|---|---|---|---|
| v1, one blended list | 0.4727 | 0.4006 | 0.6285 | 0.5446 |
| v1, separate list per type | 0.4858 | 0.4013 | 0.6285 | 0.5461 |
| v2, one blended list | 0.4957 | 0.4083 | 0.6330 | 0.5519 |
| v2, separate list per type | 0.4964 | 0.4089 | 0.6332 | 0.5522 |

1. **Separate lists per event type** (+0.0015 on v1). OTTO scores clicks, carts
   and orders separately; merging three rankers into one list left little room for
   click-only items. Almost all of the gain is on clicks.
2. **Viewed-together candidates** (+0.006). For each item the shopper touched, add the
   items other shoppers viewed within 24 hours of it (last 14 days before the cutoff).
   This raises the pool from 74 to 83 candidates per session, and `clk_rank` becomes the
   click ranker's second most important feature (10.5% of gain, after recency).

The same order held on both held-out ranker-week samples before the test week was
scored (score, seed 42 / seed 7): v1 blended 0.6086 / 0.6094, v1 separate 0.6102 / 0.6104,
v2 blended 0.6164 / 0.6173, v2 separate 0.6165 / 0.6172.
Ranker-week scores are higher than test-week scores and are only used to compare systems.

## How much room was there? (v1)

If a perfect ranker picked the best 20 items from the v1 candidate pool
(history, GRU, bought-together), the score would be **0.5717**.
The v1 ranker captured **about 40%** of the possible gain over the best
baseline (+0.017 of +0.044). Most remaining misses were items that never
reached the candidate pool, which is what motivated the v2 source.

## v1 ranker ablation (ranker training week, held-out 20% of sessions)

Share of the right answers in the candidate pool that land in the top 20:

| Target | Fixed rule | Ranker, no GRU features | Ranker + GRU features |
|---|---|---|---|
| Clicks | 0.8856 | 0.9293 | **0.9487** |
| Carts | 0.9268 | 0.9373 | **0.9497** |
| Orders | 0.9527 | 0.9641 | **0.9688** |

The GRU features (`gru_rank`, `gru_sim`, `sim_all`) improve every target,
most of all clicks.

## What mattered most (v1, LightGBM gain share)

| Feature | Clicks | Carts | Orders |
|---|---|---|---|
| `secs_since` (time since the shopper last touched the item) | 0.698 | 0.295 | 0.830 |
| `hist_rank` (position in the shopper's recent history) | 0.010 | 0.545 | 0.014 |
| `sim_all` (GRU session–item similarity) | 0.053 | 0.019 | 0.014 |
| `gru_rank` | 0.043 | 0.008 | 0.007 |
| `n_carts` (times the shopper carted the item) | 0.000 | 0.050 | 0.044 |

Recency dominates: shoppers mostly act on items they just viewed.
v2 gain shares are in `reports/rankers_v2.json`.

## Findings along the way

- **The GRU alone does not beat bought-together.** In place of
  bought-together it scored 0.5222 vs 0.5276: better on clicks
  (0.4507 vs 0.4244), worse on carts and orders. Combining the two
  only helped once a ranker chose between them item by item.
- **A ranker is only useful with diverse candidates.** Without GRU
  candidates (about 40 per session), a fixed order already caught about
  98.5% of the reachable answers, leaving the ranker almost nothing to fix.
- **More candidates beat a better merge.** Separate lists per type gained +0.0015;
  adding one candidate source gained about four times as much.
- **Shrinking the item catalog for faster search did not work.** Almost
  every item had activity in the previous 2 weeks; halving the catalog
  would have made 7.6% of next clicks impossible to recommend.
- **The search bottleneck was top-k on the Mac GPU.** Moving the top-k
  step to the CPU made the full search 3.6× faster (about 2 hours →
  about 40 minutes for 1.8M sessions) with identical results.

## Limitations

- The v2 click ranker stopped early (84 trees) and is the weakest of the three: on the
  ranker week its own list scored lower on clicks than the blended list (0.5413 vs 0.5436,
  seed 42), although on the test week it was slightly ahead (0.4964 vs 0.4957).
- Rankers were trained on a 300K-session sample of the ranker week.
- The GRU is small (64-dimensional, 1 epoch) and was still improving
  when training stopped.
