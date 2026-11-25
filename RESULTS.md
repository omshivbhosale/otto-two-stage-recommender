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
- These scores are **not comparable to the Kaggle leaderboard**, which uses
  OTTO's hidden test set.

## Main result

| System | Clicks | Carts | Orders | **Score** |
|---|---|---|---|---|
| Popular items only | 0.0053 | 0.0067 | 0.0060 | 0.0061 |
| Session history + popular | 0.3259 | 0.3122 | 0.5736 | 0.4704 |
| History + bought-together + popular | 0.4244 | 0.3831 | 0.6170 | 0.5276 |
| **Two-stage: 4 candidate sources + LightGBM ranker** | **0.4727** | **0.4006** | **0.6285** | **0.5446** |

The two-stage system improves the best baseline by **+0.017 (+3.2%)**,
with gains on every event type.

## How much room was there?

If a perfect ranker picked the best 20 items from the combined candidate
pool (history, GRU, bought-together), the score would be **0.5717**.
The ranker captured **about 40%** of the possible gain over the best
baseline (+0.017 of +0.044). Most remaining misses are items that never
reach the candidate pool, a limit of retrieval rather than ranking.

## Ranker ablation (ranker training week, held-out 20% of sessions)

Share of the right answers in the candidate pool that land in the top 20:

| Target | Fixed rule | Ranker, no GRU features | Ranker + GRU features |
|---|---|---|---|
| Clicks | 0.8856 | 0.9293 | **0.9487** |
| Carts | 0.9268 | 0.9373 | **0.9497** |
| Orders | 0.9527 | 0.9641 | **0.9688** |

The GRU features (`gru_rank`, `gru_sim`, `sim_all`) improve every target,
most of all clicks.

## What mattered most (LightGBM gain share)

| Feature | Clicks | Carts | Orders |
|---|---|---|---|
| `secs_since` (time since the shopper last touched the item) | 0.698 | 0.295 | 0.830 |
| `hist_rank` (position in the shopper's recent history) | 0.010 | 0.545 | 0.014 |
| `sim_all` (GRU session–item similarity) | 0.053 | 0.019 | 0.014 |
| `gru_rank` | 0.043 | 0.008 | 0.007 |
| `n_carts` (times the shopper carted the item) | 0.000 | 0.050 | 0.044 |

Recency dominates: shoppers mostly act on items they just viewed.
The GRU features carry less total weight but decide the hard cases,
items the shopper has not seen yet.

## Findings along the way

- **The GRU alone does not beat bought-together.** In place of
  bought-together it scored 0.5222 vs 0.5276: better on clicks
  (0.4507 vs 0.4244), worse on carts and orders. Combining the two
  only helped once a ranker chose between them item by item.
- **A ranker is only useful with diverse candidates.** Without GRU
  candidates (about 40 per session), a fixed order already caught about
  98.5% of the reachable answers, leaving the ranker almost nothing to fix.
- **Shrinking the item catalog for faster search did not work.** Almost
  every item had activity in the previous 2 weeks; halving the catalog
  would have made 7.6% of next clicks impossible to recommend.
- **The search bottleneck was top-k on the Mac GPU.** Moving the top-k
  step to the CPU made the full search 3.6× faster (about 2 hours →
  about 40 minutes for 1.8M sessions) with identical results.

## Limitations

- The three rankers (clicks, carts, orders) are merged into one 20-item
  list with a simple weighted-rank blend; a tuned blend, or separate
  lists per event type, could score higher.
- Rankers were trained on a 300K-session sample of the ranker week.
- The GRU is small (64-dimensional, 1 epoch) and was still improving
  when training stopped.