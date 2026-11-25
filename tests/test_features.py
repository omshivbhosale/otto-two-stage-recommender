from src.features import add_features, add_labels, build_candidates

EVENT_SCHEMA = "session int, aid int, ts long, type int"
DAY = 24 * 3600
CUT = 10 * DAY


def make_data(spark):
    train = spark.createDataFrame(
        [
            (1, 1, 5 * DAY, 1), (1, 2, 5 * DAY, 1),   # session 1 carts items 1 and 2
            (2, 1, 5 * DAY, 1), (2, 2, 5 * DAY, 1),   # session 2 carts items 1 and 2
            (3, 3, 5 * DAY, 2),                       # session 3 orders item 3
        ],
        EVENT_SCHEMA,
    )
    inp = spark.createDataFrame(
        [(100, 1, 100, 0), (100, 1, 200, 0), (100, 4, 300, 1)],
        EVENT_SCHEMA,
    )
    gru = spark.createDataFrame(
        [(100, 5, 1, 0.9), (100, 1, 2, 0.8)],
        "session int, aid int, gru_rank int, gru_sim float",
    )
    labels = spark.createDataFrame(
        [(100, 2, [5], None)],
        "session int, click int, carts array<int>, orders array<int>",
    )
    return train, inp, gru, labels


def test_candidates_merge_all_sources(spark):
    train, inp, gru, _ = make_data(spark)
    cands = build_candidates(inp, train, CUT, gru=gru)
    rows = {r.aid: r for r in cands.collect()}

    assert sorted(rows) == [1, 2, 3, 4, 5]  # every item exactly once
    assert rows[4].hist_rank == 1            # most recent history item
    assert rows[1].hist_rank == 2
    assert rows[2].cov_rank == 1             # bought together with item 1
    assert rows[1].pop_rank == 1             # most carted/ordered recently
    assert rows[5].gru_rank == 1
    assert abs(rows[1].gru_sim - 0.8) < 1e-6  # item 1 found by history, popular AND GRU


def test_candidates_without_gru(spark):
    train, inp, _, _ = make_data(spark)
    cands = build_candidates(inp, train, CUT)
    assert "gru_rank" not in cands.columns
    assert sorted(r.aid for r in cands.collect()) == [1, 2, 3, 4]


def test_features_and_labels(spark):
    train, inp, gru, labels = make_data(spark)
    df = add_labels(add_features(build_candidates(inp, train, CUT, gru=gru), inp, train, CUT), labels)
    rows = {r.aid: r for r in df.collect()}

    assert df.count() == 5                    # joins didn't duplicate rows
    assert rows[1].n_clicks == 2 and rows[1].n_carts == 0
    assert rows[4].n_carts == 1
    assert rows[1].secs_since == 100          # last touched at 200, session ends at 300
    assert rows[4].secs_since == 0
    assert rows[1].sess_len == 3 and rows[1].sess_items == 2
    assert rows[1].item_carts == 2            # carted twice in the week before the cutoff
    assert rows[5].n_clicks == 0              # never seen by this shopper -> 0, not null

    assert rows[2].y_click == 1               # next click was item 2
    assert rows[5].y_cart == 1                # item 5 was carted
    assert sum(r.y_order for r in rows.values()) == 0