from src.baselines import build_covisit_buys, build_covisit_clicks, popular_items, recommend

EVENT_SCHEMA = "session int, aid int, ts long, type int"
DAY = 24 * 3600


def test_popular_items_uses_recent_carts_and_orders(spark):
    cutoff = 10 * DAY
    train = spark.createDataFrame(
        [(1, 1, 5 * DAY, 1)] * 3      # item 1: carted 3 times, recently
        + [(1, 2, 5 * DAY, 2)]        # item 2: ordered once, recently
        + [(1, 3, 5 * DAY, 0)] * 10   # item 3: only clicked, so ignored
        + [(1, 4, 1 * DAY, 1)] * 5,   # item 4: carted, but too old
        EVENT_SCHEMA,
    )
    assert popular_items(train, cutoff, k=3) == [1, 2]


def test_covisit_counts_bought_together(spark):
    train = spark.createDataFrame(
        [
            (1, 1, 1, 1), (1, 2, 2, 1),   # session 1 buys 1 and 2
            (2, 1, 1, 2), (2, 2, 2, 2),   # session 2 buys 1 and 2
            (3, 1, 1, 1), (3, 3, 2, 1),   # session 3 buys 1 and 3
            (4, 1, 1, 0), (4, 9, 2, 0),   # session 4 only clicks: ignored
        ],
        EVENT_SCHEMA,
    )
    cv = build_covisit_buys(train)
    partners = [
        (r.aid_y, r["count"], r["rank"])
        for r in cv.filter("aid_x = 1").orderBy("rank").collect()
    ]
    assert partners == [(2, 2, 1), (3, 1, 2)]



def test_covisit_clicks_pairs_items_viewed_close_together(spark):
    cutoff = 10 * DAY
    train = spark.createDataFrame(
        [
            (1, 1, 5 * DAY, 0), (1, 1, 5 * DAY + 10, 0), (1, 2, 5 * DAY + 100, 0),  # 1 and 2, 1 seen twice
            (2, 1, 5 * DAY, 0), (2, 2, 5 * DAY + 50, 1),                          # 1 and 2 again (cart counts too)
            (3, 1, 5 * DAY, 0), (3, 3, 7 * DAY, 0),                               # 2 days apart: no pair
            (4, 1, 5 * DAY, 0), (4, 4, 5 * DAY + 10, 0),                          # 1 and 4
            (5, 1, 1 * DAY, 0), (5, 5, 1 * DAY + 1, 0),                           # before the window: ignored
        ],
        EVENT_SCHEMA,
    )
    cv = build_covisit_clicks(train, cutoff, days=7)
    partners = [
        (r.aid_y, r["count"], r["rank"])
        for r in cv.filter("aid_x = 1").orderBy("rank").collect()
    ]
    assert partners == [(2, 2, 1), (4, 1, 2)]  # pair counted once per session
    assert cv.filter("aid_x = 5 OR aid_y = 5").count() == 0


def test_recommend_order(spark):
    covisit = spark.createDataFrame(
        [(1, 2, 2, 1), (1, 3, 1, 2)], "aid_x int, aid_y int, count long, rank int"
    )
    test_input = spark.createDataFrame(
        [(100, 1, 1, 0), (100, 5, 2, 0)],  # saw item 1, then item 5
        EVENT_SCHEMA,
    )
    sessions = spark.createDataFrame([(100,), (200,)], "session int")  # 200 has no history
    popular = [9, 1]

    def items(preds):
        return {r.session: r["items"] for r in preds.collect()}

    full = items(recommend(test_input, sessions, popular, covisit=covisit))
    assert full[100] == [5, 1, 2, 3, 9]  # history, then partners, then popular
    assert full[200] == [9, 1]           # no history: popular only

    hist_only = items(recommend(test_input, sessions, popular))
    assert hist_only[100] == [5, 1, 9]

    pop_only = items(recommend(test_input, sessions, popular, use_history=False))
    assert pop_only[100] == [9, 1]