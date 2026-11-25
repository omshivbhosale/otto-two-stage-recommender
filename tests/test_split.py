from src.split import make_test_labels, time_split

EVENT_SCHEMA = "session int, aid int, ts long, type int"
DAY = 24 * 3600


def test_time_split(spark):
    events = spark.createDataFrame(
        [
            (1, 10, 1 * DAY, 0),   # session 1: early, fully in train
            (1, 11, 2 * DAY, 0),
            (2, 20, 12 * DAY, 0),  # session 2: crosses the cutoff
            (2, 21, 14 * DAY, 0),
            (3, 30, 15 * DAY, 0),  # session 3: starts after the cutoff
            (3, 31, 16 * DAY, 1),
            (3, 32, 20 * DAY, 2),
        ],
        EVENT_SCHEMA,
    )
    train, test, cutoff = time_split(events)

    assert cutoff == 13 * DAY + 1
    # no future events leak into train
    assert train.agg({"ts": "max"}).first()[0] < cutoff
    # session 2 keeps only its early event in train
    assert sorted(r.aid for r in train.collect()) == [10, 11, 20]
    # only session 3 is a test session, with all its events
    assert {r.session for r in test.collect()} == {3}
    assert test.count() == 3


def test_two_event_session_cuts_after_first(spark):
    test = spark.createDataFrame(
        [(10, 100, 1, 0), (10, 200, 2, 1)],  # click 100, then cart 200
        EVENT_SCHEMA,
    )
    test_input, labels = make_test_labels(test)

    assert [r.aid for r in test_input.collect()] == [100]
    row = labels.collect()[0]
    assert row.click is None
    assert row.carts == [200]
    assert row.orders is None


def test_single_event_sessions_are_dropped(spark):
    test = spark.createDataFrame(
        [(20, 1, 1, 0), (30, 2, 1, 0), (30, 3, 2, 0)],
        EVENT_SCHEMA,
    )
    _, labels = make_test_labels(test)
    assert {r.session for r in labels.collect()} == {30}


def test_cut_leaves_input_and_answers(spark):
    test = spark.createDataFrame(
        [(40, aid, ts, 0) for ts, aid in enumerate(range(400, 405))],  # 5 clicks
        EVENT_SCHEMA,
    )
    test_input, labels = make_test_labels(test)
    n_input = test_input.count()

    assert 1 <= n_input <= 4  # at least one seen event, at least one answer
    assert labels.first().click is not None