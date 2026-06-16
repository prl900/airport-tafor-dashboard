"""Phase B — benchmark harness: the promotion-significance logic is the key test."""

from wx.ai.evaluate import bootstrap_hss_diff


def test_bootstrap_promotes_clear_winner():
    # champion never detects the event; challenger is perfect.
    champion = ["miss"] * 50 + ["correct_neg"] * 50
    challenger = ["hit"] * 50 + ["correct_neg"] * 50
    res = bootstrap_hss_diff(champion, challenger, n_boot=300)
    assert res["diff"] > 0
    assert res["ci_low"] > 0
    assert res["wins"] is True


def test_bootstrap_rejects_clear_loser():
    champion = ["hit"] * 50 + ["correct_neg"] * 50
    challenger = ["miss"] * 50 + ["correct_neg"] * 50
    res = bootstrap_hss_diff(champion, challenger, n_boot=300)
    assert res["diff"] < 0
    assert res["wins"] is False


def test_bootstrap_ties_do_not_promote():
    # identical skill -> CI straddles zero -> not a win.
    outcomes = (["hit"] * 30 + ["miss"] * 20 + ["false_alarm"] * 10 + ["correct_neg"] * 40)
    res = bootstrap_hss_diff(outcomes, list(outcomes), n_boot=300)
    assert res["wins"] is False


def test_bootstrap_handles_empty():
    res = bootstrap_hss_diff([], [], n_boot=100)
    assert res["wins"] is False and res["diff"] is None
