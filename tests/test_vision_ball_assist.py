"""vision.ball_assist: the label tool's suggestions, auto-accept, spot check, flags (10-ball 1c)."""

from vision import ball_assist as ba


def test_suggestion_is_the_nearest_candidate_inside_the_kinds_radius():
    cands = [(100.0, 100.0, 0.1), (130.0, 100.0, 0.9)]
    assert ba.suggest(cands, (105.0, 100.0), "pff") == (100.0, 100.0, 0.1)  # any confidence
    assert ba.suggest(cands, (190.0, 100.0), "pff") is None  # 60 px: too far
    assert ba.suggest(cands, (185.0, 100.0), "estimated") == (130.0, 100.0, 0.9)
    assert ba.suggest(cands, (105.0, 100.0), "off_image") is None


def test_auto_accept_needs_confidence_closeness_and_no_rival():
    proj = (100.0, 100.0)
    assert ba.auto_accept([(110.0, 100.0, 0.8)], proj, "pff") == (110.0, 100.0)
    assert ba.auto_accept([(110.0, 100.0, 0.4)], proj, "pff") is None  # weak
    assert ba.auto_accept([(116.0, 100.0, 0.8)], proj, "pff") is None  # > 15 px
    assert ba.auto_accept([(110.0, 100.0, 0.8), (70.0, 100.0, 0.1)], proj, "pff") is None
    assert ba.auto_accept([(110.0, 100.0, 0.8)], proj, "estimated") is None  # pff frames only
