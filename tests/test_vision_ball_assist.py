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


def test_spot_sample_is_a_tenth_seeded_and_at_least_one():
    assert len(ba.spot_sample(list(range(95)))) == 10
    assert ba.spot_sample([7]) == [7]
    assert ba.spot_sample(list(range(50)), seed=1) == ba.spot_sample(list(range(50)), seed=1)
    assert ba.spot_sample([]) == []


def test_spot_check_turns_auto_off_above_three_percent():
    labels = {i: [1.0, 1.0] for i in range(0, 200, 5)}  # 40 auto frames
    sample = [0, 5, 10, 15]
    entry = {"labels": dict(labels), "auto": list(labels), "spot_checked": [0, 5, 10]}
    assert ba.apply_spot(entry, sample) is False  # 15 not checked yet
    entry["spot_checked"].append(15)
    assert ba.apply_spot(entry, sample) is False  # nothing changed
    entry["auto"].remove(5)  # the person moved frame 5
    assert ba.spot_result(entry) == (4, 1, 0.25)
    assert ba.apply_spot(entry, sample) is True
    assert entry["auto_off"] and entry["auto"] == []
    assert set(entry["labels"]) == {0, 5, 10, 15}  # unchecked auto labels go back to the person


def test_flags_are_resolved_by_a_decision():
    import polars as pl

    truth = pl.DataFrame(
        {
            "src": [5, 10, 15, 20],
            "kind": ["pff", "pff", "estimated", "pff"],
            "u": [100.0] * 4,
            "v": [100.0] * 4,
        }
    )
    labels = {5: [140.0, 100.0], 10: "none", 15: [300.0, 100.0], 20: [105.0, 100.0]}
    cands = {10: [(110.0, 100.0, 0.2)]}
    assert ba.flags(labels, truth, cands) == [5, 10]  # 15 is estimated: never flagged
    entry = {"labels": labels, "flag_checked": [5]}
    assert ba.unresolved_flags(entry, truth, cands) == [10]
    assert ba.flag_reason(labels, truth, cands, 5) == "label 40 px from PFF"
    assert ba.flag_reason(labels, truth, cands, 10) == "none, candidate 10 px from PFF"
