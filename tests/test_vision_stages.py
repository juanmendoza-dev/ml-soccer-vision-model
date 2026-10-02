"""The stages that run without model weights. YOLO wrappers are checked on the workstation."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vision.stages import KitColorTeams
from vision.types import PLAYER, REFEREE, Detection

GREEN, RED, BLUE = (40, 140, 40), (30, 30, 220), (220, 60, 30)
YELLOW = (40, 230, 240)  # the keeper, the ref, a floodlit sleeve: a few crops off to one side


def crop(shirt, grass_share=0.4):
    c = np.full((30, 20, 3), shirt, dtype=np.uint8)
    c[: int(30 * grass_share)] = GREEN  # grass behind the player
    return c


def test_kit_color_teams():
    teams = KitColorTeams()
    teams.add([crop(RED) for _ in range(10)] + [crop(BLUE) for _ in range(10)])
    assert teams.n_crops == 20 and not teams.fitted
    teams.fit()
    red, blue = teams.predict([crop(RED, 0.7), crop(BLUE, 0.1)])
    assert {red, blue} == {0, 1}
    assert teams.predict([np.full((10, 10, 3), GREEN, dtype=np.uint8)]) == [-1]  # all grass
    teams.reset()
    assert not teams.fitted and teams.n_crops == 0


def test_byte_tracker_keeps_ids():
    pytest.importorskip("supervision")
    from vision.stages import ByteTracker

    tracker = ByteTracker(update_rate=10)
    ids = None
    for step in range(5):
        dets = [
            Detection((100 + step, 100, 120 + step, 150), PLAYER, 0.9),
            Detection((400 - step, 200, 420 - step, 250), REFEREE, 0.8),
        ]
        tracks = tracker.update(dets)
        assert {t.cls for t in tracks} == {PLAYER, REFEREE}
        now = {t.cls: t.track_id for t in tracks}
        assert ids is None or now == ids
        ids = now
    assert tracker.update([]) == []


def player_at(step, conf=0.9):
    return Detection((100 + step, 100, 120 + step, 150), PLAYER, conf)


@pytest.mark.parametrize("gap, same_id", [(9, True), (12, False)])
def test_byte_tracker_lost_track_lasts_lost_track_s(gap, same_id):
    pytest.importorskip("supervision")
    from vision.stages import ByteTracker

    tracker = ByteTracker(update_rate=10, lost_track_s=1.0)  # 10 updates
    for step in range(5):
        (first,) = tracker.update([player_at(step)])
    for _ in range(gap):
        tracker.update([])
    tracker.update([player_at(5)])  # a new track only shows from its second update
    (back,) = tracker.update([player_at(6)])
    assert (back.track_id == first.track_id) == same_id


def test_byte_tracker_keeps_a_track_on_low_confidence_detections():
    pytest.importorskip("supervision")
    from vision.stages import ByteTracker

    tracker = ByteTracker(update_rate=10)
    for step in range(5):
        (first,) = tracker.update([player_at(step)])
    for step in range(5, 10):  # partly occluded: second-pass confidence
        (low,) = tracker.update([player_at(step, conf=0.15)])
        assert low.track_id == first.track_id


def test_kit_clusters_keep_their_numbers_after_a_refit():
    teams = KitColorTeams()
    teams.add([crop(RED) for _ in range(10)] + [crop(BLUE) for _ in range(10)])
    teams.fit()
    red, blue = teams.predict([crop(RED), crop(BLUE)])
    for _ in range(5):  # long breaks: refit on differently ordered samples
        teams.reset()
        teams.add([crop(BLUE) for _ in range(7)] + [crop(RED) for _ in range(12)])
        teams.fit()
        assert teams.predict([crop(RED), crop(BLUE)]) == [red, blue]


@pytest.mark.parametrize("seed", range(20))
def test_kit_colors_split_the_kits_not_the_odd_crops(seed):
    """smoke03: a single random init put a handful of bright crops in one cluster and
    both kits in the other, so every player came out the same team."""
    teams = KitColorTeams(seed=seed)
    teams.add(
        [crop(RED) for _ in range(30)]
        + [crop(BLUE) for _ in range(30)]
        + [crop(YELLOW) for _ in range(10)]
    )
    teams.fit()
    red, blue = teams.predict([crop(RED), crop(BLUE)])
    assert {red, blue} == {0, 1}


DARK, LIGHT = (35, 35, 35), (225, 225, 225)  # same chroma, kits apart in lightness only


@pytest.mark.parametrize("seed", range(5))
def test_kit_colors_split_kits_that_differ_in_lightness(seed):
    """vb01: navy France vs white Argentina landed in one cluster on chroma alone."""
    teams = KitColorTeams(seed=seed)
    teams.add([crop(DARK) for _ in range(20)] + [crop(LIGHT) for _ in range(20)])
    teams.fit()
    dark, light = teams.predict([crop(DARK), crop(LIGHT)])
    assert {dark, light} == {0, 1}


def test_kit_clusters_keep_their_numbers_after_a_refit_in_other_light():
    teams = KitColorTeams()
    teams.add([crop(RED) for _ in range(10)] + [crop(BLUE) for _ in range(10)])
    teams.fit()
    red, blue = teams.predict([crop(RED), crop(BLUE)])
    dim_red, dim_blue = tuple(v // 2 for v in RED), tuple(v // 2 for v in BLUE)  # floodlights
    teams.reset()
    teams.add([crop(dim_blue) for _ in range(12)] + [crop(dim_red) for _ in range(8)])
    teams.fit()
    assert teams.predict([crop(dim_red), crop(dim_blue)]) == [red, blue]
