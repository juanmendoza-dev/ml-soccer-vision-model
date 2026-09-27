"""The stages that run without model weights. YOLO wrappers are checked on the workstation."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vision.stages import KitColorTeams
from vision.types import PLAYER, REFEREE, Detection

GREEN, RED, BLUE = (40, 140, 40), (30, 30, 220), (220, 60, 30)


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
