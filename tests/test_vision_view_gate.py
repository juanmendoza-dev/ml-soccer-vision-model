import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vision.config import VisionConfig
from vision.types import MATCH, OTHER
from vision.view_gate import ViewGate, grass_share

GREEN = (40, 140, 40)  # BGR pitch green
STUDIO = (90, 60, 160)


def image(color, h=360, w=640):
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_grass_share():
    assert grass_share(image(GREEN)) > 0.95
    assert grass_share(image(STUDIO)) < 0.05
    half = image(STUDIO)
    half[:, :320] = GREEN
    assert 0.45 < grass_share(half) < 0.55


def run(gate, seconds, grass, fps=10, start=0.0, pitch_ok=None):
    views = []
    for i in range(round(seconds * fps)):
        views.append(gate.update(start + i / fps, grass, pitch_ok))
    return views


def test_hysteresis():
    gate = ViewGate(VisionConfig(off_after_s=0.5, on_after_s=1.0))
    views = run(gate, 2.0, grass=0.8)
    assert views[0] == OTHER and views[9] == OTHER and views[10] == MATCH  # on after 1 s
    # a 0.3 s cutaway doesn't switch off
    assert set(run(gate, 0.3, grass=0.0, start=2.0)) == {MATCH}
    assert run(gate, 1.0, grass=0.8, start=2.3)[-1] == MATCH
    # an ad does, after 0.5 s
    views = run(gate, 2.0, grass=0.0, start=3.3)
    assert views[4] == MATCH and views[5] == OTHER
    assert gate.last_change_t == pytest.approx(3.8)


def test_keypoints_veto_green_frames():
    gate = ViewGate(VisionConfig())
    run(gate, 2.0, grass=0.8)
    assert gate.view == MATCH
    # green close-up: grass passes but stage 4 finds no pitch
    assert run(gate, 1.0, grass=0.8, start=2.0, pitch_ok=False)[-1] == OTHER
