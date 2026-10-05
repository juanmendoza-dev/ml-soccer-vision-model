"""08 danger meter: the causal lookup, the log scale and the drawing."""

import numpy as np
import polars as pl
import pytest

cv2 = pytest.importorskip("cv2")

from demo import meter
from demo import overlay as ov


def preds(ts, ps, period=1):
    return pl.DataFrame({"period": [period] * len(ts), "t_s": ts, "p": ps})


def test_a_frame_between_rows_shows_the_earlier_one():
    m = meter.Meter(preds([0.0, 0.1, 0.2], [0.01, 0.2, 0.3]), "p")
    assert m.value(1, 0.05) == 0.01
    assert m.value(1, 0.0999) == 0.01  # just before a row: still the older one
    assert m.value(1, 0.1) == 0.2  # exactly on a row: that row (its frame is <= t)


def test_before_the_first_row_or_another_period_shows_nothing():
    m = meter.Meter(preds([1.0, 1.1], [0.1, 0.1]), "p")
    assert m.value(1, 0.95) is None
    assert m.value(2, 1.05) is None


def test_a_grid_gap_shows_nothing():
    m = meter.Meter(preds([0.0, 0.1, 0.4], [0.1, 0.2, 0.3]), "p")
    assert m.value(1, 0.2) == 0.2  # 0.1 s after the last row: still fine
    assert m.value(1, 0.25) is None  # rows 0.2 and 0.3 were skipped
    assert m.value(1, 0.4) == 0.3


def test_a_null_p_shows_nothing():
    m = meter.Meter(preds([0.0, 0.1], [0.1, None]), "p")
    assert m.value(1, 0.15) is None


def test_level_is_a_clamped_log_scale():
    assert meter.level(meter.P_MIN) == 0.0
    assert meter.level(meter.P_MAX) == 1.0
    assert meter.level(1e-5) == 0.0 and meter.level(0.9) == 1.0
    assert meter.level(None) is None and meter.level(float("nan")) is None
    lv = [meter.level(p) for p in (0.002, 0.01, 0.05, 0.2)]
    assert lv == sorted(lv)


def test_label():
    assert meter.label(None) == "--"
    assert meter.label(0.034) == "3.4%"
    assert meter.label(0.27) == "27%"


def test_the_bar_fills_to_its_level():
    box = (0, 0, meter.METER_W, 300)
    hi = np.zeros((300, meter.METER_W, 3), np.uint8)
    lo = hi.copy()
    meter.draw(hi, box, 0.3)
    meter.draw(lo, box, 0.003)
    x = 14 + 13  # middle of the bar
    y = 300 - 34 - 5  # just above the bar's bottom
    assert tuple(int(v) for v in hi[y, x]) == ov.meter_color(meter.level(0.3))
    y_mid = (40 + 300 - 34) // 2
    assert tuple(int(v) for v in hi[y_mid, x]) != (70, 70, 70)  # 0.3 fills past halfway
    assert tuple(int(v) for v in lo[y_mid, x]) == (70, 70, 70)


def test_no_value_draws_a_grey_bar():
    img = np.zeros((300, meter.METER_W, 3), np.uint8)
    meter.draw(img, (0, 0, meter.METER_W, 300), None)
    assert tuple(int(v) for v in img[300 - 34 - 5, 27]) == (70, 70, 70)


def test_meter_color_runs_green_amber_red():
    assert ov.meter_color(0.0) == ov.METER_COLORS[0]
    assert ov.meter_color(0.5) == ov.METER_COLORS[1]
    assert ov.meter_color(1.0) == ov.METER_COLORS[2]
