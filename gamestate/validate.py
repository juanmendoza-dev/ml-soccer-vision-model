"""Check a game state directory against the schema in 02.

    python -m gamestate.validate data/gamestate/<match_id> [...]

Returns a list of error strings; empty means valid.
"""

import sys
from pathlib import Path

import polars as pl

from gamestate.schema import (
    MAY_BE_EMPTY,
    PITCH_LENGTH,
    PITCH_WIDTH,
    SHOOTOUT_PERIOD,
    TABLES,
    Column,
)

# Positions this far outside the pitch are almost certainly pixels or a bad
# homography, not a player standing by the ad boards.
POSITION_MARGIN_M = 15.0


def _kind_ok(dtype: pl.DataType, kind: str) -> bool:
    if kind == "str":
        return dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum)
    if kind == "int":
        return dtype.is_integer()
    if kind == "float":
        return dtype.is_float()
    if kind == "bool":
        return dtype == pl.Boolean
    if kind == "date":
        return dtype == pl.Date
    if kind == "list_float":
        return isinstance(dtype, pl.List) and dtype.inner.is_float()
    raise ValueError(f"unknown column kind {kind}")


def _check_columns(name: str, df: pl.DataFrame, columns: list[Column]) -> list[str]:
    errors = []
    for col in columns:
        where = f"{name}.{col.name}"
        if col.name not in df.columns:
            errors.append(f"{where}: missing column")
            continue
        s = df[col.name]
        if not _kind_ok(s.dtype, col.kind):
            errors.append(f"{where}: expected {col.kind}, got {s.dtype}")
            continue
        if col.kind == "float" and (nans := s.is_nan().sum()):
            errors.append(f"{where}: {nans} NaN values; write missing values as null")
        if (nulls := s.null_count()) and not col.nullable:
            errors.append(f"{where}: {nulls} null values in a non-null column")
        if col.values is not None:
            bad = s.drop_nulls().cast(pl.String if col.kind == "str" else s.dtype).unique()
            bad = [v for v in bad.to_list() if v not in col.values]
            if bad:
                errors.append(f"{where}: values not allowed: {sorted(map(str, bad))[:5]}")
    return errors


def _check_match(t: dict[str, pl.DataFrame], dir_name: str) -> list[str]:
    errors = []
    m = t["match"]
    if m.height != 1:
        return [f"match: expected 1 row, got {m.height}"]
    match_id = m["match_id"][0]
    if match_id != dir_name:
        errors.append(f"match.match_id {match_id!r} doesn't match directory {dir_name!r}")
    for name, df in t.items():
        ids = df["match_id"].unique().to_list()
        if ids and ids != [match_id]:
            errors.append(f"{name}.match_id: expected only {match_id!r}, got {ids[:5]}")
    fps = m["native_fps"][0]
    if fps is not None and not fps > 0:
        errors.append(f"match.native_fps: must be > 0, got {fps}")
    return errors


def _check_frames(f: pl.DataFrame) -> list[str]:
    errors = []
    if f["frame_id"].n_unique() != f.height:
        errors.append("frames.frame_id: duplicate frame_ids")
    f = f.sort("frame_id")
    if not f["period"].is_sorted():
        errors.append("frames.period: goes backwards as frame_id increases")
    backwards = (
        f.with_columns(dt=pl.col("timestamp_s").diff().over("period"))
        .filter(pl.col("dt") < 0)
        .height
    )
    if backwards:
        errors.append(f"frames.timestamp_s: decreases within a period on {backwards} frames")
    bad_poly = f.filter(
        pl.col("view_polygon").is_not_null() & (pl.col("view_polygon").list.len() != 8)
    ).height
    if bad_poly:
        errors.append(f"frames.view_polygon: {bad_poly} rows without exactly 8 values")
    live_shootout = f.filter(
        (pl.col("period") == SHOOTOUT_PERIOD) & (pl.col("ball_state").fill_null("") != "dead")
    ).height
    if live_shootout:
        errors.append(f"frames: {live_shootout} shootout (period 5) frames not marked dead")
    carrier_no_team = f.filter(
        pl.col("ball_carrier_id").is_not_null() & pl.col("possession_team").is_null()
    ).height
    if carrier_no_team:
        errors.append(
            f"frames: {carrier_no_team} frames with a ball carrier but no possession_team"
        )
    return errors


def _check_positions(name: str, df: pl.DataFrame) -> list[str]:
    max_x = PITCH_LENGTH / 2 + POSITION_MARGIN_M
    max_y = PITCH_WIDTH / 2 + POSITION_MARGIN_M
    off = df.filter((pl.col("x").abs() > max_x) | (pl.col("y").abs() > max_y)).height
    if off:
        return [f"{name}: {off} rows more than {POSITION_MARGIN_M:g} m off the pitch (pixels?)"]
    return []


def _check_objects(t: dict[str, pl.DataFrame]) -> list[str]:
    o, f, p = t["objects"], t["frames"], t["players"]
    errors = _check_positions("objects", o)
    if o.select(pl.struct("frame_id", "object_id").is_duplicated().any()).item():
        errors.append("objects: duplicate (frame_id, object_id) rows")
    unknown = o.join(f, on="frame_id", how="anti").height
    if unknown:
        errors.append(f"objects: {unknown} rows with a frame_id not in frames")
    multi_ball = (
        o.filter(pl.col("object_type") == "ball")
        .group_by("frame_id")
        .len()
        .filter(pl.col("len") > 1)
    ).height
    if multi_ball:
        errors.append(f"objects: {multi_ball} frames with more than one ball")
    teamed = o.filter(
        pl.col("object_type").is_in(["ball", "referee"]) & pl.col("team").is_not_null()
    ).height
    if teamed:
        errors.append(f"objects.team: {teamed} ball/referee rows with a team")
    z_non_ball = o.filter((pl.col("object_type") != "ball") & pl.col("z").is_not_null()).height
    if z_non_ball:
        errors.append(f"objects.z: {z_non_ball} non-ball rows with z set")
    bad_conf = o.filter((pl.col("confidence") < 0) | (pl.col("confidence") > 1)).height
    if bad_conf:
        errors.append(f"objects.confidence: {bad_conf} rows outside [0, 1]")
    guessed = o.filter(~pl.col("visible") & ~pl.col("interpolated")).height
    if guessed:
        errors.append(f"objects: {guessed} rows with visible=False but interpolated=False")
    unlinked = (
        o.filter(pl.col("player_id").is_not_null())
        .select("player_id")
        .unique()
        .join(p, on="player_id", how="anti")
        .height
    )
    if unlinked:
        errors.append(f"objects.player_id: {unlinked} ids not in players")
    carriers = f.filter(pl.col("ball_carrier_id").is_not_null()).select(
        "frame_id", object_id="ball_carrier_id"
    )
    on_pitch = o.filter(pl.col("object_type").is_in(["player", "goalkeeper"])).select(
        "frame_id", "object_id"
    )
    missing = carriers.join(on_pitch, on=["frame_id", "object_id"], how="anti").height
    if missing:
        errors.append(
            f"frames.ball_carrier_id: {missing} frames where the carrier isn't a player in objects"
        )
    return errors


def _check_events(t: dict[str, pl.DataFrame]) -> list[str]:
    e, f = t["events"], t["frames"]
    if e.height == 0:
        return []
    errors = _check_positions("events", e)
    unknown = e.join(f, on="frame_id", how="anti").height
    if unknown:
        errors.append(f"events: {unknown} rows with a frame_id not in frames")
    ruled_out = e.filter(
        (pl.col("event_type") == "goal") & (pl.col("outcome").fill_null("") == "disallowed")
    ).height
    if ruled_out:
        errors.append(f"events: {ruled_out} goal events marked disallowed")
    shootout_shots = (
        e.filter(pl.col("event_type") == "shot")
        .join(f.select("frame_id", "period"), on="frame_id")
        .filter(pl.col("period") == SHOOTOUT_PERIOD)
        .height
    )
    if shootout_shots:
        errors.append(f"events: {shootout_shots} shot events in the shootout (period 5)")
    return errors


def _check_players(p: pl.DataFrame) -> list[str]:
    if p["player_id"].n_unique() != p.height:
        return ["players.player_id: duplicate player_ids"]
    return []


def validate_match(path: str | Path) -> list[str]:
    path = Path(path)
    tables, errors = {}, []
    for name in TABLES:
        file = path / f"{name}.parquet"
        if not file.exists():
            errors.append(f"{name}.parquet: missing")
            continue
        tables[name] = pl.read_parquet(file)
    if errors:
        return errors

    for name, df in tables.items():
        errors += _check_columns(name, df, TABLES[name])
        if df.height == 0 and name not in MAY_BE_EMPTY:
            errors.append(f"{name}: no rows")
    # Cross-table checks assume the columns exist with the right types.
    if errors:
        return errors

    errors += _check_match(tables, path.name)
    errors += _check_frames(tables["frames"])
    errors += _check_objects(tables)
    errors += _check_events(tables)
    errors += _check_players(tables["players"])
    return errors


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip())
        return 2
    failed = 0
    for arg in argv:
        errors = validate_match(arg)
        if errors:
            failed += 1
            print(f"FAIL {arg}")
            for err in errors:
                print(f"  {err}")
        else:
            print(f"ok   {arg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
