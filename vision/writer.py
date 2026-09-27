"""VisionFrames -> game state (02) + detections cache and view.parquet (03 Diagnostics)."""

import json
import time
from pathlib import Path

import polars as pl

from converters.common import causal_velocities, git_commit
from gamestate.schema import SCHEMA_VERSION
from gamestate.validate import validate_match
from vision.config import VisionConfig
from vision.types import BALL, GOALKEEPER, PLAYER, REFEREE, VisionFrame

OBJECT_TYPES = {PLAYER: "player", GOALKEEPER: "goalkeeper", REFEREE: "referee", BALL: "ball"}

# 03 Diagnostics; typed so a run with no detections still writes the columns
DETECTIONS_SCHEMA = {
    "match_id": pl.String,
    "frame_id": pl.Int64,
    "object_id": pl.String,
    "class": pl.String,
    "team_cluster": pl.Int64,
    "x1": pl.Float64,
    "y1": pl.Float64,
    "x2": pl.Float64,
    "y2": pl.Float64,
    "det_confidence": pl.Float64,
    "tracked_only": pl.Boolean,
    "pitch_x": pl.Float64,
    "pitch_y": pl.Float64,
    "homography_ok": pl.Boolean,
    "homography_err_m": pl.Float64,
}
VIEW_SCHEMA = {
    "match_id": pl.String,
    "frame_id": pl.Int64,
    "view": pl.String,
    "grass_share": pl.Float64,
    "keypoints_found": pl.Int64,
}


class GameStateWriter:
    def __init__(
        self,
        match_id: str,
        home_team: str,
        away_team: str,
        native_fps: float,
        config: VisionConfig,
        gamestate_dir: Path,
        cache_dir: Path | None = None,
        run_info: dict | None = None,
    ):
        self.match_id = match_id
        self.home_team, self.away_team = home_team, away_team
        self.native_fps = native_fps
        self.config = config
        self.out = Path(gamestate_dir) / match_id
        self.cache = Path(cache_dir) / match_id if cache_dir else None
        self.run_info = run_info or {}
        self._frames: list[dict] = []
        self._objects: list[dict] = []
        self._detections: list[dict] = []
        self._views: list[dict] = []
        self._started = time.time()
        self.errors: list[str] = []  # 02 validator output, set by close()

    def add(self, vf: VisionFrame) -> None:
        period = self.config.period
        self._frames.append(
            {
                "match_id": self.match_id,
                "frame_id": vf.frame_id,
                "period": period,
                "timestamp_s": vf.t,
                # +x is home's period-1 attack (02); they switch ends each period
                "home_attacks_positive_x": period % 2 == 1,
                # stage 8 (possession / ball state) isn't built yet: can't decide -> null (02)
                "ball_state": None,
                "possession_team": None,
                "ball_carrier_id": None,
                "view_polygon": vf.view_polygon,
                "set_play_phase": None,  # vision can't produce it (02)
            }
        )
        self._views.append(
            {
                "match_id": self.match_id,
                "frame_id": vf.frame_id,
                "view": vf.view,
                "grass_share": vf.grass_share,
                "keypoints_found": vf.keypoints_found,
            }
        )
        for o in [*vf.objects, *([vf.ball] if vf.ball else [])]:
            self._detections.append(
                {
                    "match_id": self.match_id,
                    "frame_id": vf.frame_id,
                    "object_id": o.object_id,
                    "class": o.cls,
                    "team_cluster": o.cluster,
                    "x1": o.box_px[0],
                    "y1": o.box_px[1],
                    "x2": o.box_px[2],
                    "y2": o.box_px[3],
                    "det_confidence": None if o.tracked_only else o.confidence,
                    "tracked_only": o.tracked_only,
                    "pitch_x": o.x,
                    "pitch_y": o.y,
                    "homography_ok": vf.homography_ok,
                    "homography_err_m": vf.homography_err_m,
                }
            )
            if o.x is None:
                continue  # no homography: no position in game state, never a guess (02)
            self._objects.append(
                {
                    "match_id": self.match_id,
                    "frame_id": vf.frame_id,
                    "object_id": o.object_id,
                    "object_type": OBJECT_TYPES[o.cls],
                    "team": o.team,
                    "player_id": None,  # until jersey OCR (stage 6)
                    "x": o.x,
                    "y": o.y,
                    "z": None,
                    "visible": True,
                    "interpolated": o.interpolated,
                    "confidence": o.confidence,
                }
            )

    def close(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        frames = pl.DataFrame(
            self._frames,
            schema={
                "match_id": pl.String,
                "frame_id": pl.Int64,
                "period": pl.Int64,
                "timestamp_s": pl.Float64,
                "home_attacks_positive_x": pl.Boolean,
                "ball_state": pl.String,
                "possession_team": pl.String,
                "ball_carrier_id": pl.String,
                "view_polygon": pl.List(pl.Float64),
                "set_play_phase": pl.Boolean,
            },
        )
        objects = pl.DataFrame(
            self._objects,
            schema={
                "match_id": pl.String,
                "frame_id": pl.Int64,
                "object_id": pl.String,
                "object_type": pl.String,
                "team": pl.String,
                "player_id": pl.String,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
                "visible": pl.Boolean,
                "interpolated": pl.Boolean,
                "confidence": pl.Float64,
            },
        )
        if objects.height:
            objects = causal_velocities(objects, frames, fps=self.native_fps)
        else:
            objects = objects.with_columns(vx=pl.lit(None, pl.Float64), vy=pl.lit(None, pl.Float64))
        match = pl.DataFrame(
            {
                "match_id": [self.match_id],
                "schema_version": [SCHEMA_VERSION],
                "source": ["vision"],
                "competition": [None],
                "season": [None],
                "date": [None],
                "home_team": [self.home_team],
                "away_team": [self.away_team],
                "native_fps": [float(self.native_fps)],
            },
            schema_overrides={"competition": pl.String, "season": pl.String, "date": pl.Date},
        )
        events = pl.DataFrame(
            schema={
                "match_id": pl.String,
                "frame_id": pl.Int64,
                "event_type": pl.String,
                "team": pl.String,
                "player_id": pl.String,
                "x": pl.Float64,
                "y": pl.Float64,
                "outcome": pl.String,
                "set_piece": pl.String,
                "set_play_phase": pl.Boolean,
            }
        )
        players = pl.DataFrame(
            schema={
                "match_id": pl.String,
                "player_id": pl.String,
                "team": pl.String,
                "jersey_number": pl.Int64,
                "position": pl.String,
                "name": pl.String,
            }
        )
        for name, df in [
            ("match", match),
            ("frames", frames),
            ("objects", objects),
            ("events", events),
            ("players", players),
        ]:
            df.write_parquet(self.out / f"{name}.parquet")
        # An all-other clip (no objects) fails here on purpose: a run that saw no
        # match is a failed run, not a valid empty one
        self.errors = validate_match(self.out)

        if self.cache is not None:
            self.cache.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(self._detections, schema=DETECTIONS_SCHEMA).write_parquet(
                self.cache / "detections.parquet"
            )
            pl.DataFrame(self._views, schema=VIEW_SCHEMA).write_parquet(self.cache / "view.parquet")
            run = {
                **self.run_info,
                "config": self.config.__dict__,
                "git_commit": git_commit(),
                "wall_time_s": time.time() - self._started,
                "validation_errors": self.errors,
            }
            (self.cache / "run.json").write_text(json.dumps(run, indent=2, default=str))
        return self.out
