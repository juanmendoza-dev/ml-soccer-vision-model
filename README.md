# ml-soccer-vision-model

Predicts, a few seconds ahead, when a goal is likely in a soccer match, from broadcast video plus player profiles. Specs are in `Docs/Specs/`; start with `00-overview.md`.

## Setup
```
uv venv --python 3.11
uv pip install -e ".[dev,converters]"   # add prediction / vision extras when needed
.venv/bin/pytest
```

Data download steps are in `data/README.md`.

Check a converted match against the schema:
```
.venv/bin/python -m gamestate.validate data/gamestate/<match_id>
```
