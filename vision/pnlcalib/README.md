# Vendored PnLCalib

From github.com/mguti97/PnLCalib, commit `8c87391`, GPL-2.0 (`LICENSE`). Only what inference needs:
`model/cls_hrnet.py`, `model/cls_hrnet_l.py`, `utils/utils_calib.py`, `utils/utils_heatmap.py`,
`utils/utils_optimize.py` and `config/hrnetv2_w48*.yaml`, flattened into this folder.

Changes from upstream, nothing else:
- `utils_calib.py`: `from utils.utils_optimize` → `from .utils_optimize`
- `utils_optimize.py`: the unused `import matplotlib.pyplot as plt` removed

Vendored rather than imported from a checkout so replay (which reruns the voting on CPU) is pinned
by the run's git commit (03). Weights (`SV_FT_WC14_kp`, `SV_FT_WC14_lines`, ...) stay outside git:
github.com/mguti97/PnLCalib/releases/tag/v1.0.0. Lint is off for this folder (`pyproject.toml`).
