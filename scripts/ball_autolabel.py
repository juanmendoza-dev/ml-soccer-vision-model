"""Ball auto-labels from PFF for the detector fine-tune (10-ball 3a). The code is in
vision/ball_autolabel.py; this is its command line.

    PYTHONPATH=. python scripts/ball_autolabel.py --sync-check     # vb01-vb03 (F1)

The sync check needs each bench clip's run cache with balls.parquet, its PFF reference
(python -m vision.ball_truth --clip <clip>) and the label frames that builds. It exits
non-zero when a clip's peak isn't at the manifest's offset to within one PFF frame.
"""

from vision.ball_autolabel import main

if __name__ == "__main__":
    main()
