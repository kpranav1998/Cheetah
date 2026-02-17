from __future__ import annotations

import pandas as pd

from patterns.base import BasePatternDetector, PatternMatch
from patterns.double_bottom import DoubleBottomDetector
from patterns.double_top import DoubleTopDetector
from patterns.head_shoulders import HeadShouldersDetector
from patterns.flag_pole import FlagPoleDetector
from patterns.cup_handle import CupHandleDetector
from patterns.triangles import TriangleDetector


ALL_DETECTORS: list[type[BasePatternDetector]] = [
    DoubleBottomDetector,
    DoubleTopDetector,
    HeadShouldersDetector,
    FlagPoleDetector,
    CupHandleDetector,
    TriangleDetector,
]

DETECTOR_MAP: dict[str, type[BasePatternDetector]] = {
    cls.name: cls for cls in ALL_DETECTORS
}


def scan_patterns(
    df: pd.DataFrame,
    pattern_names: list[str] | None = None,
) -> list[PatternMatch]:
    """Run pattern detectors on a DataFrame. If pattern_names is None, run all."""
    if pattern_names is None:
        detectors = [cls() for cls in ALL_DETECTORS]
    else:
        detectors = [DETECTOR_MAP[name]() for name in pattern_names if name in DETECTOR_MAP]

    all_matches: list[PatternMatch] = []
    for detector in detectors:
        matches = detector.detect(df)
        all_matches.extend(matches)

    return sorted(all_matches, key=lambda m: m.confirmation_timestamp)
