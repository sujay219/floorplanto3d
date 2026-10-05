"""Unit tests for opening detection and classification."""

import cv2
import numpy as np
import pytest

from floorplanto3d.models.geometry import Point2D
from floorplanto3d.models.opening import OpeningType
from floorplanto3d.models.wall import Wall
from floorplanto3d.processing.openings import (
    _drop_speckle,
    _false_runs,
    classify_openings,
    extract_openings,
    find_gaps,
)
from floorplanto3d.processing.preprocessing import preprocess


def wall_with_gaps(gaps: list[tuple[int, int]]) -> tuple[np.ndarray, Wall]:
    """A horizontal wall at y=100 with erased spans given as (x1, x2) pairs."""
    canvas = np.full((200, 300), 255, dtype=np.uint8)
    cv2.line(canvas, (20, 100), (280, 100), 0, 8)
    for x1, x2 in gaps:
        cv2.line(canvas, (x1, 100), (x2, 100), 255, 8)

    wall = Wall(
        id="wall_001",
        start=Point2D(x=24, y=100),
        end=Point2D(x=276, y=100),
        thickness=8,
    )
    return canvas, wall


class TestRunHelpers:
    def test_false_runs_finds_each_span(self):
        # Runs are reported as inclusive (start, end) index pairs.
        mask = np.array([True, True, False, False, True, False, True])
        assert _false_runs(mask) == [(2, 3), (5, 5)]

    def test_false_runs_on_all_ink_is_empty(self):
        assert _false_runs(np.ones(5, dtype=bool)) == []

    def test_drop_speckle_keeps_a_long_gap(self):
        profile = np.array([True] * 10 + [False] * 20 + [True] * 10, dtype=bool)
        assert (~_drop_speckle(profile, 3)).sum() == 20

    def test_drop_speckle_closes_a_short_gap(self):
        profile = np.array([True] * 10 + [False] * 2 + [True] * 10, dtype=bool)
        assert _drop_speckle(profile, 3).all()

    def test_drop_speckle_leaves_a_boundary_run_alone(self):
        # A gap touching the very start must not be widened.
        profile = np.array([False] * 5 + [True] * 10, dtype=bool)
        assert (~_drop_speckle(profile, 3)).sum() == 5


class TestFindGaps:
    def test_finds_a_single_gap(self):
        canvas, wall = wall_with_gaps([(120, 160)])
        gaps = find_gaps(preprocess(canvas).binary, [wall])
        assert len(gaps) == 1
        assert gaps[0].gap_length == pytest.approx(40, abs=6)

    def test_finds_several_gaps_on_one_wall(self):
        canvas, wall = wall_with_gaps([(80, 110), (170, 200)])
        gaps = find_gaps(preprocess(canvas).binary, [wall])
        assert len(gaps) == 2

    def test_solid_wall_has_no_gaps(self):
        canvas, wall = wall_with_gaps([])
        assert find_gaps(preprocess(canvas).binary, [wall]) == []

    def test_gap_spanning_almost_the_whole_wall_is_ignored(self):
        canvas, wall = wall_with_gaps([(25, 275)])
        # That is the wall simply stopping, not an opening in it.
        assert find_gaps(preprocess(canvas).binary, [wall]) == []

    def test_gaps_are_returned_largest_first(self):
        canvas, wall = wall_with_gaps([(80, 100), (150, 200)])
        lengths = [g.gap_length for g in find_gaps(preprocess(canvas).binary, [wall])]
        assert lengths == sorted(lengths, reverse=True)


class TestClassifyOpenings:
    def _classify(self, gaps):
        canvas, wall = wall_with_gaps(gaps)
        candidates = find_gaps(preprocess(canvas).binary, [wall])
        return classify_openings(candidates, [wall])

    def test_one_long_and_one_short_gap_becomes_door_and_window(self):
        doors, windows = self._classify([(90, 160), (200, 225)])
        assert len(doors) == 1
        assert len(windows) == 1
        assert doors[0].width > windows[0].width

    def test_uniform_gaps_are_all_doors(self):
        # With no size evidence, calling them all doors beats guessing.
        doors, windows = self._classify([(70, 110), (150, 190)])
        assert len(doors) == 2
        assert windows == []

    def test_every_measured_gap_is_reported(self):
        doors, windows = self._classify([(70, 100), (130, 160), (190, 220), (240, 260)])
        assert len(doors) + len(windows) == 4

    def test_openings_carry_their_wall_id(self):
        doors, windows = self._classify([(90, 160)])
        assert all(o.wall_id == "wall_001" for o in doors + windows)

    def test_types_are_set_correctly(self):
        doors, windows = self._classify([(90, 160), (200, 225)])
        assert all(d.type is OpeningType.DOOR for d in doors)
        assert all(w.type is OpeningType.WINDOW for w in windows)

    def test_no_candidates_yields_nothing(self):
        assert classify_openings([], []) == ([], [])


class TestExtractOpenings:
    def test_no_walls_means_no_openings(self):
        binary = preprocess(np.full((100, 100), 255, dtype=np.uint8))
        assert extract_openings(binary, [], 100, 100) == ([], [])
