"""Tests for Phase 2, Step C (wall-candidate identification).

Scenes are synthetic thick strokes and thin marks on a 600x450 canvas.
Assertions use geometric tolerances (position, thickness, angle) rather
than exact counts where fragments legitimately appear, and no test claims
a candidate is a wall — only that the geometry is recovered and preserved.
"""

import json

from PIL import Image, ImageDraw

from floorplanto3d.processing.wall_candidates import (
    CANDIDATE_REPORT_FILENAME,
    CANDIDATES_OVERLAY_FILENAME,
    WALL_CANDIDATES_FILENAME,
    detect_candidates,
    main,
    save_artifacts,
)
from floorplanto3d.processing.wall_detection import detect_walls
from floorplanto3d.processing.wall_detection import save_artifacts as save_walls

WIDTH, HEIGHT = 600, 450
REQUIRED_CANDIDATE_KEYS = {
    "id",
    "geometry",
    "length",
    "angle_deg",
    "orientation",
    "estimated_thickness",
    "thickness_consistency",
    "straightness",
    "continuity",
    "edge_pairing",
    "axis_support",
    "confidence",
    "confidence_evidence",
    "evidence",
    "uncertainty",
    "merged_from",
}
REQUIRED_REPORT_KEYS = {"phase", "steps", "summary", "input", "parameters", "limitations"}


def build(draw_fn) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw_fn(ImageDraw.Draw(image))
    return image


def run(plan: Image.Image):
    masks = detect_walls(plan, plan)
    return detect_candidates(
        plan, masks.foreground_mask, masks.horizontal, masks.vertical, masks.combined
    )


def horizontal_stroke(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle([60, 210, 540, 226], fill=(0, 0, 0))


def find(candidates, orientation: str, near: tuple[float, float], radius: float = 12.0):
    matches = []
    for candidate in candidates:
        mid = (
            (candidate.start[0] + candidate.end[0]) / 2.0,
            (candidate.start[1] + candidate.end[1]) / 2.0,
        )
        if candidate.orientation != orientation:
            continue
        if abs(mid[0] - near[0]) <= radius and abs(mid[1] - near[1]) <= radius:
            matches.append(candidate)
    return matches


def test_horizontal_thick_stroke_is_one_paired_candidate():
    plan = build(horizontal_stroke)

    candidates = run(plan).candidates

    matches = find(candidates, "horizontal", (300, 218))
    assert len(matches) == 1, "one thick stroke must yield one centreline, not two edges"
    candidate = matches[0]
    assert candidate.edge_pairing["paired"] is True
    assert abs(candidate.start[1] - 218) <= 4.5
    assert abs(candidate.end[1] - 218) <= 4.5
    assert candidate.length >= 420
    assert 13.0 <= candidate.estimated_thickness <= 21.0
    assert 0.0 <= candidate.confidence <= 1.0


def test_vertical_thick_stroke_is_one_paired_candidate():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([280, 40, 296, 410], fill=(0, 0, 0))

    candidates = run(build(draw_fn)).candidates

    matches = find(candidates, "vertical", (288, 225))
    assert len(matches) == 1
    candidate = matches[0]
    assert candidate.edge_pairing["paired"] is True
    assert abs(candidate.start[0] - 288) <= 4.5
    assert abs(candidate.end[0] - 288) <= 4.5
    assert candidate.length >= 330
    assert 13.0 <= candidate.estimated_thickness <= 21.0


def test_diagonal_thick_stroke_is_detected_and_paired():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.line([(80, 80), (520, 370)], fill=(0, 0, 0), width=16)

    candidates = run(build(draw_fn)).candidates

    diagonal = [c for c in candidates if c.orientation == "diagonal"]
    assert diagonal, "diagonal strokes must be supported"
    candidate = max(diagonal, key=lambda c: c.length)
    assert abs(candidate.angle_deg - 33.4) <= 6.0
    assert candidate.length >= 450
    assert 12.0 <= candidate.estimated_thickness <= 22.0
    assert candidate.edge_pairing["paired"] is True


def test_interrupted_stroke_spans_are_preserved():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 210, 250, 226], fill=(0, 0, 0))
        draw.rectangle([350, 210, 540, 226], fill=(0, 0, 0))

    candidates = run(build(draw_fn)).candidates

    left = find(candidates, "horizontal", (155, 218), radius=15)
    right = find(candidates, "horizontal", (445, 218), radius=15)
    assert len(left) == 1, "the span left of the interruption must survive"
    assert len(right) == 1, "the span right of the interruption must survive"
    for candidate in (left[0], right[0]):
        assert candidate.continuity >= 0.9
        assert candidate.length >= 150


def test_thick_stroke_does_not_duplicate_across_its_edges():
    plan = build(horizontal_stroke)

    candidates = run(plan).candidates

    centrelines = [
        c
        for c in candidates
        if c.orientation == "horizontal"
        and abs((c.start[1] + c.end[1]) / 2.0 - 218) <= 20.0
        and c.length > 100.0
    ]
    assert len(centrelines) == 1
    assert centrelines[0].merged_from or centrelines[0].edge_pairing["paired"]


def test_thin_and_short_marks_are_preserved_as_uncertain_candidates():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.line([(60, 300), (540, 320)], fill=(0, 0, 0), width=2)
        draw.line([(60, 80), (110, 80)], fill=(0, 0, 0), width=2)

    candidates = run(build(draw_fn)).candidates

    near_thin = [
        c
        for c in candidates
        if 295.0 <= (c.start[1] + c.end[1]) / 2.0 <= 325.0
    ]
    assert near_thin, "a thin long stroke must still produce candidates"
    assert sum(c.length for c in near_thin) >= 300, (
        "fragments of a thin stroke must be preserved, together covering it"
    )
    for candidate in near_thin:
        assert candidate.estimated_thickness <= 6.0
    short = [
        c
        for c in candidates
        if 30.0 <= c.length <= 90.0
        and abs((c.start[1] + c.end[1]) / 2.0 - 80) <= 10
    ]
    assert short, "short uncertain marks must not be discarded"
    for candidate in candidates:
        assert candidate.evidence or candidate.uncertainty
        assert 0.0 <= candidate.confidence <= 1.0


def test_every_candidate_is_reported_with_geometry_and_evidence():
    plan = build(horizontal_stroke)

    result = run(plan)

    ids = [c.id for c in result.candidates]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    records = result.candidates_document["candidates"]
    assert len(records) == len(result.candidates)
    for record in records:
        assert set(record) == REQUIRED_CANDIDATE_KEYS
        geometry = record["geometry"]
        assert set(geometry) == {"start", "end", "polygon", "bbox"}
        assert record["edge_pairing"]["paired"] in (True, False)
        assert 0.0 <= record["continuity"] <= 1.0
        assert 0.0 <= record["confidence"] <= 1.0
        assert "classification" not in record, "Step C must not classify candidates"
        x, y = geometry["start"]["x"], geometry["start"]["y"]
        assert 0 <= x <= plan.width and 0 <= y <= plan.height


def test_report_records_parameters_and_declares_limitations():
    plan = build(horizontal_stroke)

    report = run(plan).report

    assert set(report) == REQUIRED_REPORT_KEYS
    assert report["phase"] == {"number": 3, "name": "wall_candidates", "step": "c"}
    assert report["summary"]["total_candidates"] >= 1
    assert (
        report["summary"]["paired_edge_candidates"]
        + report["summary"]["single_edge_candidates"]
        == report["summary"]["total_candidates"]
    )
    assert "confidence_weights" in report["parameters"]
    joined = " ".join(report["limitations"]).lower()
    assert "not" in joined and "wall" in joined


def test_save_artifacts_writes_overlay_and_documents(tmp_path):
    plan = build(horizontal_stroke)

    result = run(plan)
    outputs = save_artifacts(result, tmp_path)

    assert set(outputs) == {"overlay", "candidates", "report"}
    for name, path in outputs.items():
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    candidates = json.loads((tmp_path / WALL_CANDIDATES_FILENAME).read_text(encoding="utf-8"))
    assert candidates["phase"]["step"] == "c"
    assert candidates["image"]["coordinate_space"] == "original"
    report = json.loads((tmp_path / CANDIDATE_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["summary"]["total_candidates"] == len(candidates["candidates"])
    overlay = Image.open(tmp_path / CANDIDATES_OVERLAY_FILENAME)
    assert overlay.size == plan.size


def test_runner_reuses_existing_step_ab_artifacts(tmp_path):
    plan = build(horizontal_stroke)
    masks_dir = tmp_path / "masks"
    walls = detect_walls(plan, plan)
    save_walls(walls, masks_dir)
    image_path = tmp_path / "plan.png"
    plan.save(image_path)
    out_dir = tmp_path / "phase_3"

    main([str(image_path), str(masks_dir), str(out_dir)])

    saved = json.loads((out_dir / WALL_CANDIDATES_FILENAME).read_text(encoding="utf-8"))
    assert saved["candidates"], "the runner must produce candidates from saved masks"


def test_inputs_are_not_modified():
    plan = build(horizontal_stroke)
    original = plan.resize((300, 225))
    plan_before = plan.tobytes()
    original_before = original.tobytes()
    masks = detect_walls(plan, plan)
    mask_before = masks.foreground_mask.tobytes()

    detect_candidates(original, masks.foreground_mask, masks.horizontal, masks.vertical, masks.combined)

    assert plan.tobytes() == plan_before
    assert original.tobytes() == original_before
    assert masks.foreground_mask.tobytes() == mask_before
