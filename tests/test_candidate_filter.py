"""Tests for Phase 2, Step C3 (conservative wall-candidate filtering).

Scenes are synthetic 600x450 plans run through the real Step C candidate
pipeline; text regions are supplied as documents so the filtering rules
can be exercised deterministically. Assertions check the conservative
contract: short legitimate stubs, diagonal strokes, paired edges and
candidates near openings survive; overlap with a text box alone never
causes rejection; insufficient evidence yields uncertain (kept); text
clutter is rejected but stays fully recoverable with its audit trail.
"""

import copy
import json

from PIL import Image, ImageDraw

from floorplanto3d.processing.candidate_filter import (
    COMPARISON_FILENAME,
    FILTERED_CANDIDATES_FILENAME,
    FILTERING_REPORT_FILENAME,
    ORIGINAL_OVERLAY_FILENAME,
    RETAINED_OVERLAY_FILENAME,
    REJECTED_OVERLAY_FILENAME,
    STATUS_REJECTED,
    STATUS_RETAINED,
    STATUS_UNCERTAIN,
    filter_candidates,
    main,
    save_artifacts,
)
from floorplanto3d.processing.wall_candidates import detect_candidates
from floorplanto3d.processing.wall_detection import detect_walls

WIDTH, HEIGHT = 600, 450
REQUIRED_RECORD_KEYS = {
    "id",
    "status",
    "features",
    "retain_evidence",
    "reject_evidence",
    "decision_reasons",
    "fired_rules",
    "text_region_ids",
}
REQUIRED_FEATURE_KEYS = {
    "length_px",
    "estimated_thickness_px",
    "slenderness",
    "angle_deg",
    "orientation",
    "paired_edges",
    "pairing_score",
    "pairing_overlap_fraction",
    "pairing_edge_distance_px",
    "continuity",
    "straightness",
    "thickness_consistency",
    "text_region_coverage",
    "structure_connection_count",
    "collinear_overlap_count",
    "short_neighbour_count",
}


def build(draw_fn) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw_fn(ImageDraw.Draw(image))
    return image


def text_document(boxes) -> dict:
    return {
        "image": {
            "width": WIDTH,
            "height": HEIGHT,
            "coordinate_space": "original",
        },
        "regions": [
            {"id": f"text_{index + 1:03d}", "bbox": {"x": x, "y": y, "width": w, "height": h}}
            for index, (x, y, w, h) in enumerate(boxes)
        ],
    }


def run(plan: Image.Image, boxes=()):
    masks = detect_walls(plan, plan)
    candidates = detect_candidates(
        plan, masks.foreground_mask, masks.horizontal, masks.vertical, masks.combined
    )
    result = filter_candidates(plan, candidates.candidates_document, text_document(boxes))
    return result


def records(result):
    return {record["id"]: record for record in result.document["candidates"]}


def near(result, point, radius=18.0):
    matches = []
    for record in result.document["candidates"]:
        start, end = record["_start"], record["_end"]
        mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        if abs(mid[0] - point[0]) <= radius and abs(mid[1] - point[1]) <= radius:
            matches.append(record)
    return matches


def attach_geometry(result, plan):
    """Expose candidate midpoints for the find helpers (test-side only)."""
    masks = detect_walls(plan, plan)
    candidates = detect_candidates(
        plan, masks.foreground_mask, masks.horizontal, masks.vertical, masks.combined
    )
    geometry = {
        record["id"]: (
            (record["geometry"]["start"]["x"], record["geometry"]["start"]["y"]),
            (record["geometry"]["end"]["x"], record["geometry"]["end"]["y"]),
        )
        for record in candidates.candidates_document["candidates"]
    }
    for record in result.document["candidates"]:
        record["_start"], record["_end"] = geometry[record["id"]]
    return result


# --- conservative contract --------------------------------------------


def test_short_legitimate_wall_stub_connected_to_wall_is_not_rejected():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 210, 540, 226], fill=(0, 0, 0))
        draw.rectangle([300, 226, 316, 270], fill=(0, 0, 0))

    plan = build(draw_fn)
    result = attach_geometry(run(plan), plan)

    stub = near(result, (308, 248))
    assert stub, "the short stub must produce candidate(s)"
    for record in stub:
        assert record["status"] != STATUS_REJECTED, (
            "a short wall stub joined to a wall must never be rejected"
        )
    joined = [r for r in stub if any("connected_structure" in e or "sturdy_stroke" in e
                                    for e in r["retain_evidence"])]
    assert joined, "the stub must be kept with retention evidence"


def test_diagonal_wall_candidate_is_not_rejected_for_being_diagonal():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.line([(80, 80), (520, 370)], fill=(0, 0, 0), width=16)

    plan = build(draw_fn)
    result = attach_geometry(run(plan), plan)

    diagonal = [
        record for record in result.document["candidates"]
        if record["features"]["orientation"] == "diagonal"
    ]
    assert diagonal, "the diagonal stroke must produce diagonal candidates"
    for record in diagonal:
        assert record["status"] != STATUS_REJECTED, (
            "a diagonal candidate must not be rejected for its orientation"
        )
    assert any(record["status"] == STATUS_RETAINED for record in diagonal)


def test_text_box_overlap_alone_never_rejects():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 210, 540, 226], fill=(0, 0, 0))

    plan = build(draw_fn)
    result = attach_geometry(run(plan, boxes=[(250, 198, 140, 44)]), plan)

    crossing = [
        record for record in result.document["candidates"]
        if record["features"]["text_region_coverage"] > 0
    ]
    assert crossing, "the wall must overlap the synthetic text region"
    for record in crossing:
        assert record["status"] != STATUS_REJECTED, (
            "overlapping a text region alone must never reject a candidate"
        )
        assert record["text_region_ids"], "overlapping text region IDs must be recorded"


def test_short_thin_mark_overlapping_text_survives_when_connected():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 214, 540, 222], fill=(0, 0, 0))
        draw.rectangle([300, 226, 304, 266], fill=(0, 0, 0))

    plan = build(draw_fn)
    result = attach_geometry(run(plan, boxes=[(294, 222, 18, 50)]), plan)

    inside = [
        record for record in result.document["candidates"]
        if record["features"]["text_region_coverage"] >= 0.5
    ]
    assert inside, "the thin mark must be covered by the synthetic text region"
    for record in inside:
        assert record["features"]["structure_connection_count"] >= 1, (
            "the mark must be recorded as connected to the wall"
        )
        assert record["status"] != STATUS_REJECTED, (
            "text overlap stacked with short+thin is still not enough when the "
            "mark connects to substantial structure"
        )


def test_paired_wall_edges_are_retained_with_evidence():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 210, 540, 226], fill=(0, 0, 0))

    plan = build(draw_fn)
    result = attach_geometry(run(plan), plan)

    wall = near(result, (300, 218), radius=10.0)
    assert wall
    paired = [r for r in wall if r["features"]["paired_edges"]]
    assert paired, "a thick stroke must yield a paired-edge candidate"
    for record in paired:
        assert record["status"] == STATUS_RETAINED
        assert any("paired_edges" in e for e in record["retain_evidence"])


def test_candidates_near_openings_are_preserved():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 210, 260, 226], fill=(0, 0, 0))
        draw.rectangle([320, 210, 540, 226], fill=(0, 0, 0))

    plan = build(draw_fn)
    result = attach_geometry(run(plan), plan)

    left = near(result, (160, 218))
    right = near(result, (430, 218))
    assert left and right, "both spans beside the opening must produce candidates"
    for record in left + right:
        assert record["status"] != STATUS_REJECTED, (
            "wall spans beside an opening must be preserved"
        )


def test_insufficient_evidence_is_kept_as_uncertain():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 60, 76, 200], fill=(0, 0, 0))
        draw.rectangle([250, 380, 280, 383], fill=(0, 0, 0))

    plan = build(draw_fn)
    result = attach_geometry(run(plan), plan)

    lone = near(result, (265, 381), radius=14.0)
    assert lone, "the lone short thin mark must produce a candidate"
    statuses = {record["status"] for record in lone}
    assert statuses <= {STATUS_UNCERTAIN, STATUS_RETAINED}
    assert STATUS_UNCERTAIN in statuses, (
        "a candidate with insufficient evidence must be kept as uncertain"
    )
    for record in lone:
        assert record["status"] != STATUS_REJECTED


def test_text_clutter_is_rejected_but_fully_recoverable():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 60, 76, 200], fill=(0, 0, 0))
        for index in range(6):
            x = 100 + index * 26
            draw.rectangle([x, 330, x + 4, 352], fill=(0, 0, 0))

    plan = build(draw_fn)
    result = attach_geometry(run(plan, boxes=[(92, 322, 168, 40)]), plan)

    rejected = [
        record for record in result.document["candidates"]
        if record["status"] == STATUS_REJECTED
    ]
    assert rejected, "glyph-like clutter inside a text region must be rejected"
    for record in rejected:
        assert "text_clutter" in record["fired_rules"]
        assert record["text_region_ids"]
        assert record["reject_evidence"] and record["decision_reasons"]
        assert REQUIRED_RECORD_KEYS <= set(record), "rejected candidates stay auditable"
    kept = [
        record for record in result.document["candidates"]
        if record["status"] != STATUS_REJECTED
    ]
    assert kept, "the unrelated wall in the same scene must survive"


# --- audit and scope ---------------------------------------------------


def test_every_candidate_gets_an_auditable_record():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 210, 540, 226], fill=(0, 0, 0))
        draw.line([(80, 300), (300, 400)], fill=(0, 0, 0), width=6)
        for index in range(5):
            x = 350 + index * 24
            draw.rectangle([x, 320, x + 4, 340], fill=(0, 0, 0))

    plan = build(draw_fn)
    masks = detect_walls(plan, plan)
    baseline = detect_candidates(
        plan, masks.foreground_mask, masks.horizontal, masks.vertical, masks.combined
    )
    candidates_document = copy.deepcopy(baseline.candidates_document)
    text_doc = text_document([(340, 312, 130, 36)])
    before = json.dumps(candidates_document, sort_keys=True), json.dumps(text_doc, sort_keys=True)

    result = filter_candidates(plan, candidates_document, text_doc)

    after = json.dumps(candidates_document, sort_keys=True), json.dumps(text_doc, sort_keys=True)
    assert before == after, "inputs must not be modified by filtering"
    records_list = result.document["candidates"]
    assert len(records_list) == len(baseline.candidates) == (
        result.report["summary"]["original_candidates"]
    )
    summary = result.document["summary"]
    assert summary["retained_candidates"] + summary["rejected_candidates"] + summary[
        "uncertain_candidates"
    ] == summary["original_candidates"]
    assert summary["filtered_set_count"] == (
        summary["retained_candidates"] + summary["uncertain_candidates"]
    )
    ids = [record["id"] for record in records_list]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    for record in records_list:
        assert set(record) == REQUIRED_RECORD_KEYS
        assert set(record["features"]) == REQUIRED_FEATURE_KEYS
        assert record["status"] in {STATUS_RETAINED, STATUS_REJECTED, STATUS_UNCERTAIN}
        assert record["decision_reasons"], "every decision must state its reason"
        assert "classification" not in record, "candidates must never be classified as walls"
    rejected_ids = [r["id"] for r in records_list if r["status"] == STATUS_REJECTED]
    assert set(rejected_ids) <= set(ids), "rejected candidates remain in the output"
    assert result.report["summary"]["rule_contributions"] is not None
    assert result.report["known_failure_cases"]


def test_input_coordinate_mismatch_is_refused():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 210, 540, 226], fill=(0, 0, 0))

    plan = build(draw_fn)
    masks = detect_walls(plan, plan)
    baseline = detect_candidates(
        plan, masks.foreground_mask, masks.horizontal, masks.vertical, masks.combined
    )
    bad_text = text_document([])
    bad_text["image"]["width"] = 999

    try:
        filter_candidates(plan, baseline.candidates_document, bad_text)
    except ValueError as error:
        assert "coordinate" in str(error) or "size" in str(error)
    else:
        raise AssertionError("mismatched coordinate systems must be refused")


def test_save_artifacts_writes_all_six_outputs_and_runner(tmp_path):
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 210, 540, 226], fill=(0, 0, 0))
        for index in range(6):
            x = 350 + index * 26
            draw.rectangle([x, 320, x + 4, 342], fill=(0, 0, 0))

    plan = build(draw_fn)
    result = run(plan, boxes=[(340, 312, 160, 40)])
    outputs = save_artifacts(result, tmp_path / "c3")

    assert set(outputs) == {
        "original_overlay",
        "retained_overlay",
        "rejected_overlay",
        "comparison",
        "filtered_candidates",
        "report",
    }
    for name, path in outputs.items():
        assert path.exists() and path.stat().st_size > 0, name
    for name in (ORIGINAL_OVERLAY_FILENAME, RETAINED_OVERLAY_FILENAME, REJECTED_OVERLAY_FILENAME):
        assert Image.open(tmp_path / "c3" / name).size == plan.size
    comparison = Image.open(tmp_path / "c3" / COMPARISON_FILENAME)
    assert comparison.size[0] > plan.size[0], "the comparison holds two panels"

    document = json.loads((tmp_path / "c3" / FILTERED_CANDIDATES_FILENAME).read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "c3" / FILTERING_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["summary"]["original_candidates"] == len(document["candidates"])

    image_path = tmp_path / "plan.png"
    plan.save(image_path)
    candidates_json = tmp_path / "wall_candidates.json"
    candidates_json.write_text(json.dumps(_baseline_document(plan)), encoding="utf-8")
    text_json = tmp_path / "text_regions.json"
    text_json.write_text(json.dumps(text_document([(340, 312, 160, 40)])), encoding="utf-8")
    missing_mask = tmp_path / "no_mask.png"
    out_dir = tmp_path / "runner"

    main([str(image_path), str(candidates_json), str(text_json), str(missing_mask), str(out_dir)])

    assert (out_dir / FILTERED_CANDIDATES_FILENAME).exists()
    assert (out_dir / FILTERING_REPORT_FILENAME).exists()


def _baseline_document(plan: Image.Image) -> dict:
    masks = detect_walls(plan, plan)
    baseline = detect_candidates(
        plan, masks.foreground_mask, masks.horizontal, masks.vertical, masks.combined
    )
    return baseline.candidates_document


def test_inputs_are_not_modified():
    def draw_fn(draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([60, 210, 540, 226], fill=(0, 0, 0))

    plan = build(draw_fn)
    plan_before = plan.tobytes()
    masks = detect_walls(plan, plan)
    mask_before = masks.foreground_mask.tobytes()
    candidates_document = _baseline_document(plan)
    text_doc = text_document([])

    filter_candidates(plan, candidates_document, text_doc)

    assert plan.tobytes() == plan_before
    assert masks.foreground_mask.tobytes() == mask_before
