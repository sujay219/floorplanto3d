"""Tests for Phase 2, Step C4.2 (conservative wall-candidate consolidation).

Input documents are built directly so each approved C4.1 merge rule can
be exercised on exact geometry: overlapping collinear segments merge,
small gaps merge, while offset parallel faces, perpendicular junctions,
thickness conflicts, opening-sized gaps and gap chains do not. The
conservative contract is asserted throughout: uncertain candidates are
never discarded, provenance is complete, output is deterministic, inputs
stay untouched, and nothing is ever called a confirmed wall.
"""

import copy
import json
import math

from floorplanto3d.models.geometry import classify_orientation
from floorplanto3d.processing.candidate_consolidation import (
    CONSOLIDATED_CANDIDATES_FILENAME,
    CONSOLIDATION_REPORT_FILENAME,
    ConsolidationConfig,
    consolidate_candidates,
    main,
    save_artifacts,
)

WIDTH, HEIGHT = 600, 450
REQUIRED_RECORD_KEYS = {
    "id",
    "kind",
    "status",
    "geometry",
    "length",
    "angle_deg",
    "orientation",
    "estimated_thickness",
    "estimated_thickness_range_px",
    "member_count",
    "source_candidate_ids",
    "source_statuses",
    "source_geometries",
    "merge_reasons",
    "bridged_gap_total_px",
    "uncertainty",
    "limitations",
}


def candidate_record(id_, start, end, thickness):
    length = math.dist(start, end)
    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0
    return {
        "id": id_,
        "geometry": {
            "start": {"x": start[0], "y": start[1]},
            "end": {"x": end[0], "y": end[1]},
            "polygon": [],
            "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
        },
        "length": round(length, 2),
        "angle_deg": round(angle, 3),
        "orientation": classify_orientation(end[0] - start[0], end[1] - start[1]),
        "estimated_thickness": thickness,
        "thickness_consistency": 1.0,
        "straightness": 1.0,
        "continuity": 1.0,
        "edge_pairing": {
            "paired": True,
            "edge_ids": [],
            "angle_delta_deg": 0.0,
            "distance_px": thickness,
            "overlap_fraction": 1.0,
            "score": 1.0,
        },
        "axis_support": {"horizontal": 0.0, "vertical": 0.0, "combined": 0.0},
        "confidence": 1.0,
        "confidence_evidence": {},
        "evidence": [],
        "uncertainty": [],
        "merged_from": [],
    }


def build(specs):
    """specs: list of (id, start, end, thickness, status)."""
    cand_recs = [candidate_record(i, s, e, t) for i, s, e, t, _ in specs]
    filt_recs = [{"id": i, "status": st} for i, _, _, _, st in specs]
    candidates_document = {
        "phase": {"number": 3, "name": "wall_candidates", "step": "c"},
        "image": {
            "width": WIDTH,
            "height": HEIGHT,
            "normalized_width": WIDTH,
            "normalized_height": HEIGHT,
            "coordinate_space": "original",
        },
        "candidates": cand_recs,
    }
    filtered_document = {
        "phase": {"number": 3, "name": "wall_candidate_filtering", "step": "c3"},
        "image": {"width": WIDTH, "height": HEIGHT, "coordinate_space": "original"},
        "candidates": filt_recs,
    }
    return candidates_document, filtered_document


def consolidate(specs, config=None):
    candidates_document, filtered_document = build(specs)
    before = (
        json.dumps(candidates_document, sort_keys=True),
        json.dumps(filtered_document, sort_keys=True),
    )
    result = consolidate_candidates(candidates_document, filtered_document, config)
    after = (
        json.dumps(candidates_document, sort_keys=True),
        json.dumps(filtered_document, sort_keys=True),
    )
    assert before == after, "input documents must not be modified"
    return result


def record_by_id(result, id_):
    return next(r for r in result.document["candidates"] if r["id"] == id_)


# --- approved merge rules ---------------------------------------------


def test_collinear_overlapping_segments_consolidate():
    result = consolidate([
        ("cand_001", (100, 200), (200, 200), 10, "retained"),
        ("cand_002", (150, 200), (250, 200), 11, "retained"),
    ])

    records = result.document["candidates"]
    assert len(records) == 1, "overlapping duplicates must produce one output"
    record = records[0]
    assert record["kind"] == "consolidated"
    assert record["source_candidate_ids"] == ["cand_001", "cand_002"]
    assert record["geometry"]["start"]["x"] == 100 and record["geometry"]["end"]["x"] == 250
    assert record["length"] == 150
    assert record["member_count"] == 2
    assert record["merge_reasons"] and record["uncertainty"]
    mapping = result.document["source_mapping"]
    assert mapping["cand_001"] == record["id"] == mapping["cand_002"]


def test_compatible_segments_with_small_gap_consolidate():
    result = consolidate([
        ("cand_001", (100, 200), (180, 200), 10, "uncertain"),
        ("cand_002", (195, 200), (260, 200), 10, "uncertain"),
    ])

    records = result.document["candidates"]
    assert len(records) == 1
    record = records[0]
    assert record["kind"] == "consolidated"
    assert record["source_candidate_ids"] == ["cand_001", "cand_002"]
    assert record["bridged_gap_total_px"] == 15.0
    assert record["geometry"]["end"]["x"] == 260
    assert any("gap 15" in reason for reason in record["merge_reasons"])


def test_parallel_wall_faces_stay_separate():
    result = consolidate([
        ("cand_001", (100, 200), (250, 200), 6, "retained"),
        ("cand_002", (100, 212), (250, 212), 6, "retained"),
    ])

    records = result.document["candidates"]
    assert len(records) == 2, "opposite faces of a wall must never merge"
    assert all(r["kind"] == "unchanged" for r in records)
    assert {r["id"] for r in records} == {"cand_001", "cand_002"}


def test_incompatible_orientations_stay_separate():
    result = consolidate([
        ("cand_001", (100, 200), (250, 200), 10, "retained"),
        ("cand_002", (250, 200), (250, 280), 10, "retained"),
    ])

    records = result.document["candidates"]
    assert len(records) == 2
    assert all(r["kind"] == "unchanged" for r in records)
    assert result.report["summary"]["successful_merges"] == 0


def test_different_thickness_evidence_blocks_merge():
    result = consolidate([
        ("cand_001", (100, 200), (200, 200), 16, "retained"),
        ("cand_002", (215, 200), (260, 200), 2, "uncertain"),
    ])

    records = result.document["candidates"]
    assert len(records) == 2, "conflicting thickness evidence must stay separate"
    assert result.report["summary"]["rejected_proposal_reasons"][
        "thickness_incompatible_ambiguous"
    ] == 1
    ambiguous = result.report["examples"]["ambiguous_not_merged"]
    assert ambiguous and ambiguous[0]["pair"] == ["cand_001", "cand_002"]


def test_corner_tjunction_group_is_not_forced_straight():
    result = consolidate([
        ("cand_001", (100, 200), (250, 200), 10, "retained"),
        ("cand_002", (250, 200), (250, 120), 10, "retained"),
    ])

    records = result.document["candidates"]
    assert len(records) == 2
    for record in records:
        assert record["member_count"] == 1, "corner geometry must not become one segment"
    assert result.report["summary"]["successful_merges"] == 0


def test_opening_gap_is_never_bridged():
    result = consolidate([
        ("cand_001", (100, 200), (250, 200), 10, "retained"),
        ("cand_002", (310, 200), (450, 200), 10, "retained"),
    ])

    records = result.document["candidates"]
    assert len(records) == 2, "an opening-sized gap must not be bridged"
    reasons = result.report["summary"]["rejected_proposal_reasons"]
    assert reasons["opening_gap_not_bridged"] == 1
    avoided = result.report["examples"]["potential_opening_bridges_avoided"]
    assert avoided and avoided[0]["pair"] == ["cand_001", "cand_002"]


def test_chain_guard_blocks_bridging_opening_via_small_gaps():
    result = consolidate([
        ("cand_001", (100, 200), (180, 200), 10, "retained"),
        ("cand_002", (195, 200), (275, 200), 10, "retained"),
        ("cand_003", (290, 200), (370, 200), 10, "retained"),
        ("cand_004", (385, 200), (465, 200), 10, "retained"),
    ])

    records = result.document["candidates"]
    groups = [r for r in records if r["kind"] == "consolidated"]
    assert groups, "the small gaps should still consolidate locally"
    assert max(r["member_count"] for r in groups) < 4, (
        "a chain of small gaps must not bridge the full opening"
    )
    assert result.report["summary"]["rejected_proposal_reasons"]["chain_guard"] >= 1
    assert result.report["guardrail_checks"]["chain_guard_respected"] is True


# --- conservative contract --------------------------------------------


def test_uncertain_candidates_are_preserved_and_keep_their_status():
    result = consolidate([
        ("cand_001", (100, 200), (200, 200), 10, "retained"),
        ("cand_002", (150, 200), (250, 200), 10, "uncertain"),
        ("cand_003", (100, 320), (140, 320), 4, "uncertain"),
    ])

    mapping = result.document["source_mapping"]
    assert set(mapping) == {"cand_001", "cand_002", "cand_003"}
    merged = record_by_id(result, mapping["cand_001"])
    assert merged["kind"] == "consolidated"
    assert merged["status"] == "uncertain", (
        "a group containing an uncertain member must not be promoted"
    )
    assert merged["source_statuses"] == {"cand_001": "retained", "cand_002": "uncertain"}
    lone = record_by_id(result, mapping["cand_003"])
    assert lone["kind"] == "unchanged" and lone["status"] == "uncertain"


def test_complete_source_provenance_and_no_wall_claims():
    result = consolidate([
        ("cand_001", (100, 200), (200, 200), 10, "retained"),
        ("cand_002", (150, 200), (250, 200), 10, "retained"),
        ("cand_003", (100, 320), (140, 320), 4, "uncertain"),
        ("cand_004", (300, 320), (330, 320), 3, "rejected"),
    ])

    document = result.document
    mapping = document["source_mapping"]
    assert set(mapping) == {"cand_001", "cand_002", "cand_003", "cand_004"}
    assert mapping["cand_004"]["output"] is None
    assert "rejected" in mapping["cand_004"]["note"]

    covered = [
        source_id
        for record in document["candidates"]
        for source_id in record["source_candidate_ids"]
    ]
    assert sorted(covered) == ["cand_001", "cand_002", "cand_003"], (
        "every filtered-set source must appear exactly once across outputs"
    )
    for record in document["candidates"]:
        assert set(record) == REQUIRED_RECORD_KEYS
        assert record["uncertainty"], "every output must carry uncertainty"
        assert record["source_statuses"] and record["source_geometries"]
        for geometry in record["source_geometries"]:
            assert {"id", "status", "start", "end"} <= set(geometry)
    merged = record_by_id(result, mapping["cand_001"])
    assert merged["merge_reasons"]
    assert merged["estimated_thickness"] and merged["orientation"]
    assert "NOT confirmed walls" in document["status"]
    assert result.report["guardrail_checks"]["every_source_accounted_for"] is True


def test_output_is_deterministic():
    specs = [
        ("cand_001", (100, 200), (200, 200), 10, "retained"),
        ("cand_002", (150, 200), (250, 200), 10, "uncertain"),
        ("cand_003", (195, 300), (260, 300), 10, "uncertain"),
        ("cand_004", (100, 320), (140, 320), 4, "uncertain"),
    ]
    first = consolidate(specs)
    second = consolidate(specs)

    assert json.dumps(first.document, sort_keys=True) == json.dumps(second.document, sort_keys=True)
    assert json.dumps(first.report, sort_keys=True) == json.dumps(second.report, sort_keys=True)


def test_configuration_is_recorded_and_respected():
    strict = ConsolidationConfig(merge_gap_max_px=5.0)
    specs = [
        ("cand_001", (100, 200), (180, 200), 10, "retained"),
        ("cand_002", (195, 200), (260, 200), 10, "retained"),
    ]
    loose = consolidate(specs)
    tight = consolidate(specs, config=strict)

    assert loose.document["summary"]["consolidated_candidates"] == 1
    assert tight.document["summary"]["consolidated_candidates"] == 0
    assert loose.document["configuration"]["merge_gap_max_px"] == 20.0
    assert tight.document["configuration"]["merge_gap_max_px"] == 5.0


# --- artifacts, runner and real schemas --------------------------------


def test_save_artifacts_and_runner(tmp_path):
    specs = [
        ("cand_001", (100, 200), (200, 200), 10, "retained"),
        ("cand_002", (150, 200), (250, 200), 10, "retained"),
        ("cand_003", (100, 320), (140, 320), 4, "uncertain"),
    ]
    result = consolidate(specs)
    outputs = save_artifacts(result, tmp_path / "c4")
    assert set(outputs) == {"consolidated_candidates", "report"}
    for name, path in outputs.items():
        assert path.exists() and path.stat().st_size > 0, name

    candidates_document, filtered_document = build(specs)
    filtered_document["inputs"] = {"candidates": None}
    candidates_path = tmp_path / "synthetic_wall_candidates.json"
    candidates_path.write_text(json.dumps(candidates_document), encoding="utf-8")
    filtered_document["inputs"]["candidates"] = str(candidates_path)
    filtered_path = tmp_path / "filtered_candidates.json"
    filtered_path.write_text(json.dumps(filtered_document), encoding="utf-8")
    out_dir = tmp_path / "runner"

    main([str(filtered_path), "--candidates", str(candidates_path), "--out-dir", str(out_dir)])

    saved = json.loads((out_dir / CONSOLIDATED_CANDIDATES_FILENAME).read_text(encoding="utf-8"))
    report = json.loads((out_dir / CONSOLIDATION_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert saved["summary"]["output_candidates"] == len(saved["candidates"])
    assert report["summary"]["input_candidates"] == 3

    out_dir2 = tmp_path / "runner_resolved"
    main([str(filtered_path), "--out-dir", str(out_dir2)])
    assert (out_dir2 / CONSOLIDATED_CANDIDATES_FILENAME).exists(), (
        "the runner must resolve the geometry document from the filtered document"
    )


def test_real_pipeline_documents_are_accepted():
    from PIL import Image, ImageDraw

    from floorplanto3d.processing.wall_candidates import detect_candidates
    from floorplanto3d.processing.wall_detection import detect_walls

    plan = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(plan)
    draw.rectangle([60, 210, 260, 226], fill=(0, 0, 0))
    draw.rectangle([320, 210, 540, 226], fill=(0, 0, 0))
    masks = detect_walls(plan, plan)
    baseline = detect_candidates(
        plan, masks.foreground_mask, masks.horizontal, masks.vertical, masks.combined
    )
    candidates_document = baseline.candidates_document
    filtered_document = {
        "phase": {"number": 3, "name": "wall_candidate_filtering", "step": "c3"},
        "image": {
            "width": candidates_document["image"]["width"],
            "height": candidates_document["image"]["height"],
            "coordinate_space": "original",
        },
        "candidates": [
            {"id": record["id"], "status": "uncertain"}
            for record in candidates_document["candidates"]
        ],
    }

    result = consolidate_candidates(candidates_document, filtered_document)

    summary = result.document["summary"]
    assert summary["input_candidates"] == len(candidates_document["candidates"])
    assert result.report["guardrail_checks"]["every_source_accounted_for"] is True
    covered = sum(len(r["source_candidate_ids"]) for r in result.document["candidates"])
    assert covered == summary["filtered_set_candidates"]
