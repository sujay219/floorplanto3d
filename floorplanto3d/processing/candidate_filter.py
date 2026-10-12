"""Phase 2, Step C3: conservative wall-candidate filtering (experimental).

This stage reads the Step C wall candidates and the C2 proposed text
regions and decides, for every candidate, whether to **retain**, mark
**uncertain**, or **reject** it in a filtered candidate set. It is an
experiment: filtering is not assumed to improve anything, every decision
is written out with the features and evidence behind it, and nothing is
ever classified as a confirmed wall.

Design rules that shape the code:

* The baseline is immutable. The Step C ``wall_candidates.json`` and the
  C2 artifacts are read-only; the filter writes only its own outputs and
  keeps every rejected candidate recoverable in ``filtered_candidates.json``.
* Rejection requires several independent weaknesses stacked together.
  No candidate is rejected solely for overlapping a proposed text region,
  solely for being short, or solely for being diagonal, and candidates
  near corners and openings survive because they connect to substantial
  neighbours.
* C2 text regions are coarse bounding-box proposals, so text overlap is
  weak negative evidence: it only participates in a rejection together
  with short length, thin strokes and isolation from structure.
* A candidate with paired edges (two parallel faces of one stroke) is
  protected from clutter rejection entirely; at worst it stays uncertain.
* Scores and statuses are uncalibrated heuristics. They order evidence;
  they are not probabilities, and no precision or accuracy claim is made
  without labelled ground truth.

The per-candidate feature set mixes what Step C already measured (length,
thickness, continuity, straightness, thickness consistency, edge pairing)
with what this stage derives: text-region coverage of the centreline,
connections to substantial nearby candidates, collinear overlap with
other candidates, and the local density of short strokes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from floorplanto3d.processing.image import load_image, open_image
from floorplanto3d.processing.normalize import png_bytes

logger = logging.getLogger(__name__)

# Decision statuses.
STATUS_RETAINED = "retained"
STATUS_REJECTED = "rejected"
STATUS_UNCERTAIN = "uncertain"

# Rejection rules. Each rule stacks several weaknesses; the thresholds are
# starting values and every one of them is recorded in the report.
R1_TEXT_CLUTTER = "text_clutter"
R2_ISOLATED_TEXTURE_MARK = "isolated_texture_mark"

# Geometry for the derived features.
TEXT_COVERAGE_SAMPLES = 50
CONNECT_MAX_DIST_PX = 12.0
CONNECT_MIN_OTHER_LENGTH_PX = 60.0
SHORT_NEIGHBOUR_RADIUS_PX = 70.0
SHORT_NEIGHBOUR_MAX_LENGTH_PX = 45.0
COLLINEAR_ANGLE_TOL_DEG = 4.0
COLLINEAR_LATERAL_TOL_PX = 4.0
COLLINEAR_GAP_PX = 15.0

# Retention evidence. Two pieces of evidence make a candidate retained;
# one piece leaves it uncertain. Retention never asserts "this is a wall".
K1_PAIRED_EDGES = "paired_edges"
K2_STURDY_STROKE = "sturdy_stroke"
K3_CONNECTED_STRUCTURE = "connected_structure"
K4_COLLINEAR_CORROBORATION = "collinear_corroboration"
K5_SUBSTANTIAL_LENGTH = "substantial_length"

PAIRED_SCORE_MIN = 0.5
STURDY_THICKNESS_PX = 5.0
STRONG_CONTINUITY = 0.85
STRONG_STRAIGHTNESS = 0.75
MODERATE_CONTINUITY = 0.7
MODERATE_STRAIGHTNESS = 0.7
SUBSTANTIAL_LENGTH_PX = 80.0
MIN_RETAIN_EVIDENCES = 2
CONNECTED_STRUCTURE_MIN = 2

# Overlay palette (RGB) and layout.
COLOR_RETAINED = (46, 160, 90)
COLOR_UNCERTAIN = (240, 170, 40)
COLOR_REJECTED = (210, 60, 60)
COLOR_BASELINE = (70, 110, 190)
COLOR_TEXT_BOX = (236, 76, 120)
COLOR_TEXT = (32, 32, 32)
COLOR_BANNER_BG = (255, 255, 255)
LINE_THICKNESS = 2
THIN_LINE_THICKNESS = 1
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LABEL_GAP_PX = 4

# Saved artifact filenames.
ORIGINAL_OVERLAY_FILENAME = "original_candidates_overlay.png"
RETAINED_OVERLAY_FILENAME = "retained_candidates_overlay.png"
REJECTED_OVERLAY_FILENAME = "rejected_candidates_overlay.png"
COMPARISON_FILENAME = "before_after_comparison.png"
FILTERED_CANDIDATES_FILENAME = "filtered_candidates.json"
FILTERING_REPORT_FILENAME = "filtering_report.json"


@dataclass(frozen=True)
class FilterConfig:
    """Every filtering threshold, so decisions stay reproducible."""

    r1_text_coverage_min: float = 0.85
    r1_max_length_px: float = 45.0
    r1_max_thickness_px: float = 5.0
    r2_max_length_px: float = 35.0
    r2_max_thickness_px: float = 5.0
    r2_min_short_neighbours: int = 5
    connect_max_dist_px: float = CONNECT_MAX_DIST_PX
    connect_min_other_length_px: float = CONNECT_MIN_OTHER_LENGTH_PX
    short_neighbour_radius_px: float = SHORT_NEIGHBOUR_RADIUS_PX
    short_neighbour_max_length_px: float = SHORT_NEIGHBOUR_MAX_LENGTH_PX
    paired_score_min: float = PAIRED_SCORE_MIN
    sturdy_thickness_px: float = STURDY_THICKNESS_PX
    strong_continuity: float = STRONG_CONTINUITY
    strong_straightness: float = STRONG_STRAIGHTNESS
    moderate_continuity: float = MODERATE_CONTINUITY
    moderate_straightness: float = MODERATE_STRAIGHTNESS
    substantial_length_px: float = SUBSTANTIAL_LENGTH_PX
    min_retain_evidences: int = MIN_RETAIN_EVIDENCES
    connected_structure_min: int = CONNECTED_STRUCTURE_MIN


@dataclass
class FilterDecision:
    """The auditable decision for one original candidate."""

    id: str
    status: str
    start: tuple[float, float]
    end: tuple[float, float]
    features: dict[str, Any]
    retain_evidence: list[str]
    reject_evidence: list[str]
    decision_reasons: list[str]
    fired_rules: list[str]
    text_region_ids: list[str]


@dataclass
class FilteringResult:
    """Outputs of Step C3."""

    decisions: list[FilterDecision]
    document: dict[str, Any]
    report: dict[str, Any]
    original_overlay: Image.Image
    retained_overlay: Image.Image
    rejected_overlay: Image.Image
    comparison: Image.Image


def filter_candidates(
    original: str | Path | bytes | bytearray | Image.Image,
    candidates_document: dict[str, Any],
    text_regions_document: dict[str, Any],
    text_mask: Image.Image | None = None,
    config: FilterConfig | None = None,
) -> FilteringResult:
    """Evaluate every Step C candidate and produce the filtered set.

    Args:
        original: The original floor plan image; background of every
            overlay and the coordinate space of all output. Never modified.
        candidates_document: The ``wall_candidates.json`` document.
        text_regions_document: The C2 ``text_regions.json`` document.
        text_mask: The C2 ``text_regions_mask.png``, optionally supplied
            to cross-check the region bounding boxes.
        config: Thresholds; defaults are the recorded starting values.

    Returns:
        A :class:`FilteringResult`. Every original candidate appears in
        ``document["candidates"]`` with its status and audit trail, so
        rejected candidates remain recoverable. Nothing is classified as
        a confirmed wall.
    """
    config = config or FilterConfig()
    original_image = open_image(original)
    base_rgb, original_width, original_height, _ = load_image(original_image)
    _verify_inputs(original_width, original_height, candidates_document, text_regions_document)
    mask_consistent = _verify_text_mask(text_mask, text_regions_document)

    candidates = _load_candidates(candidates_document)
    text_boxes = _load_text_boxes(text_regions_document)

    decisions = [_evaluate(candidate, candidates, text_boxes, config) for candidate in candidates]

    document = _build_document(
        decisions, candidates_document, text_regions_document, original_width, original_height,
        config, mask_consistent,
    )
    report = _build_report(decisions, candidates_document, text_regions_document, config, mask_consistent)

    original_overlay = _draw_overlay(
        base_rgb, decisions, text_boxes,
        {STATUS_RETAINED, STATUS_REJECTED, STATUS_UNCERTAIN},
        status_colors={s: COLOR_BASELINE for s in (STATUS_RETAINED, STATUS_REJECTED, STATUS_UNCERTAIN)},
        label_ids=False, show_text_boxes=False,
        title="Step C3 - original wall candidates (baseline, unfiltered)",
    )
    retained_overlay = _draw_overlay(
        base_rgb, decisions, text_boxes, {STATUS_RETAINED, STATUS_UNCERTAIN},
        status_colors={STATUS_RETAINED: COLOR_RETAINED, STATUS_UNCERTAIN: COLOR_UNCERTAIN},
        label_ids=False, show_text_boxes=True,
        title="Step C3 - filtered candidate set (retained + kept uncertain; proposals, not confirmed walls)",
    )
    rejected_overlay = _draw_overlay(
        base_rgb, decisions, text_boxes, {STATUS_REJECTED},
        status_colors={STATUS_REJECTED: COLOR_REJECTED},
        label_ids=True, show_text_boxes=True,
        title="Step C3 - rejected candidates (excluded from the filtered set; recoverable in filtered_candidates.json)",
    )
    comparison = _draw_comparison(base_rgb, decisions, text_boxes)

    logger.info(
        "wall-candidate filtering (Step C3) complete: %d original -> %d retained, "
        "%d uncertain, %d rejected",
        len(decisions),
        report["summary"]["retained_candidates"],
        report["summary"]["uncertain_candidates"],
        report["summary"]["rejected_candidates"],
    )
    return FilteringResult(
        decisions=decisions,
        document=document,
        report=report,
        original_overlay=original_overlay,
        retained_overlay=retained_overlay,
        rejected_overlay=rejected_overlay,
        comparison=comparison,
    )


def save_artifacts(result: FilteringResult, directory: str | Path) -> dict[str, Path]:
    """Write the six Step C3 artifacts into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "original_overlay": directory / ORIGINAL_OVERLAY_FILENAME,
        "retained_overlay": directory / RETAINED_OVERLAY_FILENAME,
        "rejected_overlay": directory / REJECTED_OVERLAY_FILENAME,
        "comparison": directory / COMPARISON_FILENAME,
        "filtered_candidates": directory / FILTERED_CANDIDATES_FILENAME,
        "report": directory / FILTERING_REPORT_FILENAME,
    }
    outputs["original_overlay"].write_bytes(png_bytes(result.original_overlay))
    outputs["retained_overlay"].write_bytes(png_bytes(result.retained_overlay))
    outputs["rejected_overlay"].write_bytes(png_bytes(result.rejected_overlay))
    outputs["comparison"].write_bytes(png_bytes(result.comparison))
    outputs["filtered_candidates"].write_text(
        json.dumps(result.document, indent=2) + "\n", encoding="utf-8"
    )
    outputs["report"].write_text(
        json.dumps(result.report, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


# --- input verification and loading -----------------------------------


def _verify_inputs(
    width: int,
    height: int,
    candidates_document: dict[str, Any],
    text_regions_document: dict[str, Any],
) -> None:
    """Both documents must already share the original pixel grid."""
    for name, document in (
        ("wall_candidates.json", candidates_document),
        ("text_regions.json", text_regions_document),
    ):
        image = document.get("image", {})
        if image.get("coordinate_space") != "original":
            raise ValueError(f"{name}: coordinate_space must be 'original'")
        if (image.get("width"), image.get("height")) != (width, height):
            raise ValueError(
                f"{name}: image size {image.get('width')}x{image.get('height')} "
                f"does not match the original {width}x{height}"
            )


def _verify_text_mask(
    text_mask: Image.Image | None, text_regions_document: dict[str, Any]
) -> bool | None:
    """Cross-check the C2 mask against the region boxes, when supplied."""
    if text_mask is None:
        return None
    mask = np.asarray(open_image(text_mask).convert("L"))
    image = text_regions_document["image"]
    if (mask.shape[1], mask.shape[0]) != (image["width"], image["height"]):
        raise ValueError("text_regions_mask.png size does not match text_regions.json")
    for region in text_regions_document["regions"]:
        bbox = region["bbox"]
        cx = bbox["x"] + bbox["width"] // 2
        cy = bbox["y"] + bbox["height"] // 2
        if mask[cy, cx] == 0:
            raise ValueError(f"text mask has no ink at the centre of {region['id']}")
    return True


def _load_candidates(document: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for record in document["candidates"]:
        geometry = record["geometry"]
        p0 = (geometry["start"]["x"], geometry["start"]["y"])
        p1 = (geometry["end"]["x"], geometry["end"]["y"])
        pairing = record["edge_pairing"]
        candidates.append(
            {
                "id": record["id"],
                "p0": p0,
                "p1": p1,
                "mid": ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0),
                "length": record["length"],
                "angle_deg": record["angle_deg"],
                "orientation": record["orientation"],
                "thickness": record["estimated_thickness"],
                "continuity": record["continuity"],
                "straightness": record["straightness"],
                "thickness_consistency": record["thickness_consistency"],
                "paired": bool(pairing["paired"]),
                "pairing_score": pairing["score"],
                "pairing_overlap_fraction": pairing.get("overlap_fraction"),
                "pairing_edge_distance_px": pairing.get("distance_px"),
            }
        )
    return candidates


def _load_text_boxes(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": region["id"],
            "bbox": (
                region["bbox"]["x"],
                region["bbox"]["y"],
                region["bbox"]["width"],
                region["bbox"]["height"],
            ),
        }
        for region in document["regions"]
    ]


# --- derived features --------------------------------------------------


def _point_segment_distance(point, start, end) -> float:
    px, py = point
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length_sq))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _segment_segment_distance(a0, a1, b0, b1) -> float:
    return min(
        _point_segment_distance(a0, b0, b1),
        _point_segment_distance(a1, b0, b1),
        _point_segment_distance(b0, a0, a1),
        _point_segment_distance(b1, a0, a1),
    )


def _angle_delta(a_deg: float, b_deg: float) -> float:
    delta = abs(a_deg - b_deg) % 180.0
    return min(delta, 180.0 - delta)


def _collinear_overlap(a: dict[str, Any], b: dict[str, Any], config: FilterConfig) -> bool:
    """Two candidates describe overlapping runs of one collinear stroke."""
    if _angle_delta(a["angle_deg"], b["angle_deg"]) > COLLINEAR_ANGLE_TOL_DEG:
        return False
    lateral = _point_segment_distance(b["mid"], a["p0"], a["p1"])
    if lateral > max(COLLINEAR_LATERAL_TOL_PX, 0.5 * min(a["thickness"], b["thickness"])):
        return False
    ux, uy = math.cos(math.radians(a["angle_deg"])), math.sin(math.radians(a["angle_deg"]))
    def project(point):
        return (point[0] - a["p0"][0]) * ux + (point[1] - a["p0"][1]) * uy
    a_ext = sorted((project(a["p0"]), project(a["p1"])))
    b_ext = sorted((project(b["p0"]), project(b["p1"])))
    gap = max(a_ext[0], b_ext[0]) - min(a_ext[1], b_ext[1])
    return gap <= COLLINEAR_GAP_PX


def _text_coverage(candidate: dict[str, Any], text_boxes: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """Fraction of the centreline inside any proposed text-region box."""
    if not text_boxes:
        return 0.0, []
    hits = 0
    overlapping: list[str] = []
    for step in range(TEXT_COVERAGE_SAMPLES):
        t = step / (TEXT_COVERAGE_SAMPLES - 1)
        x = candidate["p0"][0] + (candidate["p1"][0] - candidate["p0"][0]) * t
        y = candidate["p0"][1] + (candidate["p1"][1] - candidate["p0"][1]) * t
        for box in text_boxes:
            bx, by, bw, bh = box["bbox"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                hits += 1
                if box["id"] not in overlapping:
                    overlapping.append(box["id"])
                break
    return hits / TEXT_COVERAGE_SAMPLES, overlapping


def _structure_connections(
    candidate: dict[str, Any], all_candidates: list[dict[str, Any]], config: FilterConfig
) -> int:
    """Nearby substantial candidates joined at this candidate's ends."""
    count = 0
    for other in all_candidates:
        if other["id"] == candidate["id"]:
            continue
        if other["length"] < config.connect_min_other_length_px:
            continue
        distances = (
            _point_segment_distance(candidate["p0"], other["p0"], other["p1"]),
            _point_segment_distance(candidate["p1"], other["p0"], other["p1"]),
            _point_segment_distance(other["p0"], candidate["p0"], candidate["p1"]),
            _point_segment_distance(other["p1"], candidate["p0"], candidate["p1"]),
        )
        if min(distances) <= config.connect_max_dist_px:
            count += 1
    return count


def _short_neighbour_count(
    candidate: dict[str, Any], all_candidates: list[dict[str, Any]], config: FilterConfig
) -> int:
    """Short candidates in the neighbourhood (local density of short strokes)."""
    count = 0
    for other in all_candidates:
        if other["id"] == candidate["id"]:
            continue
        if other["length"] >= config.short_neighbour_max_length_px:
            continue
        if math.dist(candidate["mid"], other["mid"]) <= config.short_neighbour_radius_px:
            count += 1
    return count


# --- decision ----------------------------------------------------------


def _evaluate(
    candidate: dict[str, Any],
    all_candidates: list[dict[str, Any]],
    text_boxes: list[dict[str, Any]],
    config: FilterConfig,
) -> FilterDecision:
    coverage, text_ids = _text_coverage(candidate, text_boxes)
    connections = _structure_connections(candidate, all_candidates, config)
    short_neighbours = _short_neighbour_count(candidate, all_candidates, config)
    collinear = sum(
        1
        for other in all_candidates
        if other["id"] != candidate["id"] and _collinear_overlap(candidate, other, config)
    )

    features = {
        "length_px": round(candidate["length"], 2),
        "estimated_thickness_px": round(candidate["thickness"], 2),
        "slenderness": round(candidate["length"] / max(0.5, candidate["thickness"]), 2),
        "angle_deg": round(candidate["angle_deg"], 3),
        "orientation": candidate["orientation"],
        "paired_edges": candidate["paired"],
        "pairing_score": candidate["pairing_score"],
        "pairing_overlap_fraction": candidate["pairing_overlap_fraction"],
        "pairing_edge_distance_px": candidate["pairing_edge_distance_px"],
        "continuity": candidate["continuity"],
        "straightness": candidate["straightness"],
        "thickness_consistency": candidate["thickness_consistency"],
        "text_region_coverage": round(coverage, 4),
        "structure_connection_count": connections,
        "collinear_overlap_count": collinear,
        "short_neighbour_count": short_neighbours,
    }

    retain_evidence: list[str] = []
    if candidate["paired"] and candidate["pairing_score"] >= config.paired_score_min:
        retain_evidence.append(
            f"{K1_PAIRED_EDGES}: two parallel edges bound this stroke "
            f"(score {candidate['pairing_score']}, overlap {candidate['pairing_overlap_fraction']})"
        )
    if (
        candidate["thickness"] >= config.sturdy_thickness_px
        and candidate["continuity"] >= config.strong_continuity
        and candidate["straightness"] >= config.strong_straightness
    ):
        retain_evidence.append(
            f"{K2_STURDY_STROKE}: thick enough to be a drawn stroke "
            f"({candidate['thickness']:.1f} px) with continuous, straight ink "
            f"({candidate['continuity']:.2f}/{candidate['straightness']:.2f})"
        )
    if connections >= config.connected_structure_min and candidate["continuity"] >= config.moderate_continuity:
        retain_evidence.append(
            f"{K3_CONNECTED_STRUCTURE}: joins {connections} substantial nearby candidates "
            f"(corner/opening context)"
        )
    if collinear >= 1 and candidate["continuity"] >= config.strong_continuity:
        retain_evidence.append(
            f"{K4_COLLINEAR_CORROBORATION}: {collinear} overlapping collinear candidate(s) "
            "describe the same stroke run"
        )
    if (
        candidate["length"] >= config.substantial_length_px
        and candidate["continuity"] >= config.moderate_continuity
        and candidate["straightness"] >= config.moderate_straightness
    ):
        retain_evidence.append(
            f"{K5_SUBSTANTIAL_LENGTH}: long continuous straight run "
            f"({candidate['length']:.0f} px, continuity {candidate['continuity']:.2f})"
        )

    reject_evidence: list[str] = []
    fired_rules: list[str] = []
    paired_protected = candidate["paired"] and candidate["pairing_score"] >= config.paired_score_min

    r1_ok = (
        coverage >= config.r1_text_coverage_min
        and candidate["length"] <= config.r1_max_length_px
        and candidate["thickness"] <= config.r1_max_thickness_px
        and connections == 0
    )
    if r1_ok:
        fired_rules.append(R1_TEXT_CLUTTER)
        reject_evidence.append(
            f"{R1_TEXT_CLUTTER}: {coverage:.0%} of the centreline lies inside proposed "
            f"text region(s) {', '.join(text_ids)} and the mark is short "
            f"({candidate['length']:.0f} px), thin ({candidate['thickness']:.1f} px) "
            "and connects to no substantial neighbour"
        )

    r2_ok = (
        candidate["length"] < config.r2_max_length_px
        and candidate["thickness"] <= config.r2_max_thickness_px
        and not candidate["paired"]
        and connections == 0
        and short_neighbours >= config.r2_min_short_neighbours
    )
    if r2_ok:
        fired_rules.append(R2_ISOLATED_TEXTURE_MARK)
        reject_evidence.append(
            f"{R2_ISOLATED_TEXTURE_MARK}: isolated short thin single-edge mark "
            f"({candidate['length']:.0f} px x {candidate['thickness']:.1f} px) inside a "
            f"dense cluster of {short_neighbours} similar short marks"
        )

    decision_reasons: list[str] = []
    if fired_rules and not paired_protected:
        status = STATUS_REJECTED
        decision_reasons.append("rejected: " + ", ".join(fired_rules))
        if retain_evidence:
            decision_reasons.append(
                "retention evidence present but outweighed by stacked rejection evidence"
            )
    elif fired_rules and paired_protected:
        status = STATUS_RETAINED if len(retain_evidence) >= config.min_retain_evidences else STATUS_UNCERTAIN
        decision_reasons.append(
            "paired-edge evidence (score >= "
            f"{config.paired_score_min}) overrides clutter rejection; kept as {status}"
        )
    elif len(retain_evidence) >= config.min_retain_evidences:
        status = STATUS_RETAINED
        decision_reasons.append(
            f"retained: {len(retain_evidence)} independent retention evidences "
            "(no rejection rule fired)"
        )
    else:
        status = STATUS_UNCERTAIN
        decision_reasons.append(
            "uncertain: insufficient evidence either way; preserved in the filtered set"
        )

    return FilterDecision(
        id=candidate["id"],
        status=status,
        start=candidate["p0"],
        end=candidate["p1"],
        features=features,
        retain_evidence=retain_evidence,
        reject_evidence=reject_evidence,
        decision_reasons=decision_reasons,
        fired_rules=fired_rules,
        text_region_ids=text_ids,
    )


# --- documents ---------------------------------------------------------


def _decision_record(decision: FilterDecision) -> dict[str, Any]:
    return {
        "id": decision.id,
        "status": decision.status,
        "features": decision.features,
        "retain_evidence": decision.retain_evidence,
        "reject_evidence": decision.reject_evidence,
        "decision_reasons": decision.decision_reasons,
        "fired_rules": decision.fired_rules,
        "text_region_ids": decision.text_region_ids,
    }


def _build_document(
    decisions: list[FilterDecision],
    candidates_document: dict[str, Any],
    text_regions_document: dict[str, Any],
    width: int,
    height: int,
    config: FilterConfig,
    mask_consistent: bool | None,
) -> dict[str, Any]:
    counts = _status_counts(decisions)
    return {
        "phase": {"number": 3, "name": "wall_candidate_filtering", "step": "c3"},
        "status": "experimental conservative filter: statuses and scores are uncalibrated "
                  "heuristics, not calibrated probabilities, and no candidate is classified "
                  "as a confirmed wall",
        "image": {
            "width": width,
            "height": height,
            "coordinate_space": "original",
        },
        "inputs": {
            "candidates": "artifacts/parse2d/phase_3/wall_candidates.json",
            "text_regions": "artifacts/parse2d/phase_3_refined/c2_text_regions/text_regions.json",
            "text_mask": "artifacts/parse2d/phase_3_refined/c2_text_regions/text_regions_mask.png",
            "source_image": "examples/plans/real-time.png",
            "candidates_total": len(decisions),
            "text_regions_total": len(text_regions_document["regions"]),
            "text_mask_cross_checked": mask_consistent,
            "baseline_preserved": "Step C and C1/C2 artifacts are read-only inputs; "
                                  "this file is a separate experimental output",
        },
        "configuration": {
            "rejection_rules": {
                R1_TEXT_CLUTTER: {
                    "text_region_coverage_min": config.r1_text_coverage_min,
                    "max_length_px": config.r1_max_length_px,
                    "max_thickness_px": config.r1_max_thickness_px,
                    "requires_isolation": True,
                },
                R2_ISOLATED_TEXTURE_MARK: {
                    "max_length_px": config.r2_max_length_px,
                    "max_thickness_px": config.r2_max_thickness_px,
                    "min_short_neighbours": config.r2_min_short_neighbours,
                    "requires_single_edge": True,
                    "requires_isolation": True,
                },
            },
            "retention_evidence": {
                K1_PAIRED_EDGES: f"paired edges with score >= {config.paired_score_min}",
                K2_STURDY_STROKE: f"thickness >= {config.sturdy_thickness_px} px, "
                                  f"continuity >= {config.strong_continuity}, "
                                  f"straightness >= {config.strong_straightness}",
                K3_CONNECTED_STRUCTURE: f">= {config.connected_structure_min} substantial "
                                        f"neighbours within {config.connect_max_dist_px} px",
                K4_COLLINEAR_CORROBORATION: "overlapping collinear candidate(s) with "
                                            f"continuity >= {config.strong_continuity}",
                K5_SUBSTANTIAL_LENGTH: f"length >= {config.substantial_length_px} px, "
                                       f"continuity >= {config.moderate_continuity}, "
                                       f"straightness >= {config.moderate_straightness}",
                "retained_threshold": f">= {config.min_retain_evidences} evidences",
            },
            "protection": "candidates with paired edges (score >= "
                          f"{config.paired_score_min}) are never rejected by clutter rules",
            "geometry": {
                "connect_max_dist_px": config.connect_max_dist_px,
                "connect_min_other_length_px": config.connect_min_other_length_px,
                "short_neighbour_radius_px": config.short_neighbour_radius_px,
                "short_neighbour_max_length_px": config.short_neighbour_max_length_px,
            },
        },
        "filtered_set_definition": "retained + uncertain; only rejected candidates are "
                                   "excluded from the filtered set, and rejected candidates "
                                   "remain fully recorded in this file",
        "summary": {
            "original_candidates": len(decisions),
            "retained_candidates": counts[STATUS_RETAINED],
            "rejected_candidates": counts[STATUS_REJECTED],
            "uncertain_candidates": counts[STATUS_UNCERTAIN],
            "filtered_set_count": counts[STATUS_RETAINED] + counts[STATUS_UNCERTAIN],
            "by_orientation": _by_orientation(decisions),
            "by_pairing": _by_pairing(decisions),
        },
        "candidates": [_decision_record(decision) for decision in decisions],
    }


def _status_counts(decisions: list[FilterDecision]) -> dict[str, int]:
    counts = {STATUS_RETAINED: 0, STATUS_REJECTED: 0, STATUS_UNCERTAIN: 0}
    for decision in decisions:
        counts[decision.status] += 1
    return counts


def _by_orientation(decisions: list[FilterDecision]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for decision in decisions:
        orientation = decision.features["orientation"]
        table.setdefault(decision.status, {})
        table[decision.status][orientation] = table[decision.status].get(orientation, 0) + 1
    return table


def _by_pairing(decisions: list[FilterDecision]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for decision in decisions:
        key = "paired" if decision.features["paired_edges"] else "single_edge"
        table.setdefault(decision.status, {})
        table[decision.status][key] = table[decision.status].get(key, 0) + 1
    return table


def _build_report(
    decisions: list[FilterDecision],
    candidates_document: dict[str, Any],
    text_regions_document: dict[str, Any],
    config: FilterConfig,
    mask_consistent: bool | None,
) -> dict[str, Any]:
    counts = _status_counts(decisions)
    rule_counts: dict[str, int] = {}
    for decision in decisions:
        for rule in decision.fired_rules:
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
    rule_contributions = dict(sorted(rule_counts.items(), key=lambda kv: -kv[1]))
    retained_with_text = [
        d for d in decisions
        if d.status != STATUS_REJECTED and d.features["text_region_coverage"] >= 0.5
    ]
    rejected_without_text = [
        d for d in decisions
        if d.status == STATUS_REJECTED and R1_TEXT_CLUTTER not in d.fired_rules
    ]
    suspicious_keeps = [
        d for d in decisions
        if d.status == STATUS_RETAINED
        and d.features["length_px"] <= config.r1_max_length_px
        and d.features["estimated_thickness_px"] <= config.r1_max_thickness_px
    ]
    paired_overrides = [
        d for d in decisions
        if d.fired_rules and d.features["paired_edges"]
    ]

    return {
        "phase": {"number": 3, "name": "wall_candidate_filtering", "step": "c3"},
        "steps": {
            "c3": "experimental conservative filtering of the Step C wall candidates "
                  "against C2 proposed text regions and geometric evidence; every "
                  "decision is auditable and no candidate is classified as a wall",
        },
        "summary": {
            "original_candidates": len(decisions),
            "retained_candidates": counts[STATUS_RETAINED],
            "rejected_candidates": counts[STATUS_REJECTED],
            "uncertain_candidates": counts[STATUS_UNCERTAIN],
            "filtered_set_count": counts[STATUS_RETAINED] + counts[STATUS_UNCERTAIN],
            "by_orientation": _by_orientation(decisions),
            "by_pairing": _by_pairing(decisions),
            "rule_contributions": rule_contributions,
            "retention_evidence_counts": _evidence_counts(decisions),
            "candidates_overlapping_text_regions": sum(
                1 for d in decisions if d.features["text_region_coverage"] > 0
            ),
            "text_mask_cross_checked": mask_consistent,
        },
        "findings": [
            f"{len(decisions)} original candidates evaluated: "
            f"{counts[STATUS_RETAINED]} retained, {counts[STATUS_UNCERTAIN]} uncertain "
            f"(both kept), {counts[STATUS_REJECTED]} rejected.",
            "Most rejections came from: "
            + (
                "; ".join(f"{rule} ({count})" for rule, count in rule_contributions.items())
                if rule_contributions
                else "no rule fired"
            )
            + ".",
            f"{counts[STATUS_RETAINED] + counts[STATUS_UNCERTAIN]} candidates form the "
            "filtered set; uncertain candidates are preserved by design.",
            "C2 text overlap was treated as weak negative evidence only: rejection "
            "required it to be stacked with short length, thin strokes and isolation.",
        ],
        "known_failure_cases": [
            {
                "case": "kept candidates that sit inside proposed text regions "
                        "(filter keeps them; some are text, some are walls behind labels)",
                "count": len(retained_with_text),
                "example_ids": [d.id for d in retained_with_text[:10]],
            },
            {
                "case": "rejected candidates without text evidence (cluster rule only); "
                        "legitimate short thin stubs in dense clutter could be among them",
                "count": len(rejected_without_text),
                "example_ids": [d.id for d in rejected_without_text[:10]],
            },
            {
                "case": "retained candidates that are both short and thin (suspicious keeps)",
                "count": len(suspicious_keeps),
                "example_ids": [d.id for d in suspicious_keeps[:10]],
            },
            {
                "case": "candidates where paired-edge protection overrode a clutter "
                        "rejection (glyph stems can be paired strokes)",
                "count": len(paired_overrides),
                "example_ids": [d.id for d in paired_overrides[:10]],
            },
        ],
        "constraints_respected": [
            "Step C candidates and C1/C2 artifacts are unmodified read-only inputs.",
            "No candidate is classified as a confirmed wall.",
            "No candidate is rejected solely for overlapping a text region, solely for "
            "being short, or solely for being diagonal.",
            "Plausible wall segments near corners and openings are protected by "
            "connectivity to substantial neighbours.",
            "All uncertain candidates are preserved; rejected candidates remain "
            "recoverable in filtered_candidates.json and the rejected overlay.",
            "No precision, recall or accuracy claim is made without labelled ground truth.",
            "No production wall extraction, room detection, opening detection, /parse2d, "
            "UI or API code was modified.",
        ],
        "limitations": [
            "Statuses and evidence scores are uncalibrated heuristics, not "
            "calibrated probabilities.",
            "C2 text regions are coarse bounding-box proposals; their overlap with a "
            "candidate says little about what the candidate is.",
            "Without labelled ground truth the effect of filtering on correctness is "
            "unknown; the overlays exist so the rules can be judged visually before "
            "any rule is accepted.",
            "Connectivity protection assumes walls join within "
            f"{config.connect_max_dist_px} px; isolated short wall fragments without "
            "substantial neighbours can still be rejected by the cluster rule.",
            "Diagonal candidates are evaluated on the same evidence as axis-aligned "
            "ones; rotation is never a rejection reason.",
        ],
    }


def _evidence_counts(decisions: list[FilterDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        for entry in decision.retain_evidence:
            name = entry.split(":", 1)[0]
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# --- overlays ----------------------------------------------------------


def _draw_overlay(
    base_rgb: np.ndarray,
    decisions: list[FilterDecision],
    text_boxes: list[dict[str, Any]],
    statuses: set[str],
    status_colors: dict[str, tuple[int, int, int]],
    label_ids: bool,
    show_text_boxes: bool,
    title: str,
) -> Image.Image:
    canvas = base_rgb.copy()
    if show_text_boxes:
        for box in text_boxes:
            x, y, w, h = box["bbox"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_TEXT_BOX, THIN_LINE_THICKNESS)
    for decision in decisions:
        if decision.status not in statuses:
            continue
        color = status_colors[decision.status]
        cv2.line(canvas, _pt(decision.start), _pt(decision.end), color, LINE_THICKNESS, cv2.LINE_AA)
    if label_ids:
        placed: list[tuple[int, int, int, int]] = []
        for decision in decisions:
            if decision.status not in statuses:
                continue
            mid = (
                (decision.start[0] + decision.end[0]) / 2.0,
                (decision.start[1] + decision.end[1]) / 2.0,
            )
            _draw_label(canvas, decision.id, mid, status_colors[decision.status], placed)
    _draw_banner(canvas, title, decisions)
    return Image.fromarray(canvas)


def _pt(point) -> tuple[int, int]:
    return (round(point[0]), round(point[1]))


def _draw_label(
    canvas: np.ndarray, text: str, origin, color, placed: list[tuple[int, int, int, int]]
) -> None:
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, FONT_THICKNESS
    )
    box = None
    for dx, dy in ((0, 0), (0, -12), (0, 12), (12, 0), (-12, 0)):
        x = int(clamp(origin[0] + dx + LABEL_GAP_PX, 1, canvas.shape[1] - text_w - 2))
        y = int(clamp(origin[1] + dy - LABEL_GAP_PX, text_h + 2, canvas.shape[0] - baseline - 2))
        rect = (x - 1, y - text_h - 1, x + text_w + 1, y + baseline)
        if not any(
            not (rect[2] < other[0] or rect[0] > other[2] or rect[3] < other[1] or rect[1] > other[3])
            for other in placed
        ):
            box = rect
            break
    if box is None:
        return
    placed.append(box)
    cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), COLOR_BANNER_BG, -1)
    cv2.putText(
        canvas, text, (box[0] + 1, box[3] - baseline), cv2.FONT_HERSHEY_SIMPLEX,
        0.4, color, FONT_THICKNESS, cv2.LINE_AA,
    )


def clamp(value, lo, hi):
    return max(lo, min(value, hi))


def _draw_banner(canvas: np.ndarray, title: str, decisions: list[FilterDecision]) -> None:
    counts = _status_counts(decisions)
    rows = [
        title,
        f"original {len(decisions)}  -  retained {counts[STATUS_RETAINED]}  -  "
        f"uncertain {counts[STATUS_UNCERTAIN]}  -  rejected {counts[STATUS_REJECTED]}",
        "green: retained  -  amber: uncertain (kept)  -  red: rejected  -  "
        "magenta boxes: C2 proposed text regions",
        "candidates are geometric proposals; none is a confirmed wall; scores are uncalibrated",
    ]
    pad = LABEL_GAP_PX
    widths = [cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)[0][0] for row in rows]
    line_height = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)[0][1] + pad
    box_w = min(canvas.shape[1], max(widths) + 2 * pad)
    box_h = line_height * len(rows) + pad
    cv2.rectangle(canvas, (0, 0), (box_w, box_h), COLOR_BANNER_BG, -1)
    for index, row in enumerate(rows):
        cv2.putText(
            canvas, row, (pad, line_height * (index + 1)), cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, COLOR_TEXT, 1, cv2.LINE_AA,
        )


def _draw_comparison(
    base_rgb: np.ndarray, decisions: list[FilterDecision], text_boxes: list[dict[str, Any]]
) -> Image.Image:
    """Before/after panels side by side on the original image."""
    counts = _status_counts(decisions)
    before = _draw_overlay(
        base_rgb, decisions, text_boxes,
        {STATUS_RETAINED, STATUS_REJECTED, STATUS_UNCERTAIN},
        {s: COLOR_BASELINE for s in (STATUS_RETAINED, STATUS_REJECTED, STATUS_UNCERTAIN)},
        label_ids=False, show_text_boxes=False,
        title=f"BEFORE - {len(decisions)} original candidates (unfiltered baseline)",
    )
    after = _draw_overlay(
        base_rgb, decisions, text_boxes,
        {STATUS_RETAINED, STATUS_UNCERTAIN},
        {STATUS_RETAINED: COLOR_RETAINED, STATUS_UNCERTAIN: COLOR_UNCERTAIN},
        label_ids=False, show_text_boxes=True,
        title=(
            f"AFTER - filtered set: {counts[STATUS_RETAINED]} retained + "
            f"{counts[STATUS_UNCERTAIN]} uncertain (kept), "
            f"{counts[STATUS_REJECTED]} rejected"
        ),
    )
    before_bgr = cv2.cvtColor(np.asarray(before), cv2.COLOR_RGB2BGR)
    after_bgr = cv2.cvtColor(np.asarray(after), cv2.COLOR_RGB2BGR)
    gap = np.full((before_bgr.shape[0], 8, 3), 255, dtype=np.uint8)
    return Image.fromarray(cv2.cvtColor(np.hstack([before_bgr, gap, after_bgr]), cv2.COLOR_BGR2RGB))


# --- standalone runner -------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Generate the Step C3 artifacts from existing Step C/C2 outputs.

    Run from the project root:

        python -m floorplanto3d.processing.candidate_filter \
            [IMAGE] [CANDIDATES_JSON] [TEXT_REGIONS_JSON] [TEXT_MASK] [OUT_DIR]
    """
    root = Path.cwd()
    parser = argparse.ArgumentParser(description="Phase 2, Step C3 conservative candidate filtering")
    parser.add_argument("image", type=Path, nargs="?", default=root / "examples/plans/real-time.png")
    parser.add_argument(
        "candidates_json", type=Path, nargs="?",
        default=root / "artifacts/parse2d/phase_3/wall_candidates.json",
    )
    parser.add_argument(
        "text_regions_json", type=Path, nargs="?",
        default=root / "artifacts/parse2d/phase_3_refined/c2_text_regions/text_regions.json",
    )
    parser.add_argument(
        "text_mask", type=Path, nargs="?",
        default=root / "artifacts/parse2d/phase_3_refined/c2_text_regions/text_regions_mask.png",
    )
    parser.add_argument(
        "out_dir", type=Path, nargs="?",
        default=root / "artifacts/parse2d/phase_3_refined/c3_filtering",
    )
    args = parser.parse_args(argv)

    candidates_document = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    text_regions_document = json.loads(args.text_regions_json.read_text(encoding="utf-8"))
    text_mask = open_image(args.text_mask) if args.text_mask.exists() else None

    result = filter_candidates(
        args.image, candidates_document, text_regions_document, text_mask=text_mask
    )
    outputs = save_artifacts(result, args.out_dir)
    for name, path in outputs.items():
        print(f"wrote {path}  ({name})")
    summary = result.report["summary"]
    print(
        f"{summary['original_candidates']} original -> "
        f"{summary['retained_candidates']} retained, "
        f"{summary['uncertain_candidates']} uncertain, "
        f"{summary['rejected_candidates']} rejected"
    )


if __name__ == "__main__":
    main()
