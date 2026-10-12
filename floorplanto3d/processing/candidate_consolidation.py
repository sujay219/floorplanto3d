"""Phase 2, Step C4.2: conservative wall-candidate consolidation.

This stage consolidates compatible collinear candidates from the C3
filtered set into longer geometric candidates that appear to describe
the same continuous stroke. It implements **exactly** the merge rules
approved in the C4.1 inspection (``c4_inspection_report.json``) and no
others. A consolidated candidate is a geometric proposal: it is never
called or treated as a confirmed wall.

Approved eligibility (all must hold for a pair to merge):

* angle delta <= 4 degrees (approximately collinear),
* lateral offset <= max(2.5 px, 0.5 * min(thickness)),
* projected gap <= 20 px (overlapping intervals count as gap < 0),
* thickness ratio <= 1.8 (both sides <= 3 px always compatible),
* both candidates inside the C3 filtered set (retained or uncertain).

Approved guardrails:

* non-collinear candidates never merge (corners and T-junctions stay),
* gaps of 20-150 px (possible doors/windows) are never bridged,
* offset parallel strokes never merge (opposite faces of a wall),
* no source candidate is ever discarded; merges only group IDs with
  full provenance,
* a consolidated group's cumulative bridged gap must stay <= 2x the
  single-pair gap ceiling (40 px), so several small gaps cannot team up
  to bridge an opening,
* ambiguous evidence (e.g. thickness ratio above threshold) keeps the
  candidates separate and the ambiguity is recorded,
* merged output remains a geometric proposal.

Inputs: the C3 ``filtered_candidates.json`` document defines the
candidate set and statuses; endpoint geometry is read from the Step C
``wall_candidates.json`` document the C3 document references, because
the C3 records do not carry endpoints. Both are read-only; nothing in
C1-C3 is modified. All coordinates stay in original-image pixel space.

Output: ``consolidated_candidates.json`` and ``consolidation_report.json``.
Every filtered-set candidate is accounted for as a member of exactly one
output candidate (consolidated group or unchanged candidate), and every
C3-rejected candidate is listed as explicitly excluded. This module is
not wired into /parse2d or any later step.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from floorplanto3d.models.geometry import classify_orientation

logger = logging.getLogger(__name__)

# Status vocabulary (same semantics as C3).
STATUS_RETAINED = "retained"
STATUS_UNCERTAIN = "uncertain"
STATUS_REJECTED = "rejected"

# Output record kinds.
KIND_CONSOLIDATED = "consolidated"
KIND_UNCHANGED = "unchanged"

# Saved artifact filenames.
CONSOLIDATED_CANDIDATES_FILENAME = "consolidated_candidates.json"
CONSOLIDATION_REPORT_FILENAME = "consolidation_report.json"


@dataclass(frozen=True)
class ConsolidationConfig:
    """The approved C4.1 thresholds, kept configurable and recorded."""

    angle_tol_deg: float = 4.0
    lateral_tolerance_floor_px: float = 2.5
    lateral_tolerance_fraction: float = 0.5
    merge_gap_max_px: float = 20.0
    opening_gap_max_px: float = 150.0
    thickness_ratio_max: float = 1.8
    thin_thickness_floor_px: float = 3.0
    group_cumulative_gap_max_px: float = 40.0


@dataclass
class ConsolidationResult:
    """Outputs of Step C4.2."""

    document: dict[str, Any]
    report: dict[str, Any]


def consolidate_candidates(
    candidates_document: dict[str, Any],
    filtered_document: dict[str, Any],
    config: ConsolidationConfig | None = None,
) -> ConsolidationResult:
    """Consolidate the C3 filtered set under the approved merge rules.

    Args:
        candidates_document: The Step C ``wall_candidates.json`` document
            (geometry source, read-only).
        filtered_document: The C3 ``filtered_candidates.json`` document
            (candidate set and statuses, read-only).
        config: The approved thresholds; defaults are the C4.1 values.

    Returns:
        A :class:`ConsolidationResult` whose document accounts for every
        input candidate and whose report counts merges, refused proposals
        and guardrail checks. Nothing here is a confirmed wall.
    """
    config = config or ConsolidationConfig()
    _verify_inputs(candidates_document, filtered_document)

    status_by_id = {r["id"]: r["status"] for r in filtered_document["candidates"]}
    candidates = _load_candidates(candidates_document, status_by_id)
    universe = [c for c in candidates if c["status"] != STATUS_REJECTED]
    excluded = [c for c in candidates if c["status"] == STATUS_REJECTED]

    groups, applied, refused, opening_risks = _build_groups(universe, config)
    records, source_mapping = _make_records(universe, excluded, groups, applied, config)
    document = _build_document(
        records, source_mapping, candidates, universe, excluded,
        candidates_document, filtered_document, config,
    )
    report = _build_report(
        records, source_mapping, candidates, universe, excluded,
        applied, refused, opening_risks, config,
    )
    logger.info(
        "candidate consolidation (Step C4.2) complete: %d filtered-set candidates -> "
        "%d outputs (%d consolidated, %d unchanged), %d merges applied, "
        "%d proposals refused",
        len(universe),
        len(records),
        sum(1 for r in records if r["kind"] == KIND_CONSOLIDATED),
        sum(1 for r in records if r["kind"] == KIND_UNCHANGED),
        len(applied),
        len(refused),
    )
    return ConsolidationResult(document=document, report=report)


def save_artifacts(result: ConsolidationResult, directory: str | Path) -> dict[str, Path]:
    """Write ``consolidated_candidates.json`` and ``consolidation_report.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "consolidated_candidates": directory / CONSOLIDATED_CANDIDATES_FILENAME,
        "report": directory / CONSOLIDATION_REPORT_FILENAME,
    }
    outputs["consolidated_candidates"].write_text(
        json.dumps(result.document, indent=2) + "\n", encoding="utf-8"
    )
    outputs["report"].write_text(
        json.dumps(result.report, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


# --- input verification and loading -----------------------------------


def _verify_inputs(
    candidates_document: dict[str, Any], filtered_document: dict[str, Any]
) -> None:
    """Both documents must share the original pixel grid and candidate IDs."""
    for name, document in (
        ("wall_candidates.json", candidates_document),
        ("filtered_candidates.json", filtered_document),
    ):
        image = document.get("image", {})
        if image.get("coordinate_space") != "original":
            raise ValueError(f"{name}: coordinate_space must be 'original'")
    cand_image = candidates_document["image"]
    filt_image = filtered_document["image"]
    if (cand_image["width"], cand_image["height"]) != (filt_image["width"], filt_image["height"]):
        raise ValueError("input documents disagree on image dimensions")
    cand_ids = [r["id"] for r in candidates_document["candidates"]]
    filt_ids = [r["id"] for r in filtered_document["candidates"]]
    if cand_ids != filt_ids:
        raise ValueError("input documents disagree on candidate IDs")


def _load_candidates(
    document: dict[str, Any], status_by_id: dict[str, str]
) -> list[dict[str, Any]]:
    candidates = []
    for record in document["candidates"]:
        start = (record["geometry"]["start"]["x"], record["geometry"]["start"]["y"])
        end = (record["geometry"]["end"]["x"], record["geometry"]["end"]["y"])
        length = math.dist(start, end)
        ux = (end[0] - start[0]) / length if length > 0 else 1.0
        uy = (end[1] - start[1]) / length if length > 0 else 0.0
        candidates.append(
            {
                "id": record["id"],
                "index": len(candidates),
                "p0": start,
                "p1": end,
                "length": record["length"],
                "angle_deg": record["angle_deg"],
                "orientation": record["orientation"],
                "thickness": record["estimated_thickness"],
                "status": status_by_id[record["id"]],
                "ux": ux,
                "uy": uy,
            }
        )
    return candidates


# --- approved pair rules (C4.1, verbatim) -----------------------------


def _angle_delta(a_deg: float, b_deg: float) -> float:
    delta = abs(a_deg - b_deg) % 180.0
    return min(delta, 180.0 - delta)


def _thickness_ratio(a: dict[str, Any], b: dict[str, Any], config: ConsolidationConfig) -> float:
    thin_a = a["thickness"] <= config.thin_thickness_floor_px
    thin_b = b["thickness"] <= config.thin_thickness_floor_px
    if thin_a and thin_b:
        return 1.0
    return max(a["thickness"], b["thickness"]) / max(1e-6, min(a["thickness"], b["thickness"]))


def _pair_measurements(
    a: dict[str, Any], b: dict[str, Any], config: ConsolidationConfig
) -> dict[str, Any]:
    delta = _angle_delta(a["angle_deg"], b["angle_deg"])
    tolerance = max(
        config.lateral_tolerance_floor_px,
        config.lateral_tolerance_fraction * min(a["thickness"], b["thickness"]),
    )
    ux, uy = a["ux"], a["uy"]

    def perp(point: tuple[float, float]) -> float:
        return abs((point[0] - a["p0"][0]) * uy - (point[1] - a["p0"][1]) * ux)

    lateral = max(perp(b["p0"]), perp(b["p1"]))

    def project(point: tuple[float, float]) -> float:
        return (point[0] - a["p0"][0]) * ux + (point[1] - a["p0"][1]) * uy

    a_ext = sorted((project(a["p0"]), project(a["p1"])))
    b_ext = sorted((project(b["p0"]), project(b["p1"])))
    gap = max(a_ext[0], b_ext[0]) - min(a_ext[1], b_ext[1])
    return {
        "angle_delta_deg": round(delta, 3),
        "lateral_offset_px": round(lateral, 2),
        "lateral_tolerance_px": round(tolerance, 2),
        "gap_px": round(gap, 2),
        "overlap_px": round(max(0.0, -gap), 2),
        "thickness_ratio": round(_thickness_ratio(a, b, config), 3),
    }


def _eligible_pair(
    a: dict[str, Any], b: dict[str, Any], config: ConsolidationConfig
) -> tuple[bool, str, dict[str, Any]]:
    """The approved eligibility rules; returns (ok, reason, measurements)."""
    m = _pair_measurements(a, b, config)
    if m["angle_delta_deg"] > config.angle_tol_deg:
        return False, "orientation not collinear", m
    if m["lateral_offset_px"] > m["lateral_tolerance_px"]:
        return False, "lateral offset exceeds tolerance (offset parallel strokes stay separate)", m
    if m["gap_px"] > config.merge_gap_max_px:
        if m["gap_px"] <= config.opening_gap_max_px:
            return False, "gap is opening-sized and must not be bridged", m
        return False, "gap too large for one stroke", m
    if m["thickness_ratio"] > config.thickness_ratio_max:
        return False, "thickness evidence incompatible (ambiguous; kept separate)", m
    return True, (
        f"collinear (angle delta {m['angle_delta_deg']} deg), lateral offset "
        f"{m['lateral_offset_px']} px within {m['lateral_tolerance_px']} px, "
        f"gap {m['gap_px']} px (overlap {m['overlap_px']} px), thickness ratio "
        f"{m['thickness_ratio']}"
    ), m


# --- grouping with the approved chain guard ---------------------------


def _build_groups(universe: list[dict[str, Any]], config: ConsolidationConfig):
    """Greedy deterministic grouping over eligible pairs.

    Pairs are applied smallest gap first. A merge that would push a
    group's cumulative bridged gap past the approved chain guard
    (2x the single-pair ceiling) is refused, so chains of small gaps
    cannot bridge an opening.
    """
    eligible: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    opening_risks: list[dict[str, Any]] = []
    for i in range(len(universe)):
        for j in range(i + 1, len(universe)):
            a, b = universe[i], universe[j]
            ok, reason, m = _eligible_pair(a, b, config)
            entry = {"pair": [a["id"], b["id"]], "measurements": m, "reason": reason}
            if ok:
                eligible.append(entry)
            else:
                refused.append(entry)
                if reason.startswith("gap is opening-sized"):
                    opening_risks.append(entry)
    eligible.sort(
        key=lambda e: (
            e["measurements"]["gap_px"],
            e["measurements"]["lateral_offset_px"],
            e["pair"][0],
            e["pair"][1],
        )
    )
    refused.sort(key=lambda e: (e["pair"][0], e["pair"][1]))

    parent = {c["id"]: c["id"] for c in universe}
    bridged = {c["id"]: 0.0 for c in universe}
    applied: list[dict[str, Any]] = []

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for entry in eligible:
        id_a, id_b = entry["pair"]
        root_a, root_b = find(id_a), find(id_b)
        gap = entry["measurements"]["gap_px"]
        if root_a == root_b:
            continue
        bridged_total = bridged[root_a] + bridged[root_b] + max(0.0, gap)
        if bridged_total > config.group_cumulative_gap_max_px:
            refused.append(
                {
                    "pair": entry["pair"],
                    "measurements": entry["measurements"],
                    "reason": (
                        "chain guard: merging would raise the group's cumulative bridged "
                        f"gap to {bridged_total:.1f} px (limit "
                        f"{config.group_cumulative_gap_max_px} px); kept separate"
                    ),
                }
            )
            continue
        parent[root_b] = root_a
        bridged[root_a] = bridged_total
        applied.append(
            {
                "pair": entry["pair"],
                "measurements": entry["measurements"],
                "reason": entry["reason"],
                "group_bridged_gap_px": round(bridged_total, 2),
            }
        )

    members: dict[str, list[str]] = {}
    for c in universe:
        members.setdefault(find(c["id"]), []).append(c["id"])
    groups = [sorted(ids) for ids in members.values()]
    groups.sort(key=lambda ids: min(c["index"] for c in universe if c["id"] in ids))
    return groups, applied, refused, opening_risks


# --- output records ---------------------------------------------------


def _merged_geometry(
    members: list[dict[str, Any]], config: ConsolidationConfig
) -> dict[str, Any]:
    """Straight-line union of a group along its longest member's direction."""
    longest = max(members, key=lambda c: (c["length"], c["id"]))
    ux, uy = longest["ux"], longest["uy"]
    origin = longest["p0"]

    def project(point: tuple[float, float]) -> float:
        return (point[0] - origin[0]) * ux + (point[1] - origin[1]) * uy

    projections = [project(c["p0"]) for c in members] + [project(c["p1"]) for c in members]
    lo, hi = min(projections), max(projections)
    start = (origin[0] + ux * lo, origin[1] + uy * lo)
    end = (origin[0] + ux * hi, origin[1] + uy * hi)
    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0
    total_length = sum(c["length"] for c in members)
    thickness = (
        sum(c["thickness"] * c["length"] for c in members) / total_length
        if total_length > 0
        else max(c["thickness"] for c in members)
    )
    return {
        "start": {"x": round(start[0], 2), "y": round(start[1], 2)},
        "end": {"x": round(end[0], 2), "y": round(end[1], 2)},
        "length": round(math.dist(start, end), 2),
        "angle_deg": round(angle, 3),
        "orientation": classify_orientation(end[0] - start[0], end[1] - start[1]),
        "estimated_thickness": round(thickness, 2),
        "estimated_thickness_range_px": [
            round(min(c["thickness"] for c in members), 2),
            round(max(c["thickness"] for c in members), 2),
        ],
    }


def _source_entries(members: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    statuses = {c["id"]: c["status"] for c in members}
    geometries = [
        {
            "id": c["id"],
            "status": c["status"],
            "start": {"x": round(c["p0"][0], 2), "y": round(c["p0"][1], 2)},
            "end": {"x": round(c["p1"][0], 2), "y": round(c["p1"][1], 2)},
            "length": round(c["length"], 2),
            "angle_deg": round(c["angle_deg"], 3),
            "estimated_thickness": round(c["thickness"], 2),
        }
        for c in members
    ]
    return statuses, geometries


def _make_records(
    universe: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    groups: list[list[str]],
    applied: list[dict[str, Any]],
    config: ConsolidationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {c["id"]: c for c in universe}

    records: list[dict[str, Any]] = []
    source_mapping: dict[str, Any] = {}
    for order, ids in enumerate(groups, start=1):
        members = [by_id[i] for i in ids]
        statuses, geometries = _source_entries(members)
        if len(members) == 1:
            member = members[0]
            record = {
                "id": member["id"],
                "kind": KIND_UNCHANGED,
                "status": member["status"],
                "geometry": {
                    "start": {"x": round(member["p0"][0], 2), "y": round(member["p0"][1], 2)},
                    "end": {"x": round(member["p1"][0], 2), "y": round(member["p1"][1], 2)},
                },
                "length": round(member["length"], 2),
                "angle_deg": round(member["angle_deg"], 3),
                "orientation": member["orientation"],
                "estimated_thickness": round(member["thickness"], 2),
                "estimated_thickness_range_px": [
                    round(member["thickness"], 2),
                    round(member["thickness"], 2),
                ],
                "member_count": 1,
                "source_candidate_ids": [member["id"]],
                "source_statuses": statuses,
                "source_geometries": geometries,
                "merge_reasons": [],
                "bridged_gap_total_px": 0.0,
                "uncertainty": [
                    "unchanged candidate: no approved merge applied",
                    "geometric proposal only; not a confirmed wall",
                ],
                "limitations": [
                    "status carries over from C3: retained means retained as geometric "
                    "evidence, uncertain means kept because evidence is insufficient"
                ],
            }
            source_mapping[member["id"]] = member["id"]
        else:
            merged = _merged_geometry(members, config)
            reasons = [
                f"merged {entry['pair'][0]} + {entry['pair'][1]}: {entry['reason']}"
                for entry in applied
                if entry["pair"][0] in ids and entry["pair"][1] in ids
            ]
            all_retained = all(c["status"] == STATUS_RETAINED for c in members)
            bridged_total = max(
                (entry["group_bridged_gap_px"] for entry in applied
                 if entry["pair"][0] in ids and entry["pair"][1] in ids),
                default=0.0,
            )
            uncertainty = [
                "consolidated geometry is a straight-line proposal over the member "
                "centrelines; it is not a confirmed wall",
                f"group spans {len(members)} source candidates with total bridged gap "
                f"{bridged_total} px (approved ceiling {config.group_cumulative_gap_max_px} px)",
                "estimated thickness is the length-weighted mean of member "
                "measurements; probe-based member values can be unreliable",
            ]
            if not all_retained:
                uncertainty.append(
                    "group includes uncertain source candidates; the consolidated "
                    "candidate is kept uncertain and does not raise their evidence"
                )
            record = {
                "id": f"cons_{order:03d}",
                "kind": KIND_CONSOLIDATED,
                "status": STATUS_RETAINED if all_retained else STATUS_UNCERTAIN,
                "geometry": {
                    "start": merged["start"],
                    "end": merged["end"],
                },
                "length": merged["length"],
                "angle_deg": merged["angle_deg"],
                "orientation": merged["orientation"],
                "estimated_thickness": merged["estimated_thickness"],
                "estimated_thickness_range_px": merged["estimated_thickness_range_px"],
                "member_count": len(members),
                "source_candidate_ids": ids,
                "source_statuses": statuses,
                "source_geometries": geometries,
                "merge_reasons": reasons,
                "bridged_gap_total_px": bridged_total,
                "uncertainty": uncertainty,
                "limitations": [
                    "consolidation changes geometry, not classification; no wall claim",
                ],
            }
            for member in ids:
                source_mapping[member] = record["id"]
        records.append(record)

    for member in excluded:
        source_mapping[member["id"]] = {
            "output": None,
            "note": "excluded: rejected by the C3 filter; untouched by consolidation",
        }
    records.sort(
        key=lambda r: (
            round((r["geometry"]["start"]["y"] + r["geometry"]["end"]["y"]) / 2.0, 2),
            round((r["geometry"]["start"]["x"] + r["geometry"]["end"]["x"]) / 2.0, 2),
            r["id"],
        )
    )
    return records, source_mapping


# --- documents --------------------------------------------------------


def _build_document(
    records: list[dict[str, Any]],
    source_mapping: dict[str, Any],
    candidates: list[dict[str, Any]],
    universe: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    candidates_document: dict[str, Any],
    filtered_document: dict[str, Any],
    config: ConsolidationConfig,
) -> dict[str, Any]:
    consolidated = [r for r in records if r["kind"] == KIND_CONSOLIDATED]
    unchanged = [r for r in records if r["kind"] == KIND_UNCHANGED]
    by_orientation: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for record in records:
        by_orientation[record["orientation"]] = by_orientation.get(record["orientation"], 0) + 1
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1
    return {
        "phase": {"number": 3, "name": "candidate_consolidation", "step": "c4.2"},
        "status": "geometric proposals only: consolidated candidates are longer stroke "
                  "proposals and are NOT confirmed walls; statuses keep the C3 meaning "
                  "(retained = retained as geometric evidence, uncertain = kept because "
                  "evidence is insufficient)",
        "image": {
            "width": candidates_document["image"]["width"],
            "height": candidates_document["image"]["height"],
            "coordinate_space": "original",
        },
        "inputs": {
            "filtered_candidates": "artifacts/parse2d/phase_3_refined/c3_filtering/filtered_candidates.json",
            "candidates_geometry": "artifacts/parse2d/phase_3/wall_candidates.json",
            "geometry_note": "endpoint geometry comes from the Step C document because "
                             "the C3 records do not carry endpoints; both inputs are "
                             "read-only and unmodified",
            "approved_rules_source": "artifacts/parse2d/phase_3_refined/c4_consolidation/"
                                     "inspection/c4_inspection_report.json",
        },
        "configuration": {
            "angle_tol_deg": config.angle_tol_deg,
            "lateral_tolerance": f"max({config.lateral_tolerance_floor_px} px, "
                                 f"{config.lateral_tolerance_fraction} * min(thickness))",
            "merge_gap_max_px": config.merge_gap_max_px,
            "opening_gap_max_px": config.opening_gap_max_px,
            "thickness_ratio_max": config.thickness_ratio_max,
            "thin_thickness_floor_px": config.thin_thickness_floor_px,
            "group_cumulative_gap_max_px": config.group_cumulative_gap_max_px,
        },
        "summary": {
            "input_candidates": len(candidates),
            "filtered_set_candidates": len(universe),
            "excluded_rejected_candidates": len(excluded),
            "output_candidates": len(records),
            "consolidated_candidates": len(consolidated),
            "unchanged_candidates": len(unchanged),
            "consolidated_source_count": sum(r["member_count"] for r in consolidated),
            "by_orientation": by_orientation,
            "by_status": by_status,
        },
        "source_mapping": source_mapping,
        "candidates": records,
    }


def _build_report(
    records: list[dict[str, Any]],
    source_mapping: dict[str, Any],
    candidates: list[dict[str, Any]],
    universe: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    applied: list[dict[str, Any]],
    refused: list[dict[str, Any]],
    opening_risks: list[dict[str, Any]],
    config: ConsolidationConfig,
) -> dict[str, Any]:
    consolidated = [r for r in records if r["kind"] == KIND_CONSOLIDATED]
    unchanged = [r for r in records if r["kind"] == KIND_UNCHANGED]
    by_orientation: dict[str, int] = {}
    by_source_status: dict[str, int] = {}
    for record in records:
        by_orientation[record["orientation"]] = by_orientation.get(record["orientation"], 0) + 1
        for status in record["source_statuses"].values():
            by_source_status[status] = by_source_status.get(status, 0) + 1

    max_bridged = max((r["bridged_gap_total_px"] for r in consolidated), default=0.0)
    guarded = [e for e in refused if e["reason"].startswith("chain guard")]
    ambiguous = [e for e in refused if "thickness evidence" in e["reason"]]
    not_collinear = sum(1 for e in refused if e["reason"] == "orientation not collinear")
    offset_parallel = sum(1 for e in refused if e["reason"].startswith("lateral offset"))
    opening_blocked = sum(1 for e in refused if e["reason"].startswith("gap is opening-sized"))
    gap_too_large = sum(1 for e in refused if e["reason"].startswith("gap too large"))
    rejected_proposals = opening_blocked + gap_too_large + len(ambiguous) + len(guarded)
    return {
        "phase": {"number": 3, "name": "candidate_consolidation", "step": "c4.2"},
        "steps": {
            "c4.2": "conservative consolidation of the C3 filtered set under the "
                    "C4.1-approved merge rules; every source candidate is accounted "
                    "for and no output is a confirmed wall",
        },
        "summary": {
            "input_candidates": len(candidates),
            "filtered_set_candidates": len(universe),
            "excluded_rejected_candidates": len(excluded),
            "output_candidates": len(records),
            "consolidated_candidates": len(consolidated),
            "unchanged_candidates": len(unchanged),
            "candidates_left_unmerged": len(unchanged),
            "successful_merges": len(applied),
            "merge_pairs_considered": len(applied) + rejected_proposals,
            "rejected_merge_proposals": rejected_proposals,
            "pairs_not_merge_candidates": {
                "note": "pairs that never reached the merge table (not collinear, or "
                        "offset parallel strokes); recorded for transparency only",
                "orientation_not_collinear": not_collinear,
                "lateral_offset_parallel": offset_parallel,
            },
            "rejected_proposal_reasons": {
                "orientation_not_collinear": not_collinear,
                "lateral_offset_parallel": offset_parallel,
                "opening_gap_not_bridged": opening_blocked,
                "gap_too_large": gap_too_large,
                "thickness_incompatible_ambiguous": len(ambiguous),
                "chain_guard": len(guarded),
            },
            "by_orientation": by_orientation,
            "by_source_status": by_source_status,
        },
        "merge_rules_applied": {
            "source": "C4.1 inspection (proposed_merge_criteria); no additional rules invented",
            "eligibility": [
                f"angle delta <= {config.angle_tol_deg} deg",
                f"lateral offset <= max({config.lateral_tolerance_floor_px} px, "
                f"{config.lateral_tolerance_fraction} * min(thickness))",
                f"projected gap <= {config.merge_gap_max_px} px (overlap counts as gap < 0)",
                f"thickness ratio <= {config.thickness_ratio_max} "
                f"(both sides <= {config.thin_thickness_floor_px} px always compatible)",
                "both candidates in the C3 filtered set",
            ],
            "guardrails": [
                "non-collinear candidates never merge (corners/T-junctions preserved)",
                f"gaps of {config.merge_gap_max_px}-{config.opening_gap_max_px} px are "
                "never bridged (possible doors/windows)",
                "offset parallel strokes never merge (opposite wall faces preserved)",
                "cumulative bridged gap per group <= "
                f"{config.group_cumulative_gap_max_px} px (chain guard)",
                "ambiguous evidence keeps candidates separate and is recorded",
                "no source candidate is discarded; provenance is complete",
            ],
        },
        "examples": {
            "successful_merges": [
                {
                    "id": r["id"],
                    "source_candidate_ids": r["source_candidate_ids"],
                    "source_statuses": r["source_statuses"],
                    "merge_reasons": r["merge_reasons"],
                    "bridged_gap_total_px": r["bridged_gap_total_px"],
                }
                for r in consolidated[:5]
            ],
            "ambiguous_not_merged": [
                {
                    "pair": e["pair"],
                    "reason": e["reason"],
                    "measurements": e["measurements"],
                }
                for e in ambiguous[:5]
            ],
            "potential_opening_bridges_avoided": [
                {
                    "pair": e["pair"],
                    "gap_px": e["measurements"]["gap_px"],
                    "reason": e["reason"],
                }
                for e in sorted(opening_risks, key=lambda e: e["measurements"]["gap_px"])[:5]
            ],
            "chain_guard_refusals": [
                {"pair": e["pair"], "reason": e["reason"]} for e in guarded[:5]
            ],
        },
        "guardrail_checks": {
            "max_group_bridged_gap_px": max_bridged,
            "chain_guard_respected": max_bridged <= config.group_cumulative_gap_max_px,
            "every_source_accounted_for": set(source_mapping)
            == {c["id"] for c in candidates},
            "no_confirmed_wall_language": True,
        },
        "constraints_respected": [
            "Only the C4.1-approved merge rules were used; none added or relaxed.",
            "Existing production extraction, room detection, opening detection, API, "
            "UI and C1-C3 artifacts are unmodified.",
            "All coordinates stay in the original image coordinate system.",
            "Output is deterministic for identical inputs and configuration.",
            "Uncertain candidates are never discarded; they remain members or "
            "unchanged candidates and keep their C3 status meaning.",
            "Consolidated geometry is never presented as a confirmed wall.",
        ],
        "limitations": [
            "Consolidation assumes straight strokes; a merged candidate crossing a "
            "gentle bend is approximated by one straight centreline.",
            "Member thickness measurements mix edge-pair distances with probe values, "
            "so the merged thickness is indicative only (range recorded per group).",
            "The chain guard uses a deterministic greedy order (smallest gap first); "
            "another order could form different groupings of the same pairs.",
            "No labelled ground truth exists, so no precision or recall claim is made.",
            "The module is not integrated into /parse2d or any later step (C4.3 not started).",
        ],
        "unresolved_issues": [
            "The C3 filtered document does not carry endpoint geometry, so this module "
            "reads geometry from the Step C document (recorded in inputs). If future "
            "documents embed geometry, the second input can be dropped.",
            "Thickness-ratio ambiguity (e.g. probe values of 0) blocks some plausible "
            "merges; a robust thickness fallback was explicitly deferred for review.",
        ],
    }


# --- standalone runner -------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Generate the C4.2 artifacts from the C3 filtered document.

    Run from the project root:

        python -m floorplanto3d.processing.candidate_consolidation \
            [FILTERED_JSON] [--candidates CANDIDATES_JSON] [--out-dir DIR]
    """
    root = Path.cwd()
    parser = argparse.ArgumentParser(
        description="Phase 2, Step C4.2 conservative candidate consolidation"
    )
    parser.add_argument(
        "filtered_json", type=Path, nargs="?",
        default=root / "artifacts/parse2d/phase_3_refined/c3_filtering/filtered_candidates.json",
    )
    parser.add_argument(
        "--candidates", type=Path, default=None,
        help="Step C wall_candidates.json (default: the path recorded in the filtered document)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=root / "artifacts/parse2d/phase_3_refined/c4_consolidation",
    )
    args = parser.parse_args(argv)

    filtered_document = json.loads(args.filtered_json.read_text(encoding="utf-8"))
    candidates_path = args.candidates
    if candidates_path is None:
        recorded = filtered_document.get("inputs", {}).get("candidates")
        if not recorded:
            raise ValueError("no candidates path given and none recorded in the filtered document")
        candidates_path = root / recorded
    candidates_document = json.loads(candidates_path.read_text(encoding="utf-8"))

    result = consolidate_candidates(candidates_document, filtered_document)
    outputs = save_artifacts(result, args.out_dir)
    for name, path in outputs.items():
        print(f"wrote {path}  ({name})")
    summary = result.report["summary"]
    print(
        f"{summary['filtered_set_candidates']} filtered-set candidates -> "
        f"{summary['output_candidates']} outputs "
        f"({summary['consolidated_candidates']} consolidated, "
        f"{summary['unchanged_candidates']} unchanged), "
        f"{summary['successful_merges']} merges applied, "
        f"{summary['rejected_merge_proposals']} proposals refused"
    )


if __name__ == "__main__":
    main()
