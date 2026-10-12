You are an expert in architectural floor-plan computer vision,
OpenCV, and 2D-to-3D reconstruction.

I am building a floor-plan parser in Python using OpenCV and FastAPI.

I will provide a furnished apartment floor-plan image.

YOUR TASK

Create a visual annotation of the EXISTING floor plan.

Do not redesign, regenerate, redraw, or modify the apartment layout.

Preserve the original image and its geometry exactly. Add colored
annotations as an overlay.

ANNOTATION LEGEND

1. WALLS — RED
   - Identify visible architectural wall boundaries.
   - Draw red lines along the wall centerlines where they can
     be reliably inferred.
   - Preserve wall corners, T-junctions, and intersections.
   - Do not classify furniture outlines as walls.
   - Do not connect unrelated wall segments just because they
     are close to each other.

2. WINDOWS — BLUE DASHED LINES
   - Identify likely windows embedded in external or internal walls.
   - Draw blue dashed lines over the corresponding openings.
   - Do not classify ordinary wall gaps as windows without evidence.

3. VENTILATION OPENINGS — GREEN DASHED LINES
   - Identify ventilation openings, especially small openings
     associated with toilets and bathrooms.
   - Draw green dashed lines at their actual locations.
   - Do not confuse ventilation openings with windows or plumbing fixtures.

4. DOORS
   - Draw door opening locations and swing arcs in black.
   - Use the visible door leaf and swing arc when available.
   - Do not invent door positions or swing directions.

IMPLEMENTATION REQUIREMENTS

Do not use an image-generation model to recreate the floor plan.

Instead, use Python and OpenCV or Pillow to draw annotations
directly onto a copy of the original image.

Keep the original image unchanged.

If geometry is uncertain, leave it unannotated or mark it as
an uncertain candidate. Never invent architectural geometry.

Generate these outputs:

1. annotated_floor_plan.png
   The original floor plan with colored annotations.

2. walls_only.png
   The original floor plan with wall annotations only.

3. openings_only.png
   The original floor plan with window, ventilation, and door annotations.

4. detections.json
   Structured coordinates and classifications of detected elements.

5. detection_report.json
   Detection counts, confidence scores where available,
   and unresolved ambiguities.

Use the original image dimensions and coordinate system.

Do not crop, rotate, resize, or warp the image.

Show the annotated result for visual inspection.

IMPORTANT

The goal is geometric accuracy, not visual beautification.

Do not claim that a wall, window, or ventilation opening has been
correctly identified unless there is supporting visual evidence.

If you cannot reliably identify a feature, report that limitation.