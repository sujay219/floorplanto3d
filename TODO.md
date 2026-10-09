Phase  1 results:

Phase 1 diagnosis is complete. I did not modify files.
Root Cause
The false positives come from the current pipeline treating all dark foreground ink as candidate wall material, then accepting any long enough detected line as a wall. Hough is involved, but it is not the whole root cause; the real failure chain is:
preprocess preserves furniture/text/annotations as foreground -> detect_segments runs on the entire binary ink mask -> extract_walls has no architectural-wall validation before emitting Wall objects -> extract_rooms polygonizes noisy wall-like geometry.
Key code points:
- [preprocessing.py (line 158)](/Users/sujju/Desktop/work/floorplan/FloorPlanTo3D/floorplanto3d/processing/preprocessing.py:158) binarizes all dark ink.
- [walls.py (line 270)](/Users/sujju/Desktop/work/floorplan/FloorPlanTo3D/floorplanto3d/processing/walls.py:270) runs cv2.HoughLinesP over the full binary mask.
- [walls.py (line 471)](/Users/sujju/Desktop/work/floorplan/FloorPlanTo3D/floorplanto3d/processing/walls.py:471) emits every deduped segment as a Wall, even if thickness is None or symbol-like.
- [rooms.py (line 266)](/Users/sujju/Desktop/work/floorplan/FloorPlanTo3D/floorplanto3d/processing/rooms.py:266) snaps/extends/polygonizes whatever walls were passed in.
Findings Against Your Checklist
1. Yes, preprocessing preserves furniture, text, dimension lines, annotations, and symbols as foreground. There is no semantic filtering or component rejection.
2. Yes, the detector treats arbitrary long strokes as walls if they pass minLineLength.
3. Yes, individual furniture edges can become walls. A synthetic furniture/annotation-only image still produced 3 final walls.
4. Yes, thick walls generate duplicate parallel centrelines. A single thick horizontal wall produced 2 final walls.
5. Intersections are handled later in room polygonization using snapping, extension, and Shapely noding, but this also amplifies bad inputs because false wall segments become graph edges.
6. Rectification can introduce artifacts or crop/warp around a detected rectangular-ish contour. In the probe, furniture-only content was rectified and then detected as walls.
7. The detector does not strictly assume only horizontal/vertical walls. Hough runs all angles, _snap_to_axis only snaps near-axis segments, and diagonal detections can survive.
8. Yes, noisy detected walls cause incorrect room polygonization. A simple plan with 5 intended wall runs plus furniture/text yielded 15 walls and 7 rooms.
Probe Results
Using controlled synthetic masks:
- Thick-walled simple plan plus furniture/text: 112 raw Hough segments -> 15 final walls -> 7 rooms.
- Furniture/annotations only: 13 raw segments -> 3 final walls.
- Single thick horizontal wall: 10 raw segments -> 2 final walls.
- Single thick diagonal wall: 9 raw segments -> 2 final walls.
So Phase 2 should not simply “replace Hough.” It should first add foreground filtering and wall-candidate validation: suppress text/furniture/annotations, require wall-like stroke width/continuity/context, collapse thick-stroke faces into one centreline, and reject candidates before room polygonization.