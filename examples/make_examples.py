"""Generate the example floor plans shown in README.md.

Run from the project root:

    python examples/make_examples.py

Writes PNGs into ``examples/plans/``. These are synthetic drawings, not real
architectural plans, so they carry no dimension annotations and the library
therefore reports them in pixels.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
from plans import blank_plan, four_room_plan, skewed_plan, two_room_plan, window_plan

OUTPUT = Path(__file__).parent / "plans"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, plan in {
        "two-room": two_room_plan(),
        "four-room": four_room_plan(),
        "windows": window_plan(),
        "skewed": skewed_plan(),
        "blank": blank_plan(),
    }.items():
        path = OUTPUT / f"{name}.png"
        cv2.imwrite(str(path), plan)
        print(f"wrote {path}  ({plan.shape[1]}x{plan.shape[0]})")


if __name__ == "__main__":
    main()
