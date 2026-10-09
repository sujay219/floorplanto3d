import cv2
import numpy as np

from floorplanto3d.processing.preprocessing import preprocess
from floorplanto3d.processing.rooms import extract_rooms
from floorplanto3d.processing.walls import detect_segments, extract_walls


def _white(height: int = 240, width: int = 360) -> np.ndarray:
    return np.full((height, width, 3), 255, np.uint8)


def _walls_from_rgb(image: np.ndarray):
    pre = preprocess(image)
    return pre, detect_segments(pre.binary, min_length=20), extract_walls(
        pre.binary, min_edge_length=20
    )


def test_furniture_and_annotations_only_do_not_create_walls_or_rooms():
    image = _white()
    cv2.rectangle(image, (40, 60), (180, 125), (0, 0, 0), 2)
    cv2.line(image, (45, 92), (175, 92), (0, 0, 0), 2)
    cv2.line(image, (20, 25), (330, 25), (0, 0, 0), 2)
    cv2.putText(
        image,
        "ROOM 101",
        (55, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    _, raw, walls = _walls_from_rgb(image)

    assert len(raw) > 0
    assert walls == []
    assert extract_rooms(walls) == []


def test_single_thick_horizontal_wall_reconstructs_one_centreline():
    image = _white(160, 360)
    cv2.line(image, (30, 80), (330, 80), (0, 0, 0), 18)

    _, raw, walls = _walls_from_rgb(image)

    assert len(raw) > 1
    assert len(walls) == 1
    wall = walls[0]
    assert wall.orientation == "horizontal"
    assert abs(wall.start.y - 80) <= 3
    assert abs(wall.end.y - 80) <= 3
    assert wall.thickness is not None
    assert abs(wall.thickness - 18) <= 3


def test_single_thick_diagonal_wall_reconstructs_one_centreline():
    image = _white(220, 360)
    cv2.line(image, (40, 170), (320, 45), (0, 0, 0), 12)

    _, raw, walls = _walls_from_rgb(image)

    assert len(raw) > 1
    assert len(walls) == 1
    wall = walls[0]
    assert wall.orientation == "diagonal"
    assert abs(wall.angle_deg - 156) <= 3
    assert wall.thickness is not None
    assert abs(wall.thickness - 12) <= 3


def test_two_distinct_parallel_walls_are_not_merged():
    image = _white(220, 360)
    cv2.line(image, (35, 65), (325, 65), (0, 0, 0), 14)
    cv2.line(image, (35, 145), (325, 145), (0, 0, 0), 14)

    _, _, walls = _walls_from_rgb(image)

    assert len(walls) == 2
    ys = sorted((wall.start.y + wall.end.y) / 2 for wall in walls)
    assert abs(ys[0] - 65) <= 4
    assert abs(ys[1] - 145) <= 4


def test_one_wall_drawn_as_two_parallel_edges_becomes_one_wall():
    binary = np.zeros((160, 360), np.uint8)
    cv2.line(binary, (30, 72), (330, 72), 255, 3)
    cv2.line(binary, (30, 88), (330, 88), 255, 3)

    raw = detect_segments(binary, min_length=20)
    walls = extract_walls(binary, min_edge_length=20)

    assert len(raw) > 1
    assert len(walls) == 1
    wall = walls[0]
    assert wall.orientation == "horizontal"
    assert abs(wall.start.y - 80) <= 3
    assert abs(wall.end.y - 80) <= 3


def test_furnished_plan_keeps_major_walls_and_suppresses_furniture():
    image = _white(420, 560)
    cv2.rectangle(image, (50, 50), (510, 360), (0, 0, 0), 12)
    cv2.line(image, (280, 50), (280, 360), (0, 0, 0), 10)

    cv2.rectangle(image, (85, 95), (220, 170), (0, 0, 0), 3)
    cv2.line(image, (90, 132), (215, 132), (0, 0, 0), 2)
    cv2.rectangle(image, (325, 105), (455, 165), (0, 0, 0), 3)
    cv2.circle(image, (390, 250), 45, (0, 0, 0), 3)
    cv2.line(image, (345, 250), (435, 250), (0, 0, 0), 2)
    cv2.line(image, (50, 30), (510, 30), (0, 0, 0), 2)
    cv2.putText(
        image,
        "BED 12 x 10",
        (85, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "LIVING",
        (330, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    _, raw, walls = _walls_from_rgb(image)
    rooms = extract_rooms(walls, min_area=400)

    assert len(raw) > 50
    assert 5 <= len(walls) <= 6
    assert sum(1 for wall in walls if wall.orientation == "vertical") >= 3
    assert sum(1 for wall in walls if wall.orientation == "horizontal") >= 2
    assert len(rooms) >= 1
