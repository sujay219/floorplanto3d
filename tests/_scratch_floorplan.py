"""Synthetic floor plan: a simple 2-room layout drawn as thin wall strokes."""
import numpy as np
import cv2


def make_floorplan(w=600, h=400, thickness=8):
    img = np.full((h, w), 255, dtype=np.uint8)
    black = 0

    # Outer boundary
    cv2.rectangle(img, (50, 50), (550, 350), black, thickness)
    # Interior dividing wall
    cv2.line(img, (300, 50), (300, 350), black, thickness)
    # Door gap (erase segment) in the dividing wall
    cv2.line(img, (300, 180), (300, 240), 255, thickness)
    return img


if __name__ == "__main__":
    img = make_floorplan()
    cv2.imwrite("synthetic_floorplan.png", img)

    gray = img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"RETR_EXTERNAL contours: {len(contours)}")
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w_, h_ = cv2.boundingRect(c)
        print(f"  area={area:9.1f} bbox=({x},{y},{w_},{h_})")

    print("\n--- with RETR_TREE (all levels) ---")
    contours2, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"contours: {len(contours2)}")
    areas = sorted((cv2.contourArea(c) for c in contours2), reverse=True)[:12]
    print(f"top areas: {[round(a, 1) for a in areas]}")