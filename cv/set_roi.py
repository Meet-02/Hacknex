"""
set_roi.py

Interactive ROI Polygon Drawing Tool for Hacknex.
Allows drawing, editing, and saving a camera-specific polygon Region of Interest (ROI).

Usage:
    python set_roi.py --camera cam_03
    python set_roi.py --camera cam_01 --source videos/cam_01.mp4

Controls:
    Left click  : Add vertex point
    Right click : Remove last vertex
    'r'         : Reset / clear all points
    's'         : Save polygon to cv/roi/<camera_id>.json (requires >= 3 points)
    'q' / ESC   : Quit without saving

Automated / headless support:
    python set_roi.py --camera cam_03 --points "[[100,150],[650,150],[650,550],[100,550]]"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

from roi import save_roi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERAS = {
    "cam_01": os.path.join(BASE_DIR, "videos", "cam_01.mp4"),
    "cam_02": os.path.join(BASE_DIR, "videos", "cam_02.mp4"),
    "cam_03": os.path.join(BASE_DIR, "videos", "cam_03.mp4"),
}

# State for mouse callback
_points = []
_original_frame = None
_window_name = ""


def draw_editor_overlay(base_img: np.ndarray, pts: list) -> np.ndarray:
    """Draw points, lines, filled polygon preview, and instructions."""
    img = base_img.copy()
    overlay = base_img.copy()
    h, w = img.shape[:2]

    # Draw polygon fill & edges
    if len(pts) >= 3:
        poly = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [poly], (255, 0, 255))
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        cv2.polylines(img, [poly], isClosed=True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    elif len(pts) == 2:
        cv2.line(img, tuple(pts[0]), tuple(pts[1]), (255, 0, 255), 2, lineType=cv2.LINE_AA)

    # Draw vertices
    for i, pt in enumerate(pts):
        cv2.circle(img, tuple(pt), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, tuple(pt), 6, (255, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    # Top stats bar
    cv2.rectangle(img, (0, 0), (w, 36), (30, 30, 30), -1)
    status_text = f"ROI Editor | Points: {len(pts)} (minimum 3 required to save)"
    cv2.putText(img, status_text, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    # Bottom instruction bar
    cv2.rectangle(img, (0, h - 36), (w, h), (30, 30, 30), -1)
    help_text = "Left-click: add | Right-click: undo | 's': save | 'r': reset | 'q'/ESC: quit"
    cv2.putText(img, help_text, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def mouse_callback(event, x, y, flags, param):
    global _points, _original_frame, _window_name
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append([int(x), int(y)])
        updated = draw_editor_overlay(_original_frame, _points)
        cv2.imshow(_window_name, updated)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if _points:
            _points.pop()
            updated = draw_editor_overlay(_original_frame, _points)
            cv2.imshow(_window_name, updated)


def get_reference_frame(source_val) -> tuple[Optional[np.ndarray], int, int]:
    """Open source and grab first valid frame."""
    if isinstance(source_val, str) and not os.path.exists(source_val):
        print(f"[set_roi] ERROR: Source video file not found: {source_val}")
        return None, 0, 0

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[set_roi] ERROR: Could not open video source: {source_val}")
        return None, 0, 0

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"[set_roi] ERROR: Could not read first frame from {source_val}")
        return None, 0, 0

    h, w = frame.shape[:2]
    return frame, w, h


def main():
    global _points, _original_frame, _window_name

    parser = argparse.ArgumentParser(description="Hacknex CCTV ROI Polygon Editor")
    parser.add_argument("--camera", default="cam_03", choices=list(CAMERAS.keys()) + ["webcam"],
                        help="Camera ID to configure ROI for (default: cam_03)")
    parser.add_argument("--source", default=None,
                        help="Override video path or webcam index (0, 1...)")
    parser.add_argument("--points", default=None,
                        help="Programmatic/automated points JSON string, e.g. '[[100,200],[500,200],[500,500],[100,500]]'")
    args = parser.parse_args()

    camera_id = args.camera
    source = args.source
    if source is None:
        if camera_id == "webcam":
            source = 0
        else:
            source = CAMERAS.get(camera_id, CAMERAS["cam_03"])
    elif source.isdigit():
        source = int(source)

    frame, frame_w, frame_h = get_reference_frame(source)
    if frame is None:
        sys.exit(1)

    print(f"\n[set_roi] Camera: {camera_id}")
    print(f"[set_roi] Source: {source}")
    print(f"[set_roi] Frame Resolution: {frame_w}x{frame_h}")

    # Programmatic mode (for automated tests or headless setups)
    if args.points:
        try:
            if args.points.strip().startswith("["):
                pts = json.loads(args.points)
            else:
                raw_coords = [float(x) for x in re.split(r"[\s,;]+", args.points.strip()) if x]
                if len(raw_coords) % 2 != 0:
                    raise ValueError("Points must be pairs of (x, y) coordinates")
                pts = [[int(raw_coords[i]), int(raw_coords[i + 1])] for i in range(0, len(raw_coords), 2)]
            if not isinstance(pts, list) or len(pts) < 3:
                raise ValueError("Must provide at least 3 points")
            saved_path = save_roi(camera_id, pts, [frame_w, frame_h])
            print(f"[set_roi] SUCCESS: Programmatic ROI saved to {saved_path} with {len(pts)} vertices:")
            for i, p in enumerate(pts):
                print(f"   Vertex {i + 1}: ({p[0]}, {p[1]})")
            return
        except Exception as e:
            print(f"[set_roi] ERROR parsing programmatic points: {e}")
            sys.exit(1)

    # Interactive UI mode
    _points = []
    _original_frame = frame
    _window_name = f"Draw ROI Polygon - {camera_id} ({frame_w}x{frame_h})"

    print("\nStarting interactive ROI editor...")
    print("Controls:")
    print("  Left-click  : Add vertex point")
    print("  Right-click : Undo last vertex")
    print("  's'         : Save polygon to cv/roi/<camera_id>.json")
    print("  'r'         : Reset all points")
    print("  'q' / ESC   : Quit without saving\n")

    cv2.namedWindow(_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_window_name, min(1280, frame_w), min(720, frame_h))
    cv2.setMouseCallback(_window_name, mouse_callback)

    initial_display = draw_editor_overlay(_original_frame, _points)
    cv2.imshow(_window_name, initial_display)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            if len(_points) < 3:
                print(f"[set_roi] Need at least 3 points to form a polygon! Current count: {len(_points)}")
                continue
            saved_path = save_roi(camera_id, _points, [frame_w, frame_h])
            print(f"\n[set_roi] SUCCESS: ROI saved to {saved_path} ({len(_points)} vertices):")
            for i, p in enumerate(_points):
                print(f"   Vertex {i + 1}: ({p[0]}, {p[1]})")
            break
        elif key == ord('r'):
            _points = []
            updated = draw_editor_overlay(_original_frame, _points)
            cv2.imshow(_window_name, updated)
            print("[set_roi] Points reset.")
        elif key in (ord('q'), 27):
            print("[set_roi] Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
