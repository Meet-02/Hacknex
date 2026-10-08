"""
roi.py

Camera-specific polygon Region of Interest (ROI) utility module for Hacknex.
Adapts the Pathguard polygon ROI approach while preserving Hacknex architecture.

Public API:
    load_roi(camera_id, target_size, roi_dir) -> (polygon, meta_dict)
    save_roi(camera_id, polygon, frame_size, roi_dir) -> path
    scale_roi(polygon, original_size, target_size) -> np.ndarray
    point_in_roi(point, polygon) -> bool
    get_detection_roi_point(bbox, mode) -> (float, float)
    draw_roi_overlay(frame, polygon, is_active, label) -> frame
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROI_DIR = os.path.join(BASE_DIR, "roi")

# Default visualization styling
_COL_ROI_FILL = (255, 0, 255)       # Magenta fill
_COL_ROI_BORDER = (255, 0, 255)     # Magenta border
_COL_ROI_BADGE_BG = (40, 40, 40)    # Dark badge background
_COL_ROI_BADGE_TEXT = (0, 255, 255) # Yellow badge text


def scale_roi(polygon: Union[list, np.ndarray],
              original_size: Tuple[int, int],
              target_size: Tuple[int, int]) -> np.ndarray:
    """
    Scale a polygon from original_size (w, h) to target_size (w, h).

    polygon       : array-like of shape (N, 2)
    original_size : (orig_w, orig_h)
    target_size   : (target_w, target_h)

    Returns numpy float64 array of shape (N, 2).
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size

    if orig_w <= 0 or orig_h <= 0 or target_w <= 0 or target_h <= 0:
        return np.array(polygon, dtype=np.float64)

    sx = float(target_w) / float(orig_w)
    sy = float(target_h) / float(orig_h)

    pts = np.array(polygon, dtype=np.float64)
    return pts * np.array([sx, sy], dtype=np.float64)


def load_roi(camera_id: str,
             target_size: Optional[Tuple[int, int]] = None,
             roi_dir: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Load camera-specific ROI polygon from JSON file.

    camera_id   : e.g. 'cam_01', 'cam_02', 'cam_03'
    target_size : optional (target_w, target_h) for automatic scaling
    roi_dir     : directory containing <camera_id>.json (default: cv/roi)

    Returns:
        (polygon_np_int32, metadata_dict) if valid,
        (None, None) if missing or invalid.
    """
    folder = roi_dir or DEFAULT_ROI_DIR
    roi_file = os.path.join(folder, f"{camera_id}.json")

    # 1. Missing ROI file -> graceful fallback
    if not os.path.exists(roi_file):
        print(f"WARNING: No ROI configured for {camera_id}.\nROI danger alerts disabled.")
        return None, None

    # 2. Parse JSON
    try:
        with open(roi_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"WARNING: Invalid ROI configuration for {camera_id}. ROI danger alerts disabled. ({exc})")
        return None, None

    # 3. Validate structure
    if not isinstance(data, dict) or data.get("camera_id") != camera_id:
        print(f"WARNING: Invalid ROI configuration for {camera_id}. ROI danger alerts disabled.")
        return None, None
    raw_points = data.get("roi_polygon")
    if not isinstance(raw_points, (list, tuple)) or len(raw_points) < 3:
        print(f"WARNING: Invalid ROI configuration for {camera_id}. ROI danger alerts disabled.")
        return None, None

    try:
        points = np.array(raw_points, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2
                or not np.isfinite(points).all()):
            raise ValueError("Polygon points must be a list of [x, y] coordinates.")
    except Exception as exc:
        print(f"WARNING: Invalid ROI configuration for {camera_id}. ROI danger alerts disabled. ({exc})")
        return None, None
    if len(np.unique(points, axis=0)) < 3 or cv2.contourArea(points.astype(np.float32)) <= 0:
        print(f"WARNING: Invalid ROI configuration for {camera_id}. ROI danger alerts disabled.")
        return None, None

    # 4. Validate frame_size
    ref_size = data.get("frame_size")
    if not (isinstance(ref_size, (list, tuple)) and len(ref_size) == 2
            and all(isinstance(v, (int, float)) and np.isfinite(v) and v > 0 for v in ref_size)):
        print(f"WARNING: Invalid ROI configuration for {camera_id}. ROI danger alerts disabled.")
        return None, None

    orig_w, orig_h = int(ref_size[0]), int(ref_size[1])

    # 5. Apply frame-size aware scaling if target_size is provided
    final_w, final_h = orig_w, orig_h
    if target_size and len(target_size) == 2:
        tgt_w, tgt_h = int(target_size[0]), int(target_size[1])
        if tgt_w > 0 and tgt_h > 0 and (tgt_w != orig_w or tgt_h != orig_h):
            points = scale_roi(points, (orig_w, orig_h), (tgt_w, tgt_h))
            final_w, final_h = tgt_w, tgt_h
            print(f"[ROI] {camera_id}: scaled ROI from {orig_w}x{orig_h} "
                  f"to current frame {tgt_w}x{tgt_h}.")

    polygon = np.round(points).astype(np.int32)
    meta = {
        "camera_id": camera_id,
        "roi_id": f"{camera_id}_default",
        "original_frame_size": [orig_w, orig_h],
        "current_frame_size": [final_w, final_h],
        "vertex_count": len(polygon),
        "roi_file": os.path.relpath(roi_file, BASE_DIR).replace(os.sep, "/"),
    }
    print(f"[ROI] Loaded ROI for {camera_id} ({len(polygon)} vertices) from {meta['roi_file']}")
    return polygon, meta


def save_roi(camera_id: str,
             polygon: Union[list, np.ndarray],
             frame_size: Union[list, tuple],
             roi_dir: Optional[str] = None) -> str:
    """
    Save camera-specific ROI polygon to cv/roi/<camera_id>.json.

    camera_id  : e.g. 'cam_01', 'cam_02', 'cam_03'
    polygon    : list of [x, y] coordinates
    frame_size : [width, height] of the frame on which ROI was drawn
    roi_dir    : target folder (default: cv/roi)

    Returns absolute path of the written file.
    """
    folder = roi_dir or DEFAULT_ROI_DIR
    os.makedirs(folder, exist_ok=True)
    roi_file = os.path.join(folder, f"{camera_id}.json")

    pts_list = [[int(round(p[0])), int(round(p[1]))] for p in polygon]
    payload = {
        "camera_id": camera_id,
        "roi_polygon": pts_list,
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
    }

    with open(roi_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return roi_file


def point_in_roi(point: Tuple[float, float],
                 polygon: Optional[np.ndarray]) -> bool:
    """
    Check if (x, y) point is inside or on the boundary of polygon using OpenCV.

    point   : (x, y) float coordinate
    polygon : numpy array of shape (N, 2), dtype int32 or float32

    Returns True if point is inside or on edge (pointPolygonTest >= 0),
    False otherwise. Missing/invalid polygons are treated as outside.
    """
    if polygon is None or len(polygon) < 3:
        return False

    poly_f32 = polygon.astype(np.float32)
    dist = cv2.pointPolygonTest(poly_f32, (float(point[0]), float(point[1])), False)
    return dist >= 0.0


def get_detection_roi_point(bbox: List[int],
                            mode: str = "bottom_center") -> Tuple[float, float]:
    """
    Determine the reference point for ROI membership.
    Default mode: 'bottom_center' (x = (x1 + x2)/2, y = y2).
    Also supports 'center' (x = (x1 + x2)/2, y = (y1 + y2)/2).
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    if mode == "center":
        cy = (y1 + y2) / 2.0
    else:  # default "bottom_center"
        cy = float(y2)

    return round(cx, 1), round(cy, 1)


def draw_roi_overlay(frame: np.ndarray,
                     polygon: Optional[np.ndarray],
                     is_active: bool = True,
                     label: str = "ROI: CLEAR") -> np.ndarray:
    """
    Draw a semi-transparent polygon overlay and a small unobtrusive status badge.
    """
    if polygon is None or len(polygon) < 3:
        return frame

    annotated = frame.copy()
    overlay = frame.copy()

    danger = "DANGER" in label.upper()
    fill = (0, 0, 255) if danger else _COL_ROI_FILL
    border = (0, 0, 255) if danger else _COL_ROI_BORDER
    badge_text = (255, 255, 255) if danger else _COL_ROI_BADGE_TEXT

    # Semi-transparent fill
    cv2.fillPoly(overlay, [polygon], fill)
    cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)

    # Crisp border
    cv2.polylines(annotated, [polygon], isClosed=True,
                  color=border, thickness=3 if danger else 2, lineType=cv2.LINE_AA)

    # Status badge near top-left or first vertex
    if is_active:
        bx, by = 15, 60
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (bx - 4, by - th - 4), (bx + tw + 6, by + 4),
                      _COL_ROI_BADGE_BG, -1)
        cv2.rectangle(annotated, (bx - 4, by - th - 4), (bx + tw + 6, by + 4),
                      border, 1)
        cv2.putText(annotated, label, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, badge_text, 1, cv2.LINE_AA)

    return annotated
