"""
pipeline.py  —  Hacknex CV pipeline  (full implementation)

Architecture per frame:
  Stage A) Full-frame YOLO detection + ByteTrack person tracking
  Stage B) Second-pass YOLO on each reliable person crop (lower conf)
  Stage C) Person-object spatial association  (associator.py)
  Stage D) Optional YOLO segmentation → mask evidence
  Stage E) Temporal confirmation via ObjectTrackBuffer
  Stage F) Evidence scoring → candidate / probable / confirmed
  Stage G) Visualization + evidence frame saving + JSON output

Examples (run from inside cv/):
    python pipeline.py --camera cam_03
    python pipeline.py --camera cam_03 --show
    python pipeline.py --camera cam_03 --show --frame-skip 2
    python pipeline.py --camera cam_03 --show --frame-skip 2 --conf 0.45
    python pipeline.py --camera cam_03 --object-conf 0.20 --no-seg
    python pipeline.py --camera cam_03 --model yolo26s.pt
    python pipeline.py --camera all --frame-skip 5 --save-json
    python pipeline.py --camera cam_03 --show --all-classes      # debug: every YOLO class
    python pipeline.py --no-detect --max-frames 100              # Stage 1 only
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from typing import Optional

import cv2

from associator import ObjectCandidate, ObjectTrackBuffer, associate_object_with_person
from config import (
    ALERT_MIN_CONF,
    CARRIED_OBJECT_CLASSES,
    CONFIRMED_ASSOCIATION_THRESHOLD,
    CROP_CONF,
    DETECTION_MODEL,
    EVIDENCE_SAVE_MIN_STATE,
    MIN_OBJECT_CONFIRM_FRAMES,
    MIN_PERSON_TRACK_FRAMES,
    OBJECT_CONF,
    PERSON_ASSOCIABLE_CLASSES,
    PERSON_CLASSES,
    PERSON_CONF,
    PERSON_CROP_PADDING,
    PROBABLE_ASSOCIATION_THRESHOLD,
    PROBABLE_FRAMES,
    RELEVANT_CLASSES,
    SEGMENTATION_MODEL,
    SUSPICIOUS_OBJECTS,
    category_of,
)
from enricher import check_suspicious_support, enrich_event
from crops import PERSON_CROP_PADDING as _CROPS_DEFAULT, crop_with_padding, save_crop
from tracker import MIN_PERSON_TRACK_FRAMES as _TRACKER_DEFAULT, PersonTrackFilter
from video_input import iter_frames
from roi import (
    draw_roi_overlay,
    get_detection_roi_point,
    load_roi,
    point_in_roi,
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ── Camera → source map ───────────────────────────────────────────────────────
CAMERAS = {
    "cam_01": os.path.join(BASE_DIR, "videos", "cam_01.mp4"),
    "cam_02": os.path.join(BASE_DIR, "videos", "cam_02.mp4"),
    "cam_03": os.path.join(BASE_DIR, "videos", "cam_03.mp4"),
}

# ── Visualization colours (BGR) ───────────────────────────────────────────────
_COL_PERSON_RELIABLE = (0,   220, 0)    # green
_COL_PERSON_HELD     = (80,  80,  80)   # grey (shadow / not yet trusted)
_COL_CONFIRMED       = (0,   200, 255)  # yellow-cyan
_COL_PROBABLE        = (0,   140, 255)  # orange
_COL_CANDIDATE       = (100, 100, 200)  # muted purple
_COL_CROP_BOX        = (0,   255, 255)  # thin yellow = padded crop region
_COL_ALERT           = (0,   0,   255)  # red — suspicious / alert object

_STATE_ORDER = {"candidate": 0, "probable": 1, "confirmed": 2}


def _state_gte(state: str, min_state: str) -> bool:
    return _STATE_ORDER.get(state, 0) >= _STATE_ORDER.get(min_state, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def format_time(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def label_of(det: dict) -> str:
    """'person#17(0.94)' when tracked, 'handbag(0.67)' otherwise."""
    tid = f'#{det["track_id"]}' if det.get("track_id") is not None else ""
    return f'{det["class_name"]}{tid}({det["confidence"]:.2f})'


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
def draw_annotated(frame, persons: list, associated_objects: list,
                   ignored_persons: list = None,
                   show_candidates: bool = False,
                   roi_polygon: Optional[np.ndarray] = None,
                   roi_detections: list = None) -> "numpy.ndarray":
    """
    Return an annotated copy of the frame.

    persons           – reliable person dicts  (have track_id)
    associated_objects– ObjectCandidate list for this frame
    ignored_persons   – held-back person dicts (shadow / pending)
    show_candidates   – if True, also draw candidate-state objects
    roi_polygon       – optional camera ROI polygon (draws overlay and mutes outside)
    """
    annotated = frame.copy()

    # ── ROI polygon background overlay ──────────────────────────────────
    roi_detections = roi_detections or []
    roi_danger = bool(roi_detections)
    if roi_polygon is not None:
        state = "DANGER" if roi_danger else "CLEAR"
        annotated = draw_roi_overlay(annotated, roi_polygon, is_active=True, label=f"ROI: {state}")

    # Held-back persons (grey, thin)
    for det in (ignored_persons or []):
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _COL_PERSON_HELD, 1)
        reason = det.get("held_back_reason", "held")
        cv2.putText(annotated, f'({reason[:20]})',
                    (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, _COL_PERSON_HELD, 1, cv2.LINE_AA)

    # Reliable persons
    for det in persons:
        x1, y1, x2, y2 = det["bbox"]
        if det.get("crop_bbox"):
            cx1, cy1, cx2, cy2 = det["crop_bbox"]
            cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), _COL_CROP_BOX, 1)

        colour = _COL_PERSON_RELIABLE
        thickness = 2
        label = f'person #{det.get("track_id","?")} ({det["confidence"]:.2f})'

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)
        cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    # Associated objects
    for cand in associated_objects:
        state = cand.association_state
        if state == "candidate" and not show_candidates:
            continue

        is_suspicious = cand.class_name in SUSPICIOUS_OBJECTS
        x1, y1, x2, y2 = cand.bbox

        if is_suspicious and state in ("confirmed", "probable"):
            # Alert: bright red, thick border, ALERT prefix
            colour    = _COL_ALERT
            thickness = 3
            label = (f'!! POTENTIAL {cand.class_name.upper()} '
                     f'{cand.confidence:.2f} | {state} | {cand.evidence_score:.2f}')
        elif state == "confirmed":
            colour    = _COL_CONFIRMED
            thickness = 2
            label = (f'{cand.class_name} {cand.confidence:.2f} '
                     f'| confirmed | {cand.evidence_score:.2f}')
        elif state == "probable":
            colour    = _COL_PROBABLE
            thickness = 2
            label = (f'{cand.class_name} {cand.confidence:.2f} '
                     f'| probable | {cand.evidence_score:.2f}')
        else:
            colour    = _COL_CANDIDATE
            thickness = 1
            label = f'{cand.class_name} {cand.confidence:.2f} | candidate'

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)
        cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    # Immediate danger visualization for every detector class, independent of
    # person tracking, object association, or temporal evidence thresholds.
    for det in roi_detections:
        x1, y1, x2, y2 = det["bbox"]
        name = det.get("class_name", "object").upper()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _COL_ALERT, 3)
        label = f"[!] DANGER: {name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        top = max(0, y1 - th - 10)
        cv2.rectangle(annotated, (x1, top), (x1 + tw + 8, top + th + 8), _COL_ALERT, -1)
        cv2.putText(annotated, label, (x1 + 4, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Top banner for active suspicious-object alerts (Section 17) ──────
    active_alerts = [c for c in associated_objects
                     if c.class_name in SUSPICIOUS_OBJECTS
                     and c.association_state in ("confirmed", "probable")]
    if active_alerts:
        al = active_alerts[0]
        banner = f"ALERT: POTENTIAL {al.class_name.upper()} DETECTED (Person #{al.person_track_id})"
        w = annotated.shape[1]
        cv2.rectangle(annotated, (0, 0), (w, 34), (0, 0, 200), -1)
        cv2.putText(annotated, banner, (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# Evidence frame / crop saving
# ─────────────────────────────────────────────────────────────────────────────
_saved_evidence_keys: set = set()   # deduplicate: (camera, track_id, class, int(ts))


def save_evidence_frame(frame, camera_id: str, track_id: int,
                        class_name: str, timestamp: float,
                        frame_number: int) -> Optional[str]:
    """
    Save an evidence frame.  One per (camera, person, class, ~second).
    Returns the saved path (relative to BASE_DIR) or None if already saved.
    """
    key = (camera_id, track_id, class_name, int(timestamp))
    if key in _saved_evidence_keys:
        return None
    _saved_evidence_keys.add(key)

    folder = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(folder, exist_ok=True)
    ts_str = f"{timestamp:.2f}s".replace(".", "_")
    filename = f"{camera_id}_person_{track_id}_{class_name}_{ts_str}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, frame)
    return os.path.relpath(path, BASE_DIR).replace(os.sep, "/")


def save_evidence_crop(frame, bbox: list, camera_id: str, track_id: int,
                       class_name: str, timestamp: float) -> Optional[str]:
    """Save a tight crop around the associated object."""
    x1, y1, x2, y2 = bbox
    pad = 10
    h, w = frame.shape[:2]
    cx1 = max(0, x1 - pad);  cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad);  cy2 = min(h, y2 + pad)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    folder = os.path.join(OUTPUT_DIR, "crops")
    os.makedirs(folder, exist_ok=True)
    ts_str = f"{timestamp:.2f}s".replace(".", "_")
    filename = f"{camera_id}_person_{track_id}_{class_name}_{ts_str}.jpg"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, crop)
    return os.path.relpath(path, BASE_DIR).replace(os.sep, "/")


# ─────────────────────────────────────────────────────────────────────────────
# Association Episode Tracker (Sections 2 & 9: group consecutive observations)
# ─────────────────────────────────────────────────────────────────────────────
class AssociationEpisodeTracker:
    """
    Groups consecutive detections of (person_track_id, class_name) into
    a single consolidated event representing the interaction episode.
    Preserves the strongest evidence frame, relevant clip interval, and ROI metadata.
    """
    def __init__(self, camera_id: str, source, max_gap_seconds: float = 2.0):
        self.camera_id = camera_id
        self.source = source
        self.max_gap_seconds = max_gap_seconds
        self.roi_polygon: Optional[np.ndarray] = None
        self.roi_meta: Optional[dict] = None
        # key: (person_track_id, class_name) -> episode dict
        self.active: dict[tuple, dict] = {}
        self.completed: list[dict] = []

    def set_roi(self, polygon: Optional[np.ndarray], meta: Optional[dict]):
        self.roi_polygon = polygon
        self.roi_meta = meta

    def update(self, frame_num: int, timestamp: float, frame_img,
               reliable_persons: list, valid_candidates: list,
               args, supported_suspicious: set):
        seen_keys = set()
        for cand in valid_candidates:
            key = (cand.person_track_id, cand.class_name)
            seen_keys.add(key)
            person_info = next((p for p in reliable_persons
                                if p.get("track_id") == cand.person_track_id), None)

            # Ensure ROI membership is set on candidate
            cand_pt = get_detection_roi_point(cand.bbox, mode="bottom_center")
            cand.roi_point = list(cand_pt)
            cand.inside_roi = point_in_roi(cand_pt, self.roi_polygon) if self.roi_polygon is not None else None

            if key in self.active:
                ep = self.active[key]
                if timestamp - ep["last_timestamp"] > self.max_gap_seconds:
                    # Previous episode had a significant gap; finalize it and start fresh
                    self._finalize_episode(ep, args, supported_suspicious)
                    ep = self._start_episode(cand, person_info, frame_num, timestamp, frame_img, reliable_persons)
                    self.active[key] = ep
                else:
                    ep["last_frame"] = frame_num
                    ep["last_timestamp"] = timestamp
                    ep["frames_seen"] += 1
                    if cand.evidence_score > ep["best_cand"].evidence_score:
                        ep["best_cand"] = cand
                        ep["best_frame_num"] = frame_num
                        ep["best_timestamp"] = timestamp
                        ep["best_person"] = person_info
                        ep["best_frame_img"] = frame_img.copy()
                        ep["best_reliable_persons"] = reliable_persons
                        ep["best_roi_point"] = list(cand_pt)
                        ep["best_inside_roi"] = cand.inside_roi
            else:
                self.active[key] = self._start_episode(cand, person_info, frame_num, timestamp, frame_img, reliable_persons)

        # Finalize any active episodes whose observation timed out
        expired_keys = [k for k, ep in self.active.items()
                        if timestamp - ep["last_timestamp"] > self.max_gap_seconds]
        for k in expired_keys:
            self._finalize_episode(self.active.pop(k), args, supported_suspicious)

    def _start_episode(self, cand, person_info, frame_num, timestamp, frame_img, reliable_persons):
        cand_pt = get_detection_roi_point(cand.bbox, mode="bottom_center")
        inside = point_in_roi(cand_pt, self.roi_polygon) if self.roi_polygon is not None else None
        cand.roi_point = list(cand_pt)
        cand.inside_roi = inside

        return {
            "camera_id": self.camera_id,
            "track_id": cand.person_track_id,
            "class_name": cand.class_name,
            "start_frame": frame_num,
            "last_frame": frame_num,
            "start_timestamp": timestamp,
            "last_timestamp": timestamp,
            "frames_seen": 1,
            "best_cand": cand,
            "best_frame_num": frame_num,
            "best_timestamp": timestamp,
            "best_person": person_info,
            "best_frame_img": frame_img.copy(),
            "best_reliable_persons": reliable_persons,
            "best_roi_point": list(cand_pt),
            "best_inside_roi": inside,
        }

    def _finalize_episode(self, ep, args, supported_suspicious):
        cand = ep["best_cand"]
        best_frame = ep["best_frame_img"]
        frame_num = ep["best_frame_num"]
        timestamp = ep["best_timestamp"]
        track_id = ep["track_id"]
        class_name = ep["class_name"]

        # Annotate and save the strongest evidence frame
        annotated_evidence = draw_annotated(
            best_frame, ep["best_reliable_persons"], [cand],
            ignored_persons=None, show_candidates=False,
            roi_polygon=self.roi_polygon,
        )
        frame_path = save_evidence_frame(
            annotated_evidence, self.camera_id, track_id,
            class_name, timestamp, frame_num
        )
        crop_path = save_evidence_crop(
            best_frame, cand.bbox, self.camera_id, track_id,
            class_name, timestamp
        )

        clip_start = max(0.0, round(ep["start_timestamp"] - 1.0, 3))
        clip_end = round(ep["last_timestamp"] + 1.0, 3)
        video_path = os.path.relpath(self.source, BASE_DIR).replace(os.sep, "/") \
                     if isinstance(self.source, str) else str(self.source)

        roi_enabled = self.roi_polygon is not None
        inside_roi = ep["best_inside_roi"] if roi_enabled else None
        roi_pt = ep["best_roi_point"] if roi_enabled else None
        roi_id = self.roi_meta["roi_id"] if (roi_enabled and self.roi_meta) else None

        person_pt = None
        person_inside = None
        if ep["best_person"] and ep["best_person"].get("bbox"):
            person_pt = list(get_detection_roi_point(ep["best_person"]["bbox"], mode="bottom_center"))
            person_inside = point_in_roi(tuple(person_pt), self.roi_polygon) if roi_enabled else None

        event = {
            "event_id": str(uuid.uuid4()),
            "camera_id": self.camera_id,
            "frame_number": frame_num,
            "frame": frame_num,
            "timestamp": timestamp,
            "person_track_id": track_id,
            "person": {
                "track_id": track_id,
                "confidence": ep["best_person"]["confidence"] if ep["best_person"] else None,
                "bbox": ep["best_person"]["bbox"] if ep["best_person"] else None,
                "roi_point": person_pt,
                "inside_roi": person_inside,
            },
            "object": cand.to_dict(),
            "evidence_frame_path": frame_path,
            "evidence_crop_path": crop_path,
            "video_path": video_path,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "frames_observed": ep["frames_seen"],
            "episode_duration": round(ep["last_timestamp"] - ep["start_timestamp"], 3),
            "inside_roi": inside_roi,
            "roi_id": roi_id,
            "roi_point": roi_pt,
            "roi": {
                "enabled": roi_enabled,
                "inside": inside_roi,
                "point": roi_pt,
            },
        }
        enrich_event(
            event,
            supported_suspicious,
            recording_start=getattr(args, "recording_start", None),
            timezone=getattr(args, "timezone", None),
        )
        self.completed.append(event)

    def finalize_all(self, args, supported_suspicious) -> list:
        for ep in list(self.active.values()):
            self._finalize_episode(ep, args, supported_suspicious)
        self.active.clear()
        return self.completed


# ─────────────────────────────────────────────────────────────────────────────
# Main camera processing loop
# ─────────────────────────────────────────────────────────────────────────────
def process_camera(camera_id: str, source, detector, seg_detector, args,
                   supported_suspicious: Optional[set] = None) -> list:
    """
    Run the full pipeline on a single camera.
    Returns a list of JSON-serialisable event dicts (confirmed/probable associations).
    """
    from detector import remap_to_frame, split_relevant

    events        = []   # confirmed / probable association events
    frame_records = []   # per-frame debug records

    frames_dir = os.path.join(OUTPUT_DIR, "frames", camera_id)
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    use_tracking = bool(detector) and not args.no_track
    if detector and detector.has_tracked:
        detector.reset_tracking()

    person_filter   = PersonTrackFilter(min_frames=args.min_track_frames)
    object_buffer   = ObjectTrackBuffer(
        min_confirm_frames=args.min_confirm_frames,
        probable_frames=args.probable_frames,
        crop_padding=args.crop_padding,
    )
    episode_tracker = AssociationEpisodeTracker(camera_id, source)
    flagged_shadows: set = set()

    processed = 0
    roi_polygon = None
    roi_meta = None
    roi_checked = False
    previous_roi_inside: set[tuple] = set()

    for item in iter_frames(camera_id, source, frame_skip=args.frame_skip):
        frame      = item["frame"]
        frame_num  = item["frame_number"]
        timestamp  = item["timestamp"]

        # ── Load and scale camera ROI on first frame ──────────────────────
        if not roi_checked:
            roi_checked = True
            if not getattr(args, "no_roi", False):
                frame_h, frame_w = frame.shape[:2]
                roi_polygon, roi_meta = load_roi(camera_id, target_size=(frame_w, frame_h))
                episode_tracker.set_roi(roi_polygon, roi_meta)

        # ── Stage A: full-frame detection ─────────────────────────────────
        all_dets, ignored = [], []
        roi_detections, roi_entries = [], []
        reliable_persons, held_back_persons = [], []
        all_assoc_this_frame: list[ObjectCandidate] = []

        if detector:
            all_dets = detector.detect(frame, track=use_tracking)

            # Test every YOLO detection before the existing relevant-class and
            # association filters, so any class can trigger an ROI danger.
            if roi_polygon is not None:
                for det in all_dets:
                    pt = get_detection_roi_point(det["bbox"], mode="bottom_center")
                    det["roi_point"] = list(pt)
                    det["inside_roi"] = point_in_roi(pt, roi_polygon)
                    if det["inside_roi"]:
                        roi_detections.append(det)
                        identity = ((det.get("class_name", "object"), "track", det["track_id"])
                                    if det.get("track_id") is not None else
                                    (det.get("class_name", "object"), "bbox", *det["bbox"]))
                        if identity not in previous_roi_inside:
                            roi_entries.append((det, pt, identity))
            current_roi_inside = set()
            for det in roi_detections:
                identity = ((det.get("class_name", "object"), "track", det["track_id"])
                            if det.get("track_id") is not None else
                            (det.get("class_name", "object"), "bbox", *det["bbox"]))
                current_roi_inside.add(identity)
            previous_roi_inside = current_roi_inside

            if args.all_classes:
                detections = [{**d, "category": category_of(d["class_name"])}
                              for d in all_dets]
            else:
                detections, ignored = split_relevant(all_dets, detector.relevant_classes)

            # ── Person tracking / shadow filtering ────────────────────────
            if use_tracking:
                persons = [d for d in detections if d["category"] == "person"]
                others  = [d for d in detections if d["category"] != "person"]

                reliable_persons, held_back_persons = person_filter.update(timestamp, persons)

                # Attach ROI membership to persons
                for p in reliable_persons:
                    pt = get_detection_roi_point(p["bbox"], mode="bottom_center")
                    p["roi_point"] = list(pt)
                    p["inside_roi"] = point_in_roi(pt, roi_polygon) if roi_polygon is not None else None

                for d in held_back_persons:
                    ignored.append({**d, "ignored_reason": d.get("held_back_reason", "held")})
                    if ("reflection" in d.get("held_back_reason", "")
                            and d["track_id"] not in flagged_shadows):
                        flagged_shadows.add(d["track_id"])
                        print(f'[{camera_id}] frame {frame_num:>6}  '
                              f'held back person#{d["track_id"]}({d["confidence"]:.2f}): '
                              f'{d["held_back_reason"]}')

                # Full-frame object candidates (legacy overlap check when --loose-bags off)
                if not args.loose_bags:
                    from detector import boxes_overlap
                    person_bboxes = [p["bbox"] for p in reliable_persons]
                    kept_others = []
                    for o in others:
                        if o["category"] == "carried_object":
                            if any(boxes_overlap(o["bbox"], p) for p in person_bboxes):
                                kept_others.append(o)
                        else:
                            kept_others.append(o)
                    others = kept_others
            else:
                reliable_persons = [d for d in detections if d["category"] == "person"]
                others = [d for d in detections if d["category"] != "person"]
                for p in reliable_persons:
                    pt = get_detection_roi_point(p["bbox"], mode="bottom_center")
                    p["roi_point"] = list(pt)
                    p["inside_roi"] = point_in_roi(pt, roi_polygon) if roi_polygon is not None else None

            # ── Stage B: second-pass detection on each reliable person crop ──
            for person in reliable_persons:
                crop_img, crop_box = crop_with_padding(frame, person["bbox"],
                                                       args.crop_padding)
                person["crop_bbox"] = crop_box   # store for visualization

                if crop_img is not None:
                    crop_dets = detector.detect_on_crop(crop_img, conf=args.object_conf)

                    for cd in crop_dets:
                        if cd["class_name"] not in PERSON_ASSOCIABLE_CLASSES:
                            continue  # skip persons / vehicles detected in crop
                        if cd["class_name"] in SUSPICIOUS_OBJECTS and cd["confidence"] < ALERT_MIN_CONF:
                            continue

                        # Remap bbox from crop-local to full-frame coords
                        cd["bbox"] = remap_to_frame(cd["bbox"], crop_box)

                        cand = associate_object_with_person(
                            person_bbox=person["bbox"],
                            object_bbox=cd["bbox"],
                            detection_confidence=cd["confidence"],
                            class_name=cd["class_name"],
                            person_track_id=person["track_id"],
                            crop_padding=args.crop_padding,
                            source="crop",
                        )
                        if cand is not None:
                            all_assoc_this_frame.append(cand)

            # ── Full-frame objects → also try association ─────────────────
            for o in others:
                if o["category"] != "carried_object":
                    continue
                if o["class_name"] in SUSPICIOUS_OBJECTS and o["confidence"] < ALERT_MIN_CONF:
                    continue
                for person in reliable_persons:
                    cand = associate_object_with_person(
                        person_bbox=person["bbox"],
                        object_bbox=o["bbox"],
                        detection_confidence=o["confidence"],
                        class_name=o["class_name"],
                        person_track_id=person["track_id"],
                        crop_padding=args.crop_padding,
                        source="fullframe",
                    )
                    if cand is not None:
                        all_assoc_this_frame.append(cand)

            # ── Stage D: optional segmentation ───────────────────────────
            if seg_detector and all_assoc_this_frame:
                try:
                    seg_results = seg_detector.segment(frame, conf=0.25)
                    # Build class → mask lookup (first hit per class for simplicity)
                    seg_map = {}
                    person_mask = None
                    for sr in seg_results:
                        if sr["class_name"] == "person" and person_mask is None:
                            person_mask = sr["mask"]
                        elif sr["class_name"] not in seg_map:
                            seg_map[sr["class_name"]] = sr["mask"]

                    from associator import calculate_seg_score
                    for cand in all_assoc_this_frame:
                        obj_mask = seg_map.get(cand.class_name)
                        if person_mask is not None and obj_mask is not None:
                            cand.seg_score = calculate_seg_score(person_mask, obj_mask)
                            cand.segmentation_available = True
                except Exception as e:
                    pass   # segmentation failure must not kill the pipeline

            # ── Stage E+F: temporal update → evidence score + state ──────
            object_buffer.update(all_assoc_this_frame)

            # ── Deduplicate: if same (person, class) appears via both crop
            #    and full-frame in the same frame, keep the higher evidence ──
            best: dict[tuple, ObjectCandidate] = {}
            for cand in all_assoc_this_frame:
                key = (cand.person_track_id, cand.class_name)
                if key not in best or cand.evidence_score > best[key].evidence_score:
                    best[key] = cand
            all_assoc_this_frame = list(best.values())

            # ── Save crops of reliable persons (--save-crops) ────────────
            if args.save_crops:
                crops_dir = os.path.join(OUTPUT_DIR, "crops", camera_id)
                for person in reliable_persons:
                    crop_img2, _ = crop_with_padding(frame, person["bbox"], args.crop_padding)
                    if crop_img2 is not None:
                        path = save_crop(crop_img2, crops_dir, person["class_name"],
                                         person["track_id"], frame_num)
                        person["crop_path"] = os.path.relpath(path, BASE_DIR).replace(os.sep, "/")

            # ── Save evidence frames for probable / confirmed ─────────────
            valid_cands = [c for c in all_assoc_this_frame
                           if _state_gte(c.association_state, args.evidence_min_state)]

            if not getattr(args, "raw_events", False):
                # Section 2 & 9: group consecutive observations into episodes
                episode_tracker.update(
                    frame_num=frame_num,
                    timestamp=timestamp,
                    frame_img=frame,
                    reliable_persons=reliable_persons,
                    valid_candidates=valid_cands,
                    args=args,
                    supported_suspicious=supported_suspicious or set(),
                )
            else:
                # Per-frame raw event emission mode
                for cand in valid_cands:
                    evidence_frame = draw_annotated(
                        frame, reliable_persons, [cand],
                        ignored_persons=None, show_candidates=False
                    )
                    frame_path = save_evidence_frame(
                        evidence_frame, camera_id, cand.person_track_id,
                        cand.class_name, timestamp, frame_num
                    )
                    crop_path = save_evidence_crop(
                        frame, cand.bbox, camera_id, cand.person_track_id,
                        cand.class_name, timestamp
                    )
                    clip_start = max(0.0, round(timestamp - 1.0, 3))
                    clip_end   = round(timestamp + 1.0, 3)
                    video_path = os.path.relpath(source, BASE_DIR).replace(os.sep, "/") \
                                 if isinstance(source, str) else str(source)

                    event = {
                        "event_id":     str(uuid.uuid4()),
                        "camera_id":    camera_id,
                        "frame_number": frame_num,
                        "frame":        frame_num,
                        "timestamp":    timestamp,
                        "person_track_id": cand.person_track_id,
                        "person": {
                            "track_id":   cand.person_track_id,
                            "confidence": next(
                                (p["confidence"] for p in reliable_persons
                                 if p.get("track_id") == cand.person_track_id), None),
                            "bbox":       next(
                                (p["bbox"] for p in reliable_persons
                                 if p.get("track_id") == cand.person_track_id), None),
                        },
                        "object": cand.to_dict(),
                        "evidence_frame_path": frame_path,
                        "evidence_crop_path":  crop_path,
                        "video_path":   video_path,
                        "clip_start":   clip_start,
                        "clip_end":     clip_end,
                    }
                    enrich_event(
                        event,
                        supported_suspicious or set(),
                        recording_start=getattr(args, "recording_start", None),
                        timezone=getattr(args, "timezone", None),
                    )
                    events.append(event)

        # ── Progress line ─────────────────────────────────────────────────
        confirmed_cnt = sum(1 for c in all_assoc_this_frame
                            if c.association_state == "confirmed")
        probable_cnt  = sum(1 for c in all_assoc_this_frame
                            if c.association_state == "probable")
        person_labels = ", ".join(label_of(p) for p in reliable_persons)
        obj_labels    = ", ".join(
            f'{c.class_name}({c.confidence:.2f})[{c.association_state}]'
            for c in all_assoc_this_frame
            if c.association_state in ("confirmed", "probable")
        )
        print(f'[{camera_id}] frame {frame_num:>6} t={format_time(timestamp)}  '
              f'persons={len(reliable_persons)}  '
              f'confirmed={confirmed_cnt}  probable={probable_cnt}'
              + (f'  | {person_labels}' if person_labels else '')
              + (f'  || {obj_labels}'   if obj_labels    else ''))

        # ── Optional live window ──────────────────────────────────────────
        if args.save_frames or args.show or roi_entries:
            annotated = draw_annotated(
                frame, reliable_persons, all_assoc_this_frame,
                ignored_persons=held_back_persons if args.show_held else [],
                show_candidates=args.show_candidates,
                roi_polygon=roi_polygon,
                roi_detections=roi_detections,
            )
            # One structured ROI ENTRY event per outside-to-inside transition.
            # This does not depend on the suspicious-object evidence pipeline.
            for det, pt, identity in roi_entries:
                cls = det.get("class_name", "object")
                track_id = det.get("track_id")
                evidence_dir = os.path.join(OUTPUT_DIR, "frames")
                os.makedirs(evidence_dir, exist_ok=True)
                evidence_name = (f"{camera_id}_roi_entry_{cls}_{track_id if track_id is not None else frame_num}_"
                                 f"{frame_num}.jpg")
                evidence_abs = os.path.join(evidence_dir, evidence_name)
                cv2.imwrite(evidence_abs, annotated)
                video_path = (os.path.relpath(source, BASE_DIR).replace(os.sep, "/")
                              if isinstance(source, str) else str(source))
                event = {
                    "event_id": str(uuid.uuid4()), "camera_id": camera_id,
                    "frame_number": frame_num, "frame": frame_num, "timestamp": timestamp,
                    "person_track_id": track_id if cls == "person" else None,
                    "person": {"track_id": track_id if cls == "person" else None,
                               "confidence": det.get("confidence") if cls == "person" else None,
                               "bbox": det["bbox"] if cls == "person" else None},
                    "object": {"class_name": cls, "class": cls,
                               "detection_confidence": det.get("confidence"),
                               "confidence": det.get("confidence"), "bbox": det["bbox"],
                               "association_state": "roi_intrusion", "evidence_score": 0.0,
                               "spatial_score": 0.0, "temporal_score": 0.0, "seg_score": 0.0},
                    "video_path": video_path,
                    "clip_start": max(0.0, round(timestamp - 1.0, 3)),
                    "clip_end": round(timestamp + 1.0, 3),
                    "evidence_frame_path": os.path.relpath(evidence_abs, BASE_DIR).replace(os.sep, "/"),
                    "inside_roi": True, "roi_point": list(pt), "roi_id": f"{camera_id}_default",
                    "roi_alert": True, "roi_alert_type": "intrusion",
                    "roi_alert_label": f"{cls.title()} entered danger zone",
                    "roi_alert_severity": "high", "roi_detection_key": list(identity),
                }
                enrich_event(event, supported_suspicious or set(),
                             recording_start=getattr(args, "recording_start", None),
                             timezone=getattr(args, "timezone", None))
                events.append(event)
            if args.save_frames:
                path = os.path.join(frames_dir, f'{camera_id}_{frame_num:05d}.jpg')
                cv2.imwrite(path, annotated)
            if args.show:
                cv2.imshow(camera_id, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit requested.")
                    break

        processed += 1
        if args.max_frames and processed >= args.max_frames:
            print(f"[{camera_id}] reached --max-frames={args.max_frames}")
            break

    if args.show:
        cv2.destroyAllWindows()

    if not getattr(args, "raw_events", False):
        events.extend(episode_tracker.finalize_all(args, supported_suspicious or set()))

    ev_confirmed = sum(1 for e in events if e["object"]["association_state"] == "confirmed")
    ev_probable  = sum(1 for e in events if e["object"]["association_state"] == "probable")
    print(f"[{camera_id}] done: {processed} frames | "
          f"{len(events)} evidence events  "
          f"(confirmed={ev_confirmed}  probable={ev_probable})")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# JSON output
# ─────────────────────────────────────────────────────────────────────────────
def save_json(all_events: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load existing file so different cameras don't overwrite each other
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Backward compatibility: enrich any existing events that lack new fields
    for e in existing:
        if "description" not in e:
            if "video_path" not in e:
                cam = e.get("camera_id", "cam_01")
                e["video_path"] = f"videos/{cam}.mp4"
                ts = e.get("timestamp", 0.0)
                e["clip_start"] = max(0.0, round(ts - 1.0, 3))
                e["clip_end"] = round(ts + 1.0, 3)
            enrich_event(e, set(SUSPICIOUS_OBJECTS.keys()))

    # Merge: existing + new, deduplicate by event_id
    seen_ids = {e["event_id"] for e in existing}
    merged   = existing + [e for e in all_events if e["event_id"] not in seen_ids]

    with open(path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"[JSON] wrote {len(all_events)} new events → {path}  "
          f"(total in file: {len(merged)})")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Hacknex CV pipeline – multi-stream video intelligence"
    )

    # ── Camera / source ───────────────────────────────────────────────────
    parser.add_argument("--camera", default="cam_01",
                        choices=list(CAMERAS) + ["all"],
                        help="camera to process (default: cam_01)")
    parser.add_argument("--source", default=None,
                        help="override source: a file path or webcam index (0,1,…)")

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--model", default=DETECTION_MODEL,
                        help="detection weights file (default: yolo26s.pt)")
    parser.add_argument("--seg-model", default=SEGMENTATION_MODEL,
                        help="segmentation weights file (default: yolo26n-seg.pt)")
    parser.add_argument("--no-seg", action="store_true",
                        help="disable segmentation even if the model is available")

    # ── Confidence ────────────────────────────────────────────────────────
    parser.add_argument("--conf", type=float, default=PERSON_CONF,
                        help=f"person detection confidence threshold (default {PERSON_CONF})")
    parser.add_argument("--object-conf", type=float, default=OBJECT_CONF,
                        help=f"object candidate confidence threshold in person crops "
                             f"(default {OBJECT_CONF})")

    # ── Association / temporal ────────────────────────────────────────────
    parser.add_argument("--min-track-frames", type=int, default=MIN_PERSON_TRACK_FRAMES,
                        help=f"frames before a person track is trusted (default {MIN_PERSON_TRACK_FRAMES})")
    parser.add_argument("--min-confirm-frames", type=int, default=MIN_OBJECT_CONFIRM_FRAMES,
                        help=f"frames to reach 'confirmed' association (default {MIN_OBJECT_CONFIRM_FRAMES})")
    parser.add_argument("--probable-frames", type=int, default=PROBABLE_FRAMES,
                        help=f"frames to reach 'probable' association (default {PROBABLE_FRAMES})")
    parser.add_argument("--evidence-min-state", default=EVIDENCE_SAVE_MIN_STATE,
                        choices=["candidate", "probable", "confirmed"],
                        help="minimum association state to save as evidence "
                             f"(default: {EVIDENCE_SAVE_MIN_STATE})")

    # ── Crop ──────────────────────────────────────────────────────────────
    parser.add_argument("--crop-padding", type=float, default=PERSON_CROP_PADDING,
                        help=f"person crop padding fraction (default {PERSON_CROP_PADDING})")

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument("--save-frames", action="store_true",
                        help="save annotated frames to output/frames/<camera>/")
    parser.add_argument("--save-crops", action="store_true",
                        help="save padded person crops to output/crops/<camera>/")
    parser.add_argument("--save-json", action="store_true",
                        help="write association events to output/detections.json")

    # ── Playback / debug ──────────────────────────────────────────────────
    parser.add_argument("--show", action="store_true",
                        help="show live annotated window (press q to quit)")
    parser.add_argument("--show-candidates", action="store_true",
                        help="also draw low-confidence candidate objects in the window")
    parser.add_argument("--show-held", action="store_true",
                        help="draw held-back (shadow/pending) persons in the window")
    parser.add_argument("--all-classes", action="store_true",
                        help="debug: treat every YOLO class as relevant")
    parser.add_argument("--loose-bags", action="store_true",
                        help="keep full-frame object detections even if not near any person")
    parser.add_argument("--no-track", action="store_true",
                        help="plain detection without tracking / person filtering")
    parser.add_argument("--no-detect", action="store_true",
                        help="Stage 1 only: read frames, skip YOLO entirely")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="process every Nth frame (default 1; try 3–5 for speed)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N processed frames (0 = all)")
    parser.add_argument("--raw-events", action="store_true",
                        help="output per-frame events instead of grouping consecutive detections into episodes")
    parser.add_argument("--no-roi", action="store_true",
                        help="disable camera ROI filtering even if roi/<camera>.json exists")
    parser.add_argument("--recording-start", default=None,
                        help="recording start ISO timestamp (e.g. 2026-10-08T12:00:00+05:30)")
    parser.add_argument("--timezone", default=None,
                        help="timezone name (e.g. Asia/Kolkata)")

    # ── Legacy compat ─────────────────────────────────────────────────────
    parser.add_argument("--classes", nargs="*", default=None,
                        help="override relevant-class list, e.g. --classes person handbag")
    parser.add_argument("--bag-conf", type=float, default=None,
                        help="legacy: separate threshold for bag classes")

    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────
    for name in ("conf", "object_conf", "crop_padding", "bag_conf"):
        val = getattr(args, name)
        if val is not None and not (0.0 <= val <= 1.0):
            parser.error(f"--{name.replace('_','-')} must be between 0 and 1")

    # ── Build job list ────────────────────────────────────────────────────
    if args.camera == "all":
        jobs = list(CAMERAS.items())
    else:
        source = CAMERAS[args.camera]
        if args.source is not None:
            source = int(args.source) if args.source.isdigit() else args.source
        jobs = [(args.camera, source)]

    # ── Build detector ────────────────────────────────────────────────────
    detector = None
    if not args.no_detect:
        from detector import Detector

        relevant = args.classes if args.classes else RELEVANT_CLASSES
        # Legacy --bag-conf support
        class_conf = ({name: args.bag_conf for name in CARRIED_OBJECT_CLASSES}
                      if args.bag_conf is not None else None)
        detector = Detector(
            model_path=args.model,
            conf_threshold=args.conf,
            relevant_classes=relevant,
            class_conf=class_conf,
        )
        print(f"[Model] {os.path.basename(args.model)}  conf={args.conf:.2f}  "
              f"object_conf={args.object_conf:.2f}")

    # ── Check which suspicious classes the loaded model supports ──────────
    _supported_suspicious: set = set()
    if detector:
        _supported_suspicious = check_suspicious_support(detector.names)

    # ── Build segmentation detector (optional) ────────────────────────────
    seg_detector = None
    if detector and not args.no_seg:
        from detector import SegDetector
        seg_detector = SegDetector.create(model_path=args.seg_model)
        if seg_detector is None:
            print("[pipeline] Continuing without segmentation.")

    # ── Process cameras ───────────────────────────────────────────────────
    all_events = []
    for camera_id, source in jobs:
        try:
            events = process_camera(camera_id, source, detector, seg_detector, args,
                                    _supported_suspicious)
            all_events.extend(events)
        except RuntimeError as err:
            print(f"[ERROR] {camera_id}: {err}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    if args.save_json and all_events:
        json_path = os.path.join(OUTPUT_DIR, "detections.json")
        save_json(all_events, json_path)
    elif all_events:
        print(f"[pipeline] {len(all_events)} evidence events generated. "
              f"Run with --save-json to persist them.")


if __name__ == "__main__":
    main()
