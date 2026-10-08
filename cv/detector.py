"""
detector.py

Thin wrapper around Ultralytics YOLO.

Public API
──────────
    Detector          – detection + tracking  (Stage A: full frame)
    SegDetector       – optional segmentation model (returns masks)
    split_relevant()  – splits all detections into (relevant, ignored)
    remap_to_frame()  – maps crop-relative bbox back to full-frame coords
    boxes_overlap()   – simple bbox overlap test
    keep_carried_items() – legacy proximity filter (kept for backward compat)

Usage (Stage A)
───────────────
    detector = Detector(model_path="yolo26s.pt", conf_threshold=0.45)
    all_detections = detector.detect(frame, track=True)
    relevant, ignored = split_relevant(all_detections, detector.relevant_classes)

Usage (Stage B – person crop)
──────────────────────────────
    crop, padded_box = crop_with_padding(frame, person_bbox, padding=0.20)
    crop_dets = detector.detect_on_crop(crop, conf=0.20)
    # remap bboxes back to full frame:
    for d in crop_dets:
        d["bbox"] = remap_to_frame(d["bbox"], padded_box)

Usage (Segmentation – optional)
────────────────────────────────
    seg = SegDetector("yolo26n-seg.pt")   # None if file not present
    if seg:
        masks = seg.segment(frame, bboxes=[person_bbox, object_bbox])
"""

import os

from ultralytics import YOLO

from config import (
    CARRIED_OBJECT_CLASSES,    # backward-compat alias → PERSON_ASSOCIABLE_CLASSES
    PERSON_ASSOCIABLE_CLASSES,
    RELEVANT_CLASSES,
    SEGMENTATION_MODEL,
    category_of,
)


# ─────────────────────────────────────────────────────────────────────────────
# Stage A: Full-frame detection + ByteTrack tracking
# ─────────────────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self,
                 model_path: str = "yolo26s.pt",
                 conf_threshold: float = 0.45,
                 relevant_classes=None,
                 device=None,
                 class_conf: dict = None):
        """
        model_path     : Ultralytics weights file.  Downloaded automatically
                         the first time if not found locally.
        conf_threshold : Global minimum confidence (Stage A).
        relevant_classes : set/list of class names to treat as relevant.
                         Defaults to config.RELEVANT_CLASSES.  Unknown names
                         are skipped with a printed note.
        device         : None = auto, or "cpu", "cuda:0", "mps".
        class_conf     : per-class threshold overrides, e.g. {"handbag": 0.20}.
        """
        self.model_path     = model_path
        self.model          = YOLO(model_path)
        self.tracker_cfg    = "bytetrack.yaml"   # ships with Ultralytics
        self.has_tracked    = False
        self.conf_threshold = conf_threshold
        self.class_conf     = class_conf or {}
        self.device         = device

        # Run YOLO at the LOWEST threshold; stricter per-class limits applied after.
        self.min_conf = min([conf_threshold] + list(self.class_conf.values()))

        # model.names → {0: "person", 1: "bicycle", ...}
        self.names = self.model.names

        # Which of the wanted classes does THIS model actually know?
        wanted = set(RELEVANT_CLASSES if relevant_classes is None else relevant_classes)
        known  = set(self.names.values())
        self.relevant_classes = wanted & known
        missing = sorted(wanted - known)
        print(f"[Detector] model={os.path.basename(model_path)}  "
              f"relevant classes={len(self.relevant_classes)}")
        if missing:
            print(f"  NOTE: model has no class named {missing}; skipping them")

    # ── Tracking reset ────────────────────────────────────────────────────────
    def reset_tracking(self):
        """Forget all tracks.  Call when switching cameras to prevent ID leakage."""
        self.model = YOLO(self.model_path)
        self.has_tracked = False

    # ── Full-frame detection (Stage A) ────────────────────────────────────────
    def detect(self, frame, track: bool = False) -> list:
        """
        Run YOLO on one frame.

        track=True  → ByteTrack (model.track); each detection gets a stable track_id.
                       Call IN ORDER, one camera at a time.
        track=False → plain detection; track_id is None.

        Returns a list of dicts:
            {"class_name", "track_id", "confidence", "bbox": [x1,y1,x2,y2]}
        """
        if frame is None or frame.size == 0:
            return []

        if track:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_cfg,
                conf=self.min_conf,
                device=self.device,
                verbose=False,
            )
            self.has_tracked = True
        else:
            results = self.model.predict(
                frame,
                conf=self.min_conf,
                device=self.device,
                verbose=False,
            )

        return self._parse_results(results)

    # ── Second-pass: detect on a person crop (Stage B) ────────────────────────
    def detect_on_crop(self, crop, conf: float = 0.20) -> list:
        """
        Run plain detection on a person crop image at a lower confidence.

        Returns raw detections in CROP-RELATIVE coordinates.
        The caller must call remap_to_frame() to convert to full-frame coords.

        track_id is always None here (crops are not tracked by ByteTrack).
        """
        if crop is None or crop.size == 0:
            return []
        results = self.model.predict(
            crop,
            conf=conf,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results)

    # ── Internal: parse Ultralytics result ───────────────────────────────────
    def _parse_results(self, results) -> list:
        boxes  = results[0].boxes
        xyxys  = boxes.xyxy.cpu().numpy()
        confs  = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        ids = (boxes.id.cpu().numpy()
               if getattr(boxes, "id", None) is not None
               else [None] * len(xyxys))

        detections = []
        for xyxy, conf, cls_id, track_id in zip(xyxys, confs, cls_ids, ids):
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            name = self.names[int(cls_id)]
            # Apply per-class or global threshold
            if float(conf) < self.class_conf.get(name, self.conf_threshold):
                continue
            detections.append({
                "class_name": name,
                "track_id":   int(track_id) if track_id is not None else None,
                "confidence": round(float(conf), 3),
                "bbox":       [x1, y1, x2, y2],
            })
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Optional segmentation model
# ─────────────────────────────────────────────────────────────────────────────
class SegDetector:
    """
    Wraps a YOLO segmentation model (e.g. yolo26n-seg.pt).

    Use SegDetector.create() instead of the constructor so that a missing
    model file produces None (not an exception), keeping the pipeline alive.

    Usage:
        seg = SegDetector.create("yolo26n-seg.pt")
        if seg:
            results = seg.segment(frame)
            # results is a list of {"class_name", "confidence", "bbox",
            #                        "mask": numpy_bool_array}
    """

    def __init__(self, model_path: str, device=None):
        self.model  = YOLO(model_path)
        self.names  = self.model.names
        self.device = device

    @classmethod
    def create(cls, model_path: str = SEGMENTATION_MODEL, device=None):
        """
        Try to load the segmentation model.
        Returns a SegDetector instance, or None if the file is not present.
        Does NOT auto-download.
        """
        if not os.path.exists(model_path):
            print(f"[SegDetector] WARNING: segmentation model not found at "
                  f"'{model_path}'.  Segmentation disabled.  "
                  f"Place yolo26n-seg.pt in cv/ to enable it.")
            return None
        try:
            inst = cls(model_path, device=device)
            print(f"[SegDetector] loaded '{os.path.basename(model_path)}'")
            return inst
        except Exception as exc:
            print(f"[SegDetector] WARNING: could not load '{model_path}': {exc}")
            return None

    def segment(self, frame, conf: float = 0.25) -> list:
        """
        Run segmentation inference on a frame.

        Returns a list of dicts:
            {
              "class_name":  str,
              "confidence":  float,
              "bbox":        [x1, y1, x2, y2],   # full-frame
              "mask":        numpy bool array (H×W), same size as frame
            }
        Entries without a mask (model glitch) are skipped.
        """
        if frame is None or frame.size == 0:
            return []
        try:
            results = self.model.predict(frame, conf=conf,
                                         device=self.device, verbose=False)
            out = []
            boxes = results[0].boxes
            masks = results[0].masks

            if masks is None:
                return []

            import numpy as np
            h, w = frame.shape[:2]

            xyxys   = boxes.xyxy.cpu().numpy()
            confs   = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()

            for i, (xyxy, conf_v, cls_id) in enumerate(zip(xyxys, confs, cls_ids)):
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
                name = self.names[int(cls_id)]
                # Resize the mask to full-frame size
                try:
                    raw_mask = masks.data[i].cpu().numpy()
                    import cv2
                    mask_resized = cv2.resize(
                        raw_mask.astype("float32"), (w, h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                except Exception:
                    mask_resized = None

                out.append({
                    "class_name":  name,
                    "confidence":  round(float(conf_v), 3),
                    "bbox":        [x1, y1, x2, y2],
                    "mask":        mask_resized,
                })
            return out
        except Exception as exc:
            print(f"[SegDetector] segment() error: {exc}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────
def remap_to_frame(crop_bbox: list, padded_box: list) -> list:
    """
    Convert a bounding box from crop-relative coordinates to full-frame coords.

    crop_bbox   : [x1, y1, x2, y2] inside the crop
    padded_box  : [x1, y1, x2, y2] of the crop in the full frame
                  (returned by crops.crop_with_padding)

    Returns [x1, y1, x2, y2] in full-frame coordinates.
    """
    ox, oy = padded_box[0], padded_box[1]
    return [
        crop_bbox[0] + ox,
        crop_bbox[1] + oy,
        crop_bbox[2] + ox,
        crop_bbox[3] + oy,
    ]


def split_relevant(detections: list, relevant_classes: set):
    """
    Split ALL YOLO detections into (relevant, ignored).

    relevant  – class is in relevant_classes; gets a 'category' key added.
    ignored   – everything else; kept for debugging but not processed.
    """
    relevant, ignored = [], []
    for det in detections:
        if det["class_name"] in relevant_classes:
            relevant.append({**det, "category": category_of(det["class_name"])})
        else:
            ignored.append(det)
    return relevant, ignored


def boxes_overlap(a: list, b: list) -> bool:
    """True if two [x1, y1, x2, y2] boxes overlap at all."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def keep_carried_items(detections: list, carried_classes: set) -> list:
    """
    Legacy proximity filter: drop any carried-item detection that does not
    touch any person box.  People, vehicles, and other classes pass through.

    This is the OLD single-stage check; Stage B + associator.py replace it
    for the two-stage pipeline.  Kept for --no-track / legacy modes.
    """
    people = [d["bbox"] for d in detections if d["class_name"] == "person"]
    kept = []
    for det in detections:
        if det["class_name"] in carried_classes:
            if any(boxes_overlap(det["bbox"], p) for p in people):
                kept.append(det)
        else:
            kept.append(det)
    return kept
