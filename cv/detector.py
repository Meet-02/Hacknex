"""
detector.py  (Stage 2)

Thin wrapper around Ultralytics YOLO.

    detector = Detector(model_path="yolo26n.pt", conf_threshold=0.50)
    all_detections = detector.detect(frame)                # everything YOLO found
    relevant, ignored = split_relevant(all_detections,     # what the project cares about
                                       detector.relevant_classes)

Each detection:
    {"class_name": "person", "category": "person" (added by split_relevant),
     "confidence": 0.92, "bbox": [x1, y1, x2, y2]}
"""
from ultralytics import YOLO

from config import RELEVANT_CLASSES, category_of


class Detector:
    def __init__(self, model_path="yolo26n.pt", conf_threshold=0.50,
                 relevant_classes=None, device=None, class_conf=None):
        """
        model_path     : any Ultralytics weights file (yolo26s.pt, yolo11s.pt, yolov8n.pt, ...).
                         Downloaded automatically the first time.
        conf_threshold : detections below this confidence are dropped.
        relevant_classes : set/list of class names the project cares about
                         (default: config.RELEVANT_CLASSES). YOLO still runs on ALL
                         classes; this list only decides what counts as relevant
                         (see split_relevant). Names the model does not have are
                         skipped with a printed note instead of crashing.
        device         : None = auto, or "cpu", "cuda:0", "mps".
        class_conf     : optional per-class thresholds that override conf_threshold,
                         e.g. {"handbag": 0.2}. Useful because bags score lower than people.
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.tracker_cfg = "bytetrack.yaml"   # ships with Ultralytics (use "botsort.yaml" for BoT-SORT)
        self.has_tracked = False
        self.conf_threshold = conf_threshold
        self.class_conf = class_conf or {}
        self.device = device

        # YOLO is run at the LOWEST threshold; stricter per-class limits are applied after.
        self.min_conf = min([conf_threshold] + list(self.class_conf.values()))

        # model.names looks like {0: "person", 1: "bicycle", 2: "car", ...}
        self.names = self.model.names

        # Work out which relevant classes THIS model actually supports
        wanted = set(RELEVANT_CLASSES if relevant_classes is None else relevant_classes)
        known = set(self.names.values())
        self.relevant_classes = wanted & known
        missing = sorted(wanted - known)
        print(f"Relevant classes ({len(self.relevant_classes)}): "
              f"{', '.join(sorted(self.relevant_classes))}")
        if missing:
            print(f"  NOTE: this model has no class named {missing}; skipping them")

    def reset_tracking(self):
        """Forget all tracks. Call this when switching to another camera, so
        track ids never leak from one camera into the next."""
        self.model = YOLO(self.model_path)
        self.has_tracked = False

    def detect(self, frame, track=False):
        """Run YOLO on one frame. Returns a list of detection dicts
        (empty list if nothing was found). Handles many objects per frame.

        track=True  -> use ByteTrack (model.track) so each detection gets a
                       'track_id' that stays the same across frames.
                       Call frames IN ORDER, one camera at a time.
        track=False -> plain detection, track_id is None.
        """
        if frame is None or frame.size == 0:
            return []

        if track:
            results = self.model.track(
                frame,
                persist=True,                 # keep tracker state between frames
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
                device=self.device,  # no class filter here: relevance is decided afterwards
                verbose=False,  # silence Ultralytics' per-frame printing
            )

        boxes = results[0].boxes
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
        # boxes.id is None when nothing is being tracked in this frame
        ids = boxes.id.cpu().numpy() if getattr(boxes, "id", None) is not None else [None] * len(xyxys)

        detections = []
        for xyxy, conf, cls_id, track_id in zip(xyxys, confs, cls_ids, ids):
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            name = self.names[int(cls_id)]
            if float(conf) < self.class_conf.get(name, self.conf_threshold):
                continue  # below this class's own threshold
            detections.append({
                "class_name": name,
                "track_id": int(track_id) if track_id is not None else None,
                "confidence": round(float(conf), 3),
                "bbox": [x1, y1, x2, y2],
            })
        return detections


def split_relevant(detections, relevant_classes):
    """
    Split ALL YOLO detections into (relevant, ignored).

    relevant -> goes on through the CV pipeline (each gets a 'category':
                person / vehicle / carried_object / other)
    ignored  -> kept in the frame record in case it is useful later,
                but not printed, drawn or counted as evidence.
    """
    relevant, ignored = [], []
    for det in detections:
        if det["class_name"] in relevant_classes:
            relevant.append({**det, "category": category_of(det["class_name"])})
        else:
            ignored.append(det)
    return relevant, ignored


def boxes_overlap(a, b):
    """True if two [x1, y1, x2, y2] boxes overlap at all."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def keep_carried_items(detections, carried_classes):
    """
    Drop any carried-item detection (bag, backpack, ...) that does not touch a
    person's box. This removes stray 'handbags' on the floor or in the background.
    People, vehicles and other classes are left untouched.
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
