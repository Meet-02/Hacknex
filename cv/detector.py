"""
detector.py  (Stage 2)

Thin wrapper around Ultralytics YOLO.

    detector = Detector(model_path="yolo26s.pt", conf_threshold=0.4)
    detections = detector.detect(frame)

Each detection:
    {"class_name": "person", "confidence": 0.92, "bbox": [x1, y1, x2, y2]}
"""
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolo26s.pt", conf_threshold=0.4,
                 class_names=None, device=None, class_conf=None):
        """
        model_path     : any Ultralytics weights file (yolo26s.pt, yolo11s.pt, yolov8n.pt, ...).
                         Downloaded automatically the first time.
        conf_threshold : detections below this confidence are dropped.
        class_names    : optional list such as ["person", "car", "backpack"].
                         None = keep every class the model knows (80 COCO classes).
        device         : None = auto, or "cpu", "cuda:0", "mps".
        class_conf     : optional per-class thresholds that override conf_threshold,
                         e.g. {"handbag": 0.2}. Useful because bags score lower than people.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_conf = class_conf or {}
        self.device = device

        # YOLO is run at the LOWEST threshold; stricter per-class limits are applied after.
        self.min_conf = min([conf_threshold] + list(self.class_conf.values()))

        # model.names looks like {0: "person", 1: "bicycle", 2: "car", ...}
        self.names = self.model.names

        # Convert class names to the numeric ids YOLO expects
        self.class_ids = None
        if class_names:
            name_to_id = {name: i for i, name in self.names.items()}
            unknown = [n for n in class_names if n not in name_to_id]
            if unknown:
                raise ValueError(f"Unknown class names for this model: {unknown}")
            self.class_ids = [name_to_id[n] for n in class_names]

    def detect(self, frame):
        """Run YOLO on one frame. Returns a list of detection dicts
        (empty list if nothing was found). Handles many objects per frame."""
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            frame,
            conf=self.min_conf,
            classes=self.class_ids,
            device=self.device,
            verbose=False,  # silence Ultralytics' per-frame printing
        )

        boxes = results[0].boxes
        detections = []
        for xyxy, conf, cls_id in zip(boxes.xyxy.cpu().numpy(),
                                      boxes.conf.cpu().numpy(),
                                      boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            name = self.names[int(cls_id)]
            if float(conf) < self.class_conf.get(name, self.conf_threshold):
                continue  # below this class's own threshold
            detections.append({
                "class_name": name,
                "confidence": round(float(conf), 3),
                "bbox": [x1, y1, x2, y2],
            })
        return detections


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
