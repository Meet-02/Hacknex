"""
detector.py  (Stage 2)

Thin wrapper around Ultralytics YOLO.

    detector = Detector(model_path="yolov8n.pt", conf_threshold=0.4)
    detections = detector.detect(frame)

Each detection:
    {"class_name": "person", "confidence": 0.92, "bbox": [x1, y1, x2, y2]}
"""
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4,
                 class_names=None, device=None):
        """
        model_path     : any Ultralytics weights file (yolov8n.pt, yolo11s.pt, ...).
                         Downloaded automatically the first time.
        conf_threshold : detections below this confidence are dropped.
        class_names    : optional list such as ["person", "car", "backpack"].
                         None = keep every class the model knows (80 COCO classes).
        device         : None = auto, or "cpu", "cuda:0", "mps".
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

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
            conf=self.conf_threshold,
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
            detections.append({
                "class_name": self.names[int(cls_id)],
                "confidence": round(float(conf), 3),
                "bbox": [x1, y1, x2, y2],
            })
        return detections
