"""
pipeline.py  (Stage 1 + 2 milestone)

    video -> OpenCV frames -> YOLO -> detections
          (bbox + confidence + camera_id + frame_number + timestamp)

Examples (run from inside the cv/ folder):
    python pipeline.py                                # cam_01, prints detections
    python pipeline.py --camera cam_02 --show         # live window, press q to quit
    python pipeline.py --save-frames --frame-skip 5   # annotated JPGs in output/frames/
    python pipeline.py --no-detect --max-frames 100   # Stage 1 only (no YOLO needed)
    python pipeline.py --all-classes                  # debugging: show every COCO class
    python pipeline.py --camera webcam_0 --source 0   # try a laptop webcam
"""
import argparse
import os

import cv2

from video_input import iter_frames

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# CAMERA CONFIG  <- the ONLY place to change when moving from MP4 to webcams.
#   MP4 file : "videos/cam_01.mp4"
#   Webcam   : 0, 1, 2  (integers = OpenCV device index)
# ---------------------------------------------------------------------------
CAMERAS = {
    "cam_01": os.path.join(BASE_DIR, "videos", "cam_01.mp4"),
    "cam_02": os.path.join(BASE_DIR, "videos", "cam_02.mp4"),
    "cam_03": os.path.join(BASE_DIR, "videos", "cam_03.mp4"),
    # Later, e.g.:  "cam_01": 0,  "cam_02": 1,  "cam_03": 2,
}

DEFAULT_MODEL = "yolo26s.pt"   # newest Ultralytics family; n=fastest, s=balanced, m/l/x=more accurate
DEFAULT_CONF = 0.4

# Only these COCO classes are kept. Everything else (chair, tv, laptop, umbrella,
# potted plant ... e.g. the water cooler in the background) is ignored.
PERSON_CLASSES = ["person"]
VEHICLE_CLASSES = ["bicycle", "motorcycle", "car", "bus", "truck"]
CARRIED_CLASSES = ["backpack", "handbag", "suitcase"]   # what a person can carry
DEFAULT_CLASSES = PERSON_CLASSES + VEHICLE_CLASSES + CARRIED_CLASSES
DEFAULT_BAG_CONF = 0.2   # bags score lower than people, so they get a lower threshold
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def draw_detections(frame, detections):
    """Return a copy of the frame with boxes and labels drawn on it."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return annotated


def format_time(seconds):
    """4.1 -> '00:00:04.100'"""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def process_camera(camera_id, source, detector, args):
    """Run Stage 1 + 2 on a single camera. Returns the list of per-frame records."""
    records = []
    frames_dir = os.path.join(OUTPUT_DIR, "frames", camera_id)
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    processed = 0
    for item in iter_frames(camera_id, source, frame_skip=args.frame_skip):
        frame = item["frame"]
        detections = detector.detect(frame) if detector else []
        if detector and not args.loose_bags:
            from detector import keep_carried_items  # needs ultralytics, so import lazily
            detections = keep_carried_items(detections, CARRIED_CLASSES)

        record = {
            "camera_id": item["camera_id"],
            "frame_number": item["frame_number"],
            "timestamp": item["timestamp"],
            "detections": detections,
        }
        records.append(record)
        processed += 1

        # ---- progress output ----
        summary = ", ".join(f'{d["class_name"]}({d["confidence"]:.2f})' for d in detections)
        print(f'[{camera_id}] frame {item["frame_number"]:>6} '
              f't={format_time(item["timestamp"])}  '
              f'{len(detections)} objects  {summary}')

        # ---- optional visual output ----
        if args.save_frames or args.show:
            annotated = draw_detections(frame, detections)
            if args.save_frames:
                path = os.path.join(frames_dir, f'{camera_id}_{item["frame_number"]:05d}.jpg')
                cv2.imwrite(path, annotated)
            if args.show:
                cv2.imshow(camera_id, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit requested.")
                    break

        if args.max_frames and processed >= args.max_frames:
            print(f"[{camera_id}] reached --max-frames={args.max_frames}")
            break

    if args.show:
        cv2.destroyAllWindows()

    total_objects = sum(len(r["detections"]) for r in records)
    print(f"[{camera_id}] done: {processed} frames processed, {total_objects} detections")
    return records


def main():
    parser = argparse.ArgumentParser(description="CV pipeline - Stage 1 + 2")
    parser.add_argument("--camera", default="cam_01", choices=list(CAMERAS) + ["all"],
                        help="which camera from CAMERAS to process (default: cam_01)")
    parser.add_argument("--source", default=None,
                        help="override the source: a file path, or a webcam index like 0")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights file")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES,
                        help="classes to keep (default: person + vehicles + bags)")
    parser.add_argument("--all-classes", action="store_true",
                        help="keep all 80 COCO classes (chair, tv, ... too)")
    parser.add_argument("--bag-conf", type=float, default=DEFAULT_BAG_CONF,
                        help="confidence threshold for bag classes (default 0.2)")
    parser.add_argument("--loose-bags", action="store_true",
                        help="also keep bags that are NOT touching a person")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="process every Nth frame (videos are ~60fps, so 5 is a good speed-up)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0 = all)")
    parser.add_argument("--show", action="store_true", help="show a live window (press q to quit)")
    parser.add_argument("--save-frames", action="store_true", help="save annotated frames to output/frames/")
    parser.add_argument("--no-detect", action="store_true", help="Stage 1 only: read frames, skip YOLO")
    args = parser.parse_args()

    # Build the list of (camera_id, source) to process
    if args.camera == "all":
        jobs = list(CAMERAS.items())
    else:
        source = CAMERAS[args.camera]
        if args.source is not None:
            # "0" -> webcam 0, anything else -> file path
            source = int(args.source) if args.source.isdigit() else args.source
        jobs = [(args.camera, source)]

    detector = None
    if not args.no_detect:
        from detector import Detector  # imported here so --no-detect works without ultralytics
        classes = None if args.all_classes else args.classes
        bag_conf = {name: args.bag_conf for name in CARRIED_CLASSES}
        detector = Detector(args.model, conf_threshold=args.conf,
                            class_names=classes, class_conf=bag_conf)

    for camera_id, source in jobs:
        try:
            process_camera(camera_id, source, detector, args)
        except RuntimeError as err:  # e.g. video could not be opened
            print(f"ERROR: {err}")


if __name__ == "__main__":
    main()
