"""
pipeline.py  (Stage 1 + 2 milestone)

    video -> OpenCV frames -> YOLO -> detections
          (bbox + confidence + camera_id + frame_number + timestamp)

Examples (run from inside the cv/ folder):
    python pipeline.py                                # cam_01, conf 0.50
    python pipeline.py --show --frame-skip 3 --conf 0.5
    python pipeline.py --camera cam_02 --show         # live window, press q to quit
    python pipeline.py --save-frames --frame-skip 5   # annotated JPGs in output/frames/
    python pipeline.py --no-detect --max-frames 100   # Stage 1 only (no YOLO needed)
    python pipeline.py --camera cam_03 --show --frame-skip 2   # tracking + reliable persons
    python pipeline.py --camera cam_03 --frame-skip 5 --save-crops --show   # + padded person crops
    python pipeline.py --all-classes                  # debugging: show every COCO class
    python pipeline.py --camera webcam_0 --source 0   # try a laptop webcam
"""
import argparse
import os

import cv2

from crops import PERSON_CROP_PADDING, crop_with_padding, save_crop
from config import CARRIED_OBJECT_CLASSES, RELEVANT_CLASSES, category_of
from tracker import MIN_PERSON_TRACK_FRAMES, PersonTrackFilter
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

DEFAULT_MODEL = "yolo26n.pt"   # n = fastest. Larger: yolo26s.pt / yolo26m.pt (more accurate, slower)
DEFAULT_CONF = 0.50   # one threshold for ALL classes; lower it explicitly with --conf

OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def label_of(det):
    """'person#17(0.94)' when tracked, 'handbag(0.67)' otherwise."""
    tid = f'#{det["track_id"]}' if det.get("track_id") is not None else ""
    return f'{det["class_name"]}{tid}({det["confidence"]:.2f})'


def draw_detections(frame, detections):
    """Return a copy of the frame with boxes and labels drawn on it."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        tid = f' ID:{det["track_id"]}' if det.get("track_id") is not None else ""
        label = f'{det["class_name"]}{tid} {det["confidence"]:.2f}'
        if det.get("crop_bbox"):   # thin yellow box = the padded person crop (--save-crops)
            cx1, cy1, cx2, cy2 = det["crop_bbox"]
            cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 255), 1)
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

    use_tracking = bool(detector) and not args.no_track
    if detector and detector.has_tracked:
        detector.reset_tracking()          # new camera -> fresh tracker, no id leakage
    person_filter = PersonTrackFilter(min_frames=args.min_track_frames)
    flagged = set()                        # track ids we already printed a warning for

    processed = 0
    for item in iter_frames(camera_id, source, frame_skip=args.frame_skip):
        frame = item["frame"]
        detections, ignored = [], []
        if detector:
            from detector import keep_carried_items, split_relevant  # lazy: needs ultralytics
            all_detections = detector.detect(frame, track=use_tracking)   # everything YOLO found
            if args.all_classes:                              # debug: treat everything as relevant
                detections = [{**d, "category": category_of(d["class_name"])} for d in all_detections]
            else:
                detections, ignored = split_relevant(all_detections, detector.relevant_classes)

            # Step 2: only RELIABLE persons continue (persistence + shadow check)
            if use_tracking:
                persons = [d for d in detections if d["category"] == "person"]
                others = [d for d in detections if d["category"] != "person"]
                reliable, held_back = person_filter.update(item["timestamp"], persons)
                for d in held_back:
                    ignored.append({**d, "ignored_reason": d["held_back_reason"]})
                    if "reflection" in d["held_back_reason"] and d["track_id"] not in flagged:
                        flagged.add(d["track_id"])
                        print(f'[{camera_id}] frame {item["frame_number"]:>6}  '
                              f'held back person#{d["track_id"]}({d["confidence"]:.2f}): '
                              f'{d["held_back_reason"]}')
                detections = reliable + others
            if not args.loose_bags:
                detections = keep_carried_items(detections, CARRIED_OBJECT_CLASSES)

        # Step 3: padded crop of every RELIABLE person (vehicles/objects: not yet)
        if args.save_crops:
            crops_dir = os.path.join(OUTPUT_DIR, "crops", camera_id)
            for det in detections:
                if det.get("category") != "person":
                    continue
                crop, box = crop_with_padding(frame, det["bbox"], args.crop_padding)
                det["crop_bbox"] = box                      # padded box, full-frame coordinates
                if crop is not None:
                    path = save_crop(crop, crops_dir, det["class_name"],
                                     det["track_id"], item["frame_number"])
                    det["crop_path"] = os.path.relpath(path, BASE_DIR).replace(os.sep, "/")

        record = {
            "camera_id": item["camera_id"],
            "frame_number": item["frame_number"],
            "timestamp": item["timestamp"],
            "detections": detections,        # relevant only
            "ignored_detections": ignored,   # kept in memory, not shown
        }
        records.append(record)
        processed += 1

        # ---- progress output ----
        summary = ", ".join(label_of(d) for d in detections)
        print(f'[{camera_id}] frame {item["frame_number"]:>6} '
              f't={format_time(item["timestamp"])}  '
              f'{len(detections)} objects  {summary}'
              + (f'   [{len(ignored)} ignored]' if ignored else ''))

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
    parser.add_argument("--classes", nargs="*", default=None,
                        help="override the relevant-class list from config.py, "
                             "e.g. --classes person car handbag")
    parser.add_argument("--all-classes", action="store_true",
                        help="debug: show EVERY class YOLO finds (chair, tv, ...)")
    parser.add_argument("--bag-conf", type=float, default=None,
                        help="OPTIONAL separate threshold for bag classes. "
                             "Default: bags use the same threshold as --conf")
    parser.add_argument("--loose-bags", action="store_true",
                        help="also keep bags that are NOT touching a person")
    parser.add_argument("--min-track-frames", type=int, default=MIN_PERSON_TRACK_FRAMES,
                        help="a person must be tracked in this many processed frames "
                             "to count as reliable (default 3)")
    parser.add_argument("--save-crops", action="store_true",
                        help="save a padded crop of every reliable person to output/crops/<camera>/")
    parser.add_argument("--crop-padding", type=float, default=PERSON_CROP_PADDING,
                        help="padding around the person box as a fraction of its size (default 0.15)")
    parser.add_argument("--no-track", action="store_true",
                        help="debug: plain detection without tracking / person filtering")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="process every Nth frame (videos are ~60fps, so 5 is a good speed-up)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N processed frames (0 = all)")
    parser.add_argument("--show", action="store_true", help="show a live window (press q to quit)")
    parser.add_argument("--save-frames", action="store_true", help="save annotated frames to output/frames/")
    parser.add_argument("--no-detect", action="store_true", help="Stage 1 only: read frames, skip YOLO")
    args = parser.parse_args()
    if not 0.0 <= args.crop_padding <= 1.0:
        parser.error("--crop-padding must be between 0 and 1")
    for name in ("conf", "bag_conf"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")

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
        relevant = args.classes if args.classes else RELEVANT_CLASSES
        # Per-class override only if the user explicitly asked for one
        bag_conf = ({name: args.bag_conf for name in CARRIED_OBJECT_CLASSES}
                    if args.bag_conf is not None else None)
        detector = Detector(args.model, conf_threshold=args.conf,
                            relevant_classes=relevant, class_conf=bag_conf)
        print(f"Model: {args.model}   confidence threshold: {args.conf:.2f}"
              + (f"   (bags: {args.bag_conf:.2f})" if bag_conf else "   (all relevant classes)"))

    for camera_id, source in jobs:
        try:
            process_camera(camera_id, source, detector, args)
        except RuntimeError as err:  # e.g. video could not be opened
            print(f"ERROR: {err}")


if __name__ == "__main__":
    main()
