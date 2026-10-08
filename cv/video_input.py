"""
video_input.py  (Stage 1)

Reads frames from a video source and attaches camera_id, frame_number
and timestamp to every frame.

A "source" is either:
    - a file path   -> "videos/cam_01.mp4"
    - a webcam index -> 0, 1, 2   (so switching from MP4 to webcams later
                                    only means changing the CAMERAS dict
                                    in pipeline.py)
"""
import time

import cv2


def open_source(source):
    """Open a video file or webcam. Returns an opened cv2.VideoCapture,
    or raises RuntimeError if it cannot be opened."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source!r}")
    return cap


def iter_frames(camera_id, source, frame_skip=1):
    """
    Generator that yields one dict per frame:

        {
            "camera_id":    "cam_01",
            "frame_number": 123,
            "timestamp":    2.05,        # seconds since start of video
            "frame":        <numpy BGR image>
        }

    frame_skip=1 -> every frame, 3 -> every 3rd frame, etc.
    (frame_number and timestamp always refer to the ORIGINAL video,
     so skipping frames does not change timestamps.)
    """
    cap = open_source(source)
    is_webcam = isinstance(source, int)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # 0, negative or NaN
        fps = 30.0
        print(f"[{camera_id}] WARNING: could not read FPS, assuming {fps}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{camera_id}] opened {source!r}  fps={fps:.2f}  "
          f"frames={total if total > 0 else 'unknown (live)'}")

    start_time = time.time()
    frame_number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of video (or webcam disconnected)

            # Skip empty / invalid frames instead of crashing
            if frame is None or frame.size == 0:
                print(f"[{camera_id}] WARNING: empty frame at {frame_number}, skipping")
                frame_number += 1
                continue

            if frame_number % frame_skip == 0:
                if is_webcam:
                    # Webcams have no reliable frame counter / FPS,
                    # so use wall-clock time since we started.
                    timestamp = time.time() - start_time
                else:
                    timestamp = frame_number / fps

                yield {
                    "camera_id": camera_id,
                    "frame_number": frame_number,
                    "timestamp": round(timestamp, 3),
                    "frame": frame,
                }
            frame_number += 1
    finally:
        cap.release()
