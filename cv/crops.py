"""
crops.py  (Step 3)

Cut a person out of the frame with some padding around the bounding box.

The padding matters: a handbag hanging from the hand can stick out of the
person's box, so we include a margin (default 15% of the box width/height).

crop_with_padding() also returns the padded box in FULL-FRAME coordinates.
Step 5 needs it: a bag found inside the crop at (x, y) is at
(x + box_x1, y + box_y1) in the original frame.
"""
import os

import cv2

PERSON_CROP_PADDING = 0.15


def padded_box(bbox, padding, frame_width, frame_height):
    """Grow [x1, y1, x2, y2] by `padding` (a fraction of its own width/height)
    on every side, then clamp it to the frame."""
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return [
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(frame_width, int(round(x2 + pad_x))),
        min(frame_height, int(round(y2 + pad_y))),
    ]


def crop_with_padding(frame, bbox, padding=PERSON_CROP_PADDING):
    """Returns (crop_image, padded_box). crop_image is None if the box is empty."""
    frame_height, frame_width = frame.shape[:2]
    box = padded_box(bbox, padding, frame_width, frame_height)
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None, box
    return frame[y1:y2, x1:x2].copy(), box


def save_crop(crop, folder, class_name, track_id, frame_number):
    """Save e.g. output/crops/cam_03/person_track_1_00200.jpg and return its path."""
    os.makedirs(folder, exist_ok=True)
    tid = track_id if track_id is not None else "none"
    path = os.path.join(folder, f"{class_name}_track_{tid}_{frame_number:05d}.jpg")
    cv2.imwrite(path, crop)
    return path
