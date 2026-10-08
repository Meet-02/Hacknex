"""
tracker.py  (Step 2)

Temporal filtering of PERSON tracks.

The actual tracking (giving each object a stable track_id) is done by
Ultralytics' built-in ByteTrack inside Detector.detect(frame, track=True).
This file decides which tracked persons are RELIABLE.

Why not just "seen in >= 3 frames"?
    A reflection in a glossy wall can be detected as a 'person' for many
    frames in a row, with a confidence well above the threshold. So a person
    track is reliable only if:

      1. PERSISTENCE : it has been seen in at least `min_frames` processed frames
                       (or it is very confident: >= `strong_conf`), AND
      2. NOT A SHADOW: it does not look like a weaker, younger copy of an
                       established person standing right next to it
                       (same height range, side by side, lower confidence,
                       short-lived). If it keeps living for
                       `shadow_max_seconds`, it is accepted as a real person
                       (e.g. someone walking beside the first person).

Everything held back is NOT deleted: the pipeline keeps it in
`ignored_detections` together with the reason.
"""

# ---- defaults (all configurable from the Detector/pipeline side) ----------
MIN_PERSON_TRACK_FRAMES = 20     # processed frames a person must be seen in
PERSON_STRONG_CONF = 0.85       # this confident = reliable even if brand new
SHADOW_MAX_SECONDS = 2.0        # a "shadow" that lives longer than this is a real person
ESTABLISHED_FRAMES = 5          # the "main" person must have been seen this many frames
SHADOW_CONF_MARGIN = 0.10       # shadow must be this much weaker than the main person (mean conf)
SHADOW_NEAR_FACTOR = 2.0        # shadow centre within this many person-widths of the main person
SHADOW_HEIGHT_RATIO = (0.5, 1.5)  # shadow height / main person height
SHADOW_MIN_V_OVERLAP = 0.3      # vertical overlap (fraction of the smaller box height)


class _TrackStats:
    """What we remember about one track_id."""

    def __init__(self, first_ts):
        self.first_ts = first_ts
        self.last_ts = first_ts
        self.hits = 0
        self.conf_sum = 0.0
        self.shadow_of = None     # track_id of the person this looks like a shadow of
        self.promoted = False     # lived long enough -> never flagged again

    @property
    def mean_conf(self):
        return self.conf_sum / self.hits if self.hits else 0.0


def _box_w_h(bbox):
    return max(bbox[2] - bbox[0], 1), max(bbox[3] - bbox[1], 1)


class PersonTrackFilter:
    """One instance per camera (track ids are only meaningful inside one camera)."""

    def __init__(self, min_frames=MIN_PERSON_TRACK_FRAMES,
                 strong_conf=PERSON_STRONG_CONF,
                 shadow_max_seconds=SHADOW_MAX_SECONDS):
        self.min_frames = min_frames
        self.strong_conf = strong_conf
        self.shadow_max_seconds = shadow_max_seconds
        self.tracks = {}   # track_id -> _TrackStats

    # ------------------------------------------------------------------
    def update(self, timestamp, person_detections):
        """
        Call once per processed frame with that frame's PERSON detections
        (each must have 'track_id', 'confidence', 'bbox').

        Returns (reliable, held_back): two lists of detection dicts.
        Each returned dict has extra keys: 'track_hits', and for held-back
        ones 'held_back_reason'.
        """
        # 1) update per-track statistics
        for det in person_detections:
            tid = det.get("track_id")
            if tid is None:
                continue
            stats = self.tracks.setdefault(tid, _TrackStats(timestamp))
            stats.hits += 1
            stats.conf_sum += det["confidence"]
            stats.last_ts = timestamp

        # 2) shadow check (needs to look at the other people in this frame)
        for det in person_detections:
            tid = det.get("track_id")
            if tid is None:
                continue
            stats = self.tracks[tid]
            if timestamp - stats.first_ts >= self.shadow_max_seconds:
                stats.promoted = True
                stats.shadow_of = None
            if stats.promoted or stats.shadow_of is not None:
                continue
            for other in person_detections:
                other_id = other.get("track_id")
                if other_id is None or other_id == tid:
                    continue
                if self._looks_like_shadow(det, stats, other, self.tracks[other_id]):
                    stats.shadow_of = other_id
                    break

        # 3) final decision per detection
        reliable, held_back = [], []
        for det in person_detections:
            tid = det.get("track_id")
            out = dict(det)
            if tid is None:
                out["held_back_reason"] = "no track id"
                held_back.append(out)
                continue
            stats = self.tracks[tid]
            out["track_hits"] = stats.hits
            persistent = stats.hits >= self.min_frames or det["confidence"] >= self.strong_conf
            if stats.shadow_of is not None:
                out["held_back_reason"] = f"likely reflection/shadow of person #{stats.shadow_of}"
                held_back.append(out)
            elif not persistent:
                out["held_back_reason"] = f"pending ({stats.hits}/{self.min_frames} frames)"
                held_back.append(out)
            else:
                reliable.append(out)
        return reliable, held_back

    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_shadow(det, stats, other, other_stats):
        """Is `det` (a young track) a weaker copy standing next to `other`?"""
        # `other` must be an established, older and clearly stronger person
        if other_stats.hits < ESTABLISHED_FRAMES:
            return False
        if other_stats.first_ts >= stats.first_ts:
            return False
        if other_stats.mean_conf < stats.mean_conf + SHADOW_CONF_MARGIN:
            return False

        wa, ha = _box_w_h(other["bbox"])
        wb, hb = _box_w_h(det["bbox"])
        if not SHADOW_HEIGHT_RATIO[0] <= hb / ha <= SHADOW_HEIGHT_RATIO[1]:
            return False

        # side by side (or overlapping): centres close, measured in person-widths
        cxa = (other["bbox"][0] + other["bbox"][2]) / 2
        cxb = (det["bbox"][0] + det["bbox"][2]) / 2
        if abs(cxa - cxb) > SHADOW_NEAR_FACTOR * wa:
            return False

        # similar vertical position
        top = max(other["bbox"][1], det["bbox"][1])
        bottom = min(other["bbox"][3], det["bbox"][3])
        v_overlap = max(0, bottom - top) / min(ha, hb)
        return v_overlap >= SHADOW_MIN_V_OVERLAP
