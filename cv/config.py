"""
config.py

The project's "what do we care about?" lists. No heavy imports here, so any
file (or a future teammate's script) can import it.

Background objects (chair, tv, bottle, potted plant, ...) are simply NOT in
these lists. We never name them; anything not listed is "irrelevant".
"""

PERSON_CLASSES = {"person"}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# Things a person might carry. First-pass detections of these are only
# *candidates*; they are tied to a person in a later step.
#
# "umbrella" is deliberately NOT in the default list: in the test videos the
# small nano model keeps labelling a dangling black bag as "umbrella", and a
# wrong label is worse than no label. To switch it back on for scenes where
# umbrellas really matter (rain, outdoor cameras), add "umbrella" to this set.
CARRIED_OBJECT_CLASSES = {"handbag", "backpack", "suitcase"}

RELEVANT_CLASSES = PERSON_CLASSES | VEHICLE_CLASSES | CARRIED_OBJECT_CLASSES


def category_of(class_name):
    """'person' | 'vehicle' | 'carried_object' | 'other'"""
    if class_name in PERSON_CLASSES:
        return "person"
    if class_name in VEHICLE_CLASSES:
        return "vehicle"
    if class_name in CARRIED_OBJECT_CLASSES:
        return "carried_object"
    return "other"
