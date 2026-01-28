import math

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_height(bbox):
    x1, y1, x2, y2 = bbox
    return abs(y2 - y1)

def bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return abs(x2 - x1)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def edge_proximity_score(person_bbox, image_height, edge_band_ratio=0.20):
    """
    Score in [0,1] indicating how close the person is to the bottom edge band.
    If a person's bbox bottom (y2) lies inside the bottom band => score increases.
    """
    _, _, _, y2 = person_bbox
    band_top = image_height * (1.0 - edge_band_ratio)

    if y2 < band_top:
        return 0.0

    score = (y2 - band_top) / (image_height - band_top + 1e-6)
    return clamp01(score)

def pairwise_pixel_distance(bbox_a, bbox_b):
    ax, ay = bbox_center(bbox_a)
    bx, by = bbox_center(bbox_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def compute_scene_floor_y(person_bboxes, default_floor_y):
    """
    Estimate scene 'floor/work-surface' as the maximum y2 among persons.
    """
    if not person_bboxes:
        return default_floor_y
    return max(b[3] for b in person_bboxes)  

def distance_to_floor_score(person_bbox, floor_y, max_allowed=250.0):
    """
    If feet are close to the estimated floor line => score high.
    If far above => score low.
    """
    _, _, _, y2 = person_bbox
    dist = abs(floor_y - y2)

    score = 1.0 - (dist / (max_allowed + 1e-6))
    return clamp01(score), dist