from typing import Tuple, List
from vision.geometry import clamp01

def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_inside_box(px, py, box):
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)

def person_at_height_risk(person_conf: float, floor_distance_px: float, floor_score: float, min_floor_score: float = 0.35) -> Tuple[bool, float, list]:
    """
    Height/edge hazard:
    We only care if floor_score is meaningful.
    """

    floor_score = clamp01(floor_score)
    if floor_score < min_floor_score:
        prob = 0.0
    else:
        prob = clamp01(person_conf * floor_score)

    triggered = prob >= 0.45
    evidence = [
        f"person_conf={person_conf:.2f}",
        f"floor_distance_px={floor_distance_px:.1f}",
        f"floor_score={floor_score:.2f}",
        f"risk_probability={prob:.2f}"
    ]
    return triggered, prob, evidence

def person_person_proximity_risk(dist_norm: float, conf_a: float, conf_b: float,
                                 near_thresh_norm: float = 2.0) -> Tuple[bool, float, List[str]]:
    """
    Being close is not always a risk.
    So we trigger only if they are VERY close (near_thresh_norm small).
    """

    proximity_score = 1.0 - (dist_norm / (near_thresh_norm + 1e-6))
    proximity_score = clamp01(proximity_score)

    conf_term = clamp01((conf_a + conf_b) / 2.0)

    prob = clamp01(proximity_score * conf_term * 0.40)

    triggered = prob >= 0.45

    evidence = [
        f"dist_norm={dist_norm:.2f}",
        f"near_thresh_norm={near_thresh_norm:.2f}",
        f"proximity_score={proximity_score:.2f}",
        f"conf_avg={conf_term:.2f}",
        f"risk_probability={prob:.2f}"
    ]

    return triggered, prob, evidence

def person_near_machine_risk(dist_norm: float, person_conf: float, machine_conf: float,
                             near_thresh_norm: float = 1.4) -> Tuple[bool, float, List[str]]:
    """
    Machine/vehicle proximity = higher risk.
    We trigger only if person is truly close.
    """

    proximity_score = 1.0 - (dist_norm / (near_thresh_norm + 1e-6))
    proximity_score = clamp01(proximity_score)

    conf_term = clamp01((person_conf + machine_conf) / 2.0)

    prob = clamp01(proximity_score * conf_term)
    triggered = prob >= 0.45

    evidence = [
        f"dist_norm={dist_norm:.2f}",
        f"near_thresh_norm={near_thresh_norm:.2f}",
        f"proximity_score={proximity_score:.2f}",
        f"conf_avg={conf_term:.2f}",
        f"risk_probability={prob:.2f}",
    ]

    return triggered, prob, evidence

def helmet_overlap_score(person_bbox, helmet_bbox) -> float:
    """
    Computes overlap ratio between helmet box and top region of person box.
    Returns 0..1.
    """
    px1, py1, px2, py2 = person_bbox
    hx1, hy1, hx2, hy2 = helmet_bbox

    # top 35% region of person bbox = head area approx
    head_y2 = py1 + 0.35 * (py2 - py1)

    # head box region
    head_box = [px1, py1, px2, head_y2]

    ax1, ay1, ax2, ay2 = head_box
    bx1, by1, bx2, by2 = helmet_bbox

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    helmet_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1)) + 1e-6
    overlap = clamp01(inter_area / helmet_area)
    cx, cy = _bbox_center(helmet_bbox)
    if overlap < 0.05 and _point_inside_box(cx, cy, head_box):
        overlap = max(overlap, 0.35)

    return clamp01(overlap)


def person_no_helmet_risk(person_conf: float, best_helmet_overlap: float, best_helmet_conf: float):
    prob = _ppe_missing_prob(person_conf, best_helmet_overlap, best_helmet_conf, importance=1.0)
    triggered = prob >= 0.45
    evidence = [
        f"person_conf={person_conf:.2f}",
        f"best_helmet_overlap={best_helmet_overlap:.2f}",
        f"best_helmet_conf={best_helmet_conf:.2f}",
        f"missing_score={1.0 - max(best_helmet_overlap, best_helmet_conf):.2f}",
        f"risk_probability={prob:.2f}",
    ]
    return triggered, prob, evidence

def person_height_no_helmet_risk(p_height: float, p_no_helmet: float) -> tuple[bool, float, list]:
    """
    Combined contextual risk:
    Person at height AND no helmet.
    Using AND-style probability.
    """
    prob = clamp01(p_height * p_no_helmet)
    triggered = prob >= 0.45

    evidence = [
        f"p_height={p_height:.3f}",
        f"p_no_helmet={p_no_helmet:.3f}",
        f"combined_probability={prob:.3f}"
    ]
    return triggered, prob, evidence

def vest_overlap_score(person_bbox, vest_bbox) -> float:
    """
    Vest match score between vest bbox and person's torso region.
    Returns 0..1.

    Improvement:
    - Fallback: vest CENTER inside torso region => accept partial score.
    """
    px1, py1, px2, py2 = person_bbox
    vx1, vy1, vx2, vy2 = vest_bbox

    person_h = py2 - py1
    torso_y1 = py1 + 0.30 * person_h
    torso_y2 = py1 + 0.85 * person_h

    torso_box = [px1, torso_y1, px2, torso_y2]

    ax1, ay1, ax2, ay2 = torso_box
    bx1, by1, bx2, by2 = vest_bbox

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    vest_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1)) + 1e-6

    overlap = clamp01(inter_area / vest_area)

    cx, cy = _bbox_center(vest_bbox)
    if overlap < 0.05 and _point_inside_box(cx, cy, torso_box):
        overlap = max(overlap, 0.35)

    return clamp01(overlap)

def person_no_vest_risk(person_conf: float, best_vest_overlap: float, best_vest_conf: float):
    prob = _ppe_missing_prob(person_conf, best_vest_overlap, best_vest_conf, importance=0.6)
    triggered = prob >= 0.50
    evidence = [
        f"person_conf={person_conf:.2f}",
        f"best_vest_overlap={best_vest_overlap:.2f}",
        f"best_vest_conf={best_vest_conf:.2f}",
        f"missing_score={1.0 - max(best_vest_overlap, best_vest_conf):.2f}",
        f"risk_probability={prob:.2f}",
    ]
    return triggered, prob, evidence

def mask_overlap_score(person_bbox, mask_bbox) -> float:
    """
    Mask match score between mask bbox and person's face region.
    Returns 0..1.

    Improvement:
    - Fallback: mask CENTER inside face region => accept partial score.
    """
    px1, py1, px2, py2 = person_bbox
    mx1, my1, mx2, my2 = mask_bbox

    person_h = py2 - py1
    face_y1 = py1 + 0.10 * person_h
    face_y2 = py1 + 0.45 * person_h

    face_box = [px1, face_y1, px2, face_y2]

    ax1, ay1, ax2, ay2 = face_box
    bx1, by1, bx2, by2 = mask_bbox

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    mask_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1)) + 1e-6

    overlap = clamp01(inter_area / mask_area)

    cx, cy = _bbox_center(mask_bbox)
    if overlap < 0.05 and _point_inside_box(cx, cy, face_box):
        overlap = max(overlap, 0.35)

    return clamp01(overlap)

def person_no_mask_risk(person_conf: float, best_mask_overlap: float, best_mask_conf: float):
    # Mask should not dominate risk
    prob = _ppe_missing_prob(person_conf, best_mask_overlap, best_mask_conf, importance=0.3)
    triggered = prob >= 0.55
    evidence = [
        f"person_conf={person_conf:.2f}",
        f"best_mask_overlap={best_mask_overlap:.2f}",
        f"best_mask_conf={best_mask_conf:.2f}",
        f"missing_score={1.0 - max(best_mask_overlap, best_mask_conf):.2f}",
        f"risk_probability={prob:.2f}",
    ]
    return triggered, prob, evidence

def person_machine_close_context_risk(person_near_machine_prob: float, dist_norm: float,
                                     danger_dist_norm: float = 0.55):
    """
    Extra boost ONLY when very close.
    dist_norm is normalized distance (not px).
    """

    danger_score = clamp01(1.0 - (dist_norm / (danger_dist_norm + 1e-6)))

    prob = clamp01(person_near_machine_prob * (0.6 + 0.4 * danger_score))

    triggered = prob >= 0.55

    evidence = [
        f"dist_px={dist_norm:.2f}",
        f"danger_dist_px={danger_dist_norm:.2f}",
        f"danger_score={danger_score:.2f}",
        f"base_person_near_machine_prob={person_near_machine_prob:.3f}",
        f"risk_probability={prob:.3f}"
    ]

    return triggered, prob, evidence

def _ppe_missing_prob(
    person_conf: float,
    best_overlap: float,
    best_conf: float,
    importance: float
) -> float:
    """
    importance:
      helmet -> 1.0 (critical)
      vest   -> 0.6 (medium)
      mask   -> 0.3 (low)
    """
    presence_score = clamp01(max(best_overlap, best_conf))
    missing_score = clamp01(1.0 - presence_score)

    prob = clamp01(person_conf * missing_score * importance)

    return prob