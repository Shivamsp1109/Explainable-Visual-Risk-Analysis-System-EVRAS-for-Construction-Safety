from typing import Tuple, List
from vision.geometry import clamp01

def person_at_height_risk(person_conf: float, floor_distance_px: float, floor_score: float) -> Tuple[bool, float, list]:
    """
    We treat: person standing on a narrow structure / elevated work surface
    as a potential 'height/edge' indicator.
    """

    base = 0.05
    prob = clamp01(base + 0.95 * floor_score) * clamp01(person_conf)

    triggered = prob >= 0.35
    evidence = [
        f"person_conf={person_conf:.2f}",
        f"floor_distance_px={floor_distance_px:.1f}",
        f"floor_score={floor_score:.2f}",
        f"risk_probability={prob:.2f}"
    ]
    return triggered, prob, evidence

def person_person_proximity_risk(dist_px: float, conf_a: float, conf_b: float,
                                 near_thresh_px: float = 350.0) -> Tuple[bool, float, List[str]]:
    """
    Risk indicator: two persons are working too close.
    dist_px: distance between person centers in pixels.
    """

    proximity_score = 1.0 - (dist_px / (near_thresh_px + 1e-6))
    proximity_score = clamp01(proximity_score)

    conf_term = clamp01((conf_a + conf_b) / 2.0)

    prob = clamp01(proximity_score * conf_term)

    triggered = prob >= 0.35

    evidence = [
        f"dist_px={dist_px:.1f}",
        f"near_thresh_px={near_thresh_px:.1f}",
        f"proximity_score={proximity_score:.2f}",
        f"conf_avg={conf_term:.2f}",
        f"risk_probability={prob:.2f}"
    ]

    return triggered, prob, evidence