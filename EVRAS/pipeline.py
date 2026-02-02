import json
import os
from typing import Optional, Tuple

import cv2

from vision.detector import ObjectDetector
from vision.geometry import compute_scene_floor_y, distance_to_floor_score, pairwise_pixel_distance, clamp01

from risk.rules import (
    person_at_height_risk,
    person_height_no_helmet_risk,
    person_person_proximity_risk,
    person_near_machine_risk,
    helmet_overlap_score,
    person_no_helmet_risk,
    vest_overlap_score,
    person_no_vest_risk,
    mask_overlap_score,
    person_no_mask_risk,
    person_machine_close_context_risk
)

from risk.risk_graph import ObjectNode, RelationEdge, RiskFactor, build_risk_graph
from llm.explain import explain_risk_with_ollama


IMG_SIZE = 320

# Detection + filtering settings (tuned for YOLO PPE)
DETECT_CONF_THRESH = 0.35
PERSON_MIN_SIDE = 18
PERSON_MIN_AREA = 750
PPE_MIN_SIDE = 6
PPE_MIN_AREA = 60
MACHINE_MIN_SIDE = 16
MACHINE_MIN_AREA = 900

# Balanced settings (recommended for website deployment)
PERSON_MIN_CONF_RAW = 0.40      # ignore person below this
PERSON_CONF_BOOST = 0.05        # soft boost for distance-based risks (NOT hard floor)
PPE_PRESENT_THRESH = 0.15       # lower threshold since model is weak
NOISY_OR_MIN_FACTOR = 0.08      # ignore very small probabilities to reduce inflation
PPE_MIN_CONF = 0.30
MACHINE_MIN_CONF = 0.40
PPE_FALLBACK_CONF = 0.45
TOP_K_FACTORS_PER_PERSON = 3


def risk_level(score: float) -> str:
    if score < 0.35:
        return "LOW"
    if score < 0.70:
        return "MEDIUM"
    return "HIGH"


def noisy_or_steps(probabilities, top_k=None):
    probabilities = [float(p) for p in probabilities if p is not None and p > 0.0]
    probabilities.sort(reverse=True)

    if top_k is not None:
        probabilities = probabilities[:top_k]

    if len(probabilities) == 0:
        return {
            "formula": "P = 1 - Pi(1 - p_i)",
            "factor_probs": [],
            "one_minus_terms": [],
            "product": 1.0,
            "final_score": 0.0,
            "calculation_steps": [
                "No factors available, so Pi(1 - p_i) = 1.0",
                "Final P = 1 - 1.0 = 0.0"
            ]
        }

    terms = [round(1.0 - p, 6) for p in probabilities]
    product = 1.0
    for t in terms:
        product *= t

    final = 1.0 - product

    return {
        "formula": "P = 1 - Pi(1 - p_i)",
        "factor_probs": [round(p, 3) for p in probabilities],
        "one_minus_terms": terms,
        "product": round(product, 6),
        "final_score": round(final, 3),
        "calculation_steps": [
            "P = 1 - " + " * ".join([f"(1-{round(p, 3)})" for p in probabilities]),
            "1 - p_i terms = " + " * ".join([str(t) for t in terms]),
            f"Product Pi(1-p_i) = {round(product, 6)}",
            f"Final P = 1 - {round(product, 6)} = {round(final, 3)}"
        ]
    }


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _point_inside_bbox(px, py, bbox):
    x1, y1, x2, y2 = bbox
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def _head_region_bbox(person_bbox):
    px1, py1, px2, py2 = person_bbox
    head_y2 = py1 + 0.35 * (py2 - py1)
    return [px1, py1, px2, head_y2]


def _torso_region_bbox(person_bbox):
    px1, py1, px2, py2 = person_bbox
    person_h = py2 - py1
    torso_y1 = py1 + 0.30 * person_h
    torso_y2 = py1 + 0.85 * person_h
    return [px1, torso_y1, px2, torso_y2]


def _face_region_bbox(person_bbox):
    px1, py1, px2, py2 = person_bbox
    person_h = py2 - py1
    face_y1 = py1 + 0.10 * person_h
    face_y2 = py1 + 0.45 * person_h
    return [px1, face_y1, px2, face_y2]


def _ppe_present_fallback(ppe_bbox, region_bbox):
    cx, cy = _bbox_center(ppe_bbox)
    return _point_inside_bbox(cx, cy, region_bbox)


def _ensure_parent_dir(path: str) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _draw_detections(img_bgr, detections):
    palette = {
        "person": (0, 200, 255),
        "helmet": (0, 255, 0),
        "mask": (255, 180, 0),
        "safety_vest": (255, 0, 200),
        "safety_cone": (255, 90, 0),
        "machinery": (0, 120, 255),
        "vehicle": (120, 255, 120),
    }

    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = palette.get(label, (255, 255, 255))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        label_text = f"{label} {det['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return img_bgr


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union


def _nms_persons(candidates, iou_thresh=0.6):
    persons = [c for c in candidates if c["label"] == "person"]
    others = [c for c in candidates if c["label"] != "person"]

    persons.sort(key=lambda x: x["confidence"], reverse=True)
    kept = []
    for cand in persons:
        keep = True
        for k in kept:
            if _bbox_iou(cand["bbox_scaled"], k["bbox_scaled"]) > iou_thresh:
                keep = False
                break
        if keep:
            kept.append(cand)

    return kept + others


def _scale_bbox(bbox, sx: float, sy: float):
    x1, y1, x2, y2 = bbox
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def analyze_image(
    image_path: str,
    include_llm: bool = True,
    output_json_path: Optional[str] = None,
    output_image_path: Optional[str] = None
) -> Tuple[dict, Optional[str]]:
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Image not found or failed to load: {image_path}")

    orig_h, orig_w = orig.shape[:2]
    img = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
    h, w, _ = img.shape

    detector = ObjectDetector()
    detections = detector.detect(image_path, conf_thresh=DETECT_CONF_THRESH)
    sx = IMG_SIZE / float(orig_w)
    sy = IMG_SIZE / float(orig_h)
    detections_scaled = []
    for d in detections:
        detections_scaled.append({**d, "bbox": _scale_bbox(d["bbox"], sx, sy)})

    objects = []
    relations = []
    factors = []
    risk_probs = []

    person_height_prob = {}
    person_no_helmet_prob = {}
    person_machine_prob = {}
    person_machine_dist = {}

    label_map = {
        "Person": "person",
        "Hardhat": "helmet",
        "Mask": "mask",
        "Safety Vest": "safety_vest",
        "Safety Cone": "safety_cone",
        "machinery": "machinery",
        "vehicle": "vehicle",
        "NO-Hardhat": None,
        "NO-Mask": None,
        "NO-Safety Vest": None,
    }
    allowed_labels = {
        "person",
        "helmet",
        "mask",
        "safety_vest",
        "safety_cone",
        "machinery",
        "vehicle"
    }

    filtered_candidates = []

    for d_orig, d in zip(detections, detections_scaled):
        raw_label = d["label"]
        norm_label = label_map.get(raw_label, raw_label)
        if norm_label is None:
            continue
        if norm_label not in allowed_labels:
            continue

        x1, y1, x2, y2 = d["bbox"]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh
        conf = float(d["confidence"])

        if norm_label == "person":
            if bw < PERSON_MIN_SIDE or bh < PERSON_MIN_SIDE or area < PERSON_MIN_AREA:
                continue
            if conf < PERSON_MIN_CONF_RAW:
                continue
        elif norm_label in ["helmet", "mask", "safety_vest", "safety_cone"]:
            if bw < PPE_MIN_SIDE or bh < PPE_MIN_SIDE or area < PPE_MIN_AREA:
                continue
            if conf < PPE_MIN_CONF:
                continue
        elif norm_label in ["machinery", "vehicle"]:
            if bw < MACHINE_MIN_SIDE or bh < MACHINE_MIN_SIDE or area < MACHINE_MIN_AREA:
                continue
            if conf < MACHINE_MIN_CONF:
                continue

        filtered_candidates.append({
            "label": norm_label,
            "confidence": conf,
            "bbox_scaled": [float(x) for x in d["bbox"]],
            "bbox_orig": [float(x) for x in d_orig["bbox"]]
        })

    filtered_candidates = _nms_persons(filtered_candidates, iou_thresh=0.6)

    filtered_detections = []
    for cand in filtered_candidates:
        obj_id = f"o{len(objects) + 1}"
        objects.append(ObjectNode(
            id=obj_id,
            type=cand["label"],
            confidence=float(cand["confidence"]),
            bbox=[float(x) for x in cand["bbox_scaled"]]
        ))

        filtered_detections.append({
            "label": cand["label"],
            "confidence": cand["confidence"],
            "bbox": cand["bbox_orig"]
        })

    persons_all = [o for o in objects if o.type == "person"]
    valid_persons = [p for p in persons_all if float(p.confidence) >= PERSON_MIN_CONF_RAW]
    per_person_probs = {p.id: [] for p in valid_persons}

    helmets = [o for o in objects if o.type == "helmet"]
    vests = [o for o in objects if o.type == "safety_vest"]
    masks = [o for o in objects if o.type == "mask"]
    machines = [o for o in objects if o.type in ["machinery", "vehicle"]]

    # Floor estimation should use valid persons only
    floor_y = compute_scene_floor_y([p.bbox for p in valid_persons], default_floor_y=h)
    floor_reliable = len(valid_persons) >= 1

    # -----------------------------
    # PPE compliance per person
    # -----------------------------
    per_person_ppe = []
    overlap_cache = {}

    for p in valid_persons:
        head_box = _head_region_bbox(p.bbox)
        torso_box = _torso_region_bbox(p.bbox)
        face_box = _face_region_bbox(p.bbox)

        # Helmet
        best_helmet_overlap = 0.0
        best_helmet_id = None
        best_helmet_conf = 0.0
        helmet_present = False

        for h_obj in helmets:
            ov = helmet_overlap_score(p.bbox, h_obj.bbox)
            if ov > best_helmet_overlap:
                best_helmet_overlap = ov
                best_helmet_id = h_obj.id
                best_helmet_conf = float(h_obj.confidence)

            if ov >= PPE_PRESENT_THRESH:
                helmet_present = True
            elif float(h_obj.confidence) >= PPE_FALLBACK_CONF and _ppe_present_fallback(h_obj.bbox, head_box):
                helmet_present = True

        # Vest
        best_vest_overlap = 0.0
        best_vest_id = None
        best_vest_conf = 0.0
        vest_present = False

        for v_obj in vests:
            ov = vest_overlap_score(p.bbox, v_obj.bbox)
            if ov > best_vest_overlap:
                best_vest_overlap = ov
                best_vest_id = v_obj.id
                best_vest_conf = float(v_obj.confidence)

            if ov >= PPE_PRESENT_THRESH:
                vest_present = True
            elif float(v_obj.confidence) >= PPE_FALLBACK_CONF and _ppe_present_fallback(v_obj.bbox, torso_box):
                vest_present = True

        # Mask
        best_mask_overlap = 0.0
        best_mask_id = None
        best_mask_conf = 0.0
        mask_present = False

        for m_obj in masks:
            ov = mask_overlap_score(p.bbox, m_obj.bbox)
            if ov > best_mask_overlap:
                best_mask_overlap = ov
                best_mask_id = m_obj.id
                best_mask_conf = float(m_obj.confidence)

            if ov >= PPE_PRESENT_THRESH:
                mask_present = True
            elif float(m_obj.confidence) >= PPE_FALLBACK_CONF and _ppe_present_fallback(m_obj.bbox, face_box):
                mask_present = True

        score = (int(helmet_present) + int(vest_present) + int(mask_present)) / 3.0

        per_person_ppe.append({
            "person_id": p.id,
            "helmet": helmet_present,
            "vest": vest_present,
            "mask": mask_present,
            "score": round(score, 3),
            "helmet_overlap": round(best_helmet_overlap, 3),
            "helmet_conf": round(best_helmet_conf, 3),
            "vest_overlap": round(best_vest_overlap, 3),
            "vest_conf": round(best_vest_conf, 3),
            "mask_overlap": round(best_mask_overlap, 3)
            ,
            "mask_conf": round(best_mask_conf, 3)
        })

        overlap_cache[p.id] = {
            "helmet_overlap": float(best_helmet_overlap),
            "helmet_id": best_helmet_id,
            "helmet_conf": float(best_helmet_conf),
            "vest_overlap": float(best_vest_overlap),
            "vest_id": best_vest_id,
            "vest_conf": float(best_vest_conf),
            "mask_overlap": float(best_mask_overlap),
            "mask_id": best_mask_id
            ,
            "mask_conf": float(best_mask_conf)
        }

    # Scene-level PPE summary
    if len(per_person_ppe) == 0:
        ppe_summary = {
            "helmet": None,
            "mask": None,
            "vest": None,
            "score": None,
            "note": f"No valid person detected (PERSON_MIN_CONF_RAW={PERSON_MIN_CONF_RAW})"
        }
    else:
        helmet_ok = all(x["helmet"] for x in per_person_ppe)
        vest_ok = all(x["vest"] for x in per_person_ppe)
        mask_ok = all(x["mask"] for x in per_person_ppe)
        avg_score = sum(x["score"] for x in per_person_ppe) / len(per_person_ppe)

        ppe_summary = {
            "helmet": helmet_ok,
            "mask": mask_ok,
            "vest": vest_ok,
            "score": round(avg_score, 3)
        }

    for p in valid_persons:
        person_conf_raw = float(p.confidence)
        person_conf_used = min(1.0, person_conf_raw + PERSON_CONF_BOOST)

        floor_score, floor_dist_px = distance_to_floor_score(
            person_bbox=p.bbox,
            floor_y=floor_y,
            max_allowed=250.0
        )

        # Elevation proxy: if feet are far from image bottom, treat as elevated
        bottom_y = float(p.bbox[3])
        proxy_thresh = 0.85 * IMG_SIZE
        proxy_score = 0.0
        if bottom_y < proxy_thresh:
            proxy_score = clamp01((proxy_thresh - bottom_y) / (0.35 * IMG_SIZE))

        combined_floor_score = max(floor_score, proxy_score)

        triggered, prob, evidence = person_at_height_risk(
            person_conf=person_conf_used,
            floor_distance_px=floor_dist_px,
            floor_score=combined_floor_score
        )

        if proxy_score > floor_score:
            evidence.append("note=height_proxy_used")

        person_height_prob[p.id] = prob

        factors.append(RiskFactor(
            name="person_at_height_indicator",
            triggered=triggered,
            probability=prob,
            evidence=[f"object_id={p.id}"] + evidence + [f"person_conf_used={person_conf_used:.2f}"]
        ))

        relations.append(RelationEdge(
            from_id=p.id,
            to_id="scene_floor_estimate",
            relation="distance_to_floor_line_px",
            value=float(floor_dist_px),
            prob=float(floor_score)
        ))

        if prob >= NOISY_OR_MIN_FACTOR:
            risk_probs.append(prob)
            per_person_probs[p.id].append(prob)


    for i in range(len(valid_persons)):
        for j in range(i + 1, len(valid_persons)):
            a = valid_persons[i]
            b = valid_persons[j]

            dist_px = pairwise_pixel_distance(a.bbox, b.bbox)
            avg_w = max(1.0, (abs(a.bbox[2] - a.bbox[0]) + abs(b.bbox[2] - b.bbox[0])) / 2.0)
            dist_norm = dist_px / avg_w

            triggered, prob, evidence = person_person_proximity_risk(
                dist_norm=dist_norm,
                conf_a=float(a.confidence),
                conf_b=float(b.confidence),
                near_thresh_norm=6.0
            )

            if prob <= 0.0:
                continue

            relations.append(RelationEdge(
                from_id=a.id,
                to_id=b.id,
                relation="person_person_distance_norm",
                value=float(dist_norm),
                prob=float(prob)
            ))

            factors.append(RiskFactor(
                name="person_person_proximity_indicator",
                triggered=triggered,
                probability=prob,
                evidence=[f"pair={a.id}-{b.id}"] + evidence
            ))

            if prob >= NOISY_OR_MIN_FACTOR:
                risk_probs.append(prob)
                per_person_probs[a.id].append(prob)
                per_person_probs[b.id].append(prob)

    # -----------------------------
    # Risk: person near machine
    # -----------------------------
    for p in valid_persons:
        person_conf_raw = float(p.confidence)
        person_conf_used = min(1.0, person_conf_raw + PERSON_CONF_BOOST)

        for m in machines:
            dist_px = pairwise_pixel_distance(p.bbox, m.bbox)
            avg_w = max(1.0, abs(p.bbox[2] - p.bbox[0]))
            dist_norm = dist_px / avg_w

            triggered, prob, evidence = person_near_machine_risk(
                dist_norm=dist_norm,
                person_conf=person_conf_used,
                machine_conf=float(m.confidence),
                near_thresh_norm=3.0
            )

            person_machine_prob[(p.id, m.id)] = prob
            person_machine_dist[(p.id, m.id)] = dist_norm

            if prob <= 0.0:
                continue

            relations.append(RelationEdge(
                from_id=p.id,
                to_id=m.id,
                relation="person_machine_distance_norm",
                value=float(dist_norm),
                prob=float(prob)
            ))

            factors.append(RiskFactor(
                name="person_near_machine_indicator",
                triggered=triggered,
                probability=prob,
                evidence=[f"pair={p.id}-{m.id}"] + evidence
            ))

            if prob >= NOISY_OR_MIN_FACTOR:
                risk_probs.append(prob)
                per_person_probs[p.id].append(prob)

    # -----------------------------
    # Risk: person extremely close to machine (context)
    # -----------------------------
    for (pid, mid), base_prob in person_machine_prob.items():
        dist_px = person_machine_dist[(pid, mid)]

        triggered, prob, evidence = person_machine_close_context_risk(
            person_near_machine_prob=base_prob,
            dist_px=dist_px,
            danger_dist_px=1.0
        )

        if prob <= 0.0:
            continue

        relations.append(RelationEdge(
            from_id=pid,
            to_id=mid,
            relation="person_machine_close_context",
            value=float(dist_px),
            prob=float(prob)
        ))

        factors.append(RiskFactor(
            name="person_machine_close_context_indicator",
            triggered=triggered,
            probability=prob,
            evidence=[f"pair={pid}-{mid}"] + evidence
        ))

        if prob >= NOISY_OR_MIN_FACTOR:
            risk_probs.append(prob)
            per_person_probs[pid].append(prob)

    # -----------------------------
    # PPE missing risks (helmet/vest/mask)
    # -----------------------------
    for p in valid_persons:
        person_conf_raw = float(p.confidence)
        cached = overlap_cache.get(p.id, {})

        best_helmet_overlap = float(cached.get("helmet_overlap", 0.0))
        best_helmet_id = cached.get("helmet_id", None)
        best_helmet_conf = float(cached.get("helmet_conf", 0.0))

        best_vest_overlap = float(cached.get("vest_overlap", 0.0))
        best_vest_id = cached.get("vest_id", None)
        best_vest_conf = float(cached.get("vest_conf", 0.0))

        best_mask_overlap = float(cached.get("mask_overlap", 0.0))
        best_mask_id = cached.get("mask_id", None)
        best_mask_conf = float(cached.get("mask_conf", 0.0))

        # Helmet
        triggered, prob, evidence = person_no_helmet_risk(
            person_conf=person_conf_raw,
            best_helmet_overlap=best_helmet_overlap,
            best_helmet_conf=best_helmet_conf
        )
        person_no_helmet_prob[p.id] = prob

        relations.append(RelationEdge(
            from_id=p.id,
            to_id=(best_helmet_id if best_helmet_id else "helmet_missing"),
            relation="helmet_overlap_score",
            value=float(best_helmet_overlap),
            prob=float(1.0 - best_helmet_overlap)
        ))

        factors.append(RiskFactor(
            name="person_no_helmet_indicator",
            triggered=triggered,
            probability=prob,
            evidence=[f"object_id={p.id}"] + evidence
        ))

        if prob >= NOISY_OR_MIN_FACTOR:
            risk_probs.append(prob)
            per_person_probs[p.id].append(prob)

        # Vest
        triggered, prob, evidence = person_no_vest_risk(
            person_conf=person_conf_raw,
            best_vest_overlap=best_vest_overlap,
            best_vest_conf=best_vest_conf
        )

        relations.append(RelationEdge(
            from_id=p.id,
            to_id=(best_vest_id if best_vest_id else "vest_missing"),
            relation="vest_overlap_score",
            value=float(best_vest_overlap),
            prob=float(1.0 - best_vest_overlap)
        ))

        factors.append(RiskFactor(
            name="person_no_vest_indicator",
            triggered=triggered,
            probability=prob,
            evidence=[f"object_id={p.id}"] + evidence
        ))

        if prob >= NOISY_OR_MIN_FACTOR:
            risk_probs.append(prob)
            per_person_probs[p.id].append(prob)

        # Mask
        triggered, prob, evidence = person_no_mask_risk(
            person_conf=person_conf_raw,
            best_mask_overlap=best_mask_overlap,
            best_mask_conf=best_mask_conf
        )

        relations.append(RelationEdge(
            from_id=p.id,
            to_id=(best_mask_id if best_mask_id else "mask_missing"),
            relation="mask_overlap_score",
            value=float(best_mask_overlap),
            prob=float(1.0 - best_mask_overlap)
        ))

        factors.append(RiskFactor(
            name="person_no_mask_indicator",
            triggered=triggered,
            probability=prob,
            evidence=[f"object_id={p.id}"] + evidence
        ))

        if prob >= NOISY_OR_MIN_FACTOR:
            risk_probs.append(prob)
            per_person_probs[p.id].append(prob)

    # Combined contextual risk: height + no helmet
    for p in valid_persons:
        p_height = person_height_prob.get(p.id, 0.0)
        p_no_helmet = person_no_helmet_prob.get(p.id, 0.0)

        triggered, prob, evidence = person_height_no_helmet_risk(p_height, p_no_helmet)

        relations.append(RelationEdge(
            from_id=p.id,
            to_id="context_height_no_helmet",
            relation="combined_risk",
            value=float(prob),
            prob=float(prob)
        ))

        factors.append(RiskFactor(
            name="person_height_no_helmet_indicator",
            triggered=triggered,
            probability=prob,
            evidence=[f"object_id={p.id}"] + evidence
        ))

        if prob >= NOISY_OR_MIN_FACTOR:
            risk_probs.append(prob)
            per_person_probs[p.id].append(prob)

    # Final score: compute per-person, then scene = max(per-person)
    per_person_scores = {}
    for pid, probs in per_person_probs.items():
        per_person_scores[pid] = noisy_or_steps(probs)["final_score"] if probs else 0.0

    final_score = max(per_person_scores.values()) if per_person_scores else 0.0
    final_level = risk_level(final_score)

    output = {
        "risk_level": final_level,
        "confidence": round(final_score, 3),
        "ppe_compliance": ppe_summary,
        "ppe_per_person": per_person_ppe,
        "factors": [
            f"{f.name} (p={f.probability:.2f}) for {', '.join(f.evidence[:2])}"
            for f in sorted(factors, key=lambda x: x.probability, reverse=True)[:5]
        ],
        "risk_graph": build_risk_graph(objects, relations, factors),
        "math_transparency": {
            "method": "Max-Per-Person",
            "scene_floor_y": float(floor_y),
            "floor_reliable": bool(floor_reliable),
            "per_person_scores": {k: round(v, 3) for k, v in per_person_scores.items()},
            "scene_score": round(final_score, 3)
        }
    }

    if include_llm:
        try:
            output["llm_explanation"] = explain_risk_with_ollama(output, model="llama3.1:8b")
        except Exception as e:
            output["llm_explanation_error"] = str(e)

    if output_json_path:
        _ensure_parent_dir(output_json_path)
        with open(output_json_path, "w") as f:
            json.dump(output, f, indent=2)

    output_image_saved = None
    if output_image_path:
        _ensure_parent_dir(output_image_path)
        annotated = _draw_detections(orig.copy(), filtered_detections)
        cv2.imwrite(output_image_path, annotated)
        output_image_saved = output_image_path

    return output, output_image_saved
