import cv2
import json

from vision.detector import ObjectDetector
from vision.geometry import compute_scene_floor_y, distance_to_floor_score, pairwise_pixel_distance

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

# Balanced settings (recommended for website deployment)
PERSON_MIN_CONF_RAW = 0.15      # ignore person below this
PERSON_CONF_BOOST = 0.10        # soft boost for distance-based risks (NOT hard floor)
PPE_PRESENT_THRESH = 0.12       # lower threshold since model is weak
NOISY_OR_MIN_FACTOR = 0.05      # ignore very small probabilities to reduce inflation


def risk_level(score: float) -> str:
    if score < 0.35:
        return "LOW"
    if score < 0.70:
        return "MEDIUM"
    return "HIGH"


def noisy_or_steps(probabilities):
    probabilities = [float(p) for p in probabilities if p is not None]

    if len(probabilities) == 0:
        return {
            "formula": "P = 1 - Π(1 - p_i)",
            "factor_probs": [],
            "one_minus_terms": [],
            "product": 1.0,
            "final_score": 0.0,
            "calculation_steps": [
                "No factors available, so Π(1 - p_i) = 1.0",
                "Final P = 1 - 1.0 = 0.0"
            ]
        }

    terms = [round(1.0 - p, 6) for p in probabilities]
    product = 1.0
    for t in terms:
        product *= t

    final = 1.0 - product

    return {
        "formula": "P = 1 - Π(1 - p_i)",
        "factor_probs": [round(p, 3) for p in probabilities],
        "one_minus_terms": terms,
        "product": round(product, 6),
        "final_score": round(final, 3),
        "calculation_steps": [
            "P = 1 - " + " * ".join([f"(1-{round(p, 3)})" for p in probabilities]),
            "1 - p_i terms = " + " * ".join([str(t) for t in terms]),
            f"Product Π(1-p_i) = {round(product, 6)}",
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


def main():
    image_path = "samples/image1.jpg"

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or failed to load: {image_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    h, w, _ = img.shape

    detector = ObjectDetector()
    detections = detector.detect(image_path)

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
    }

    for i, d in enumerate(detections):
        raw_label = d["label"]
        norm_label = label_map.get(raw_label, raw_label)

        objects.append(ObjectNode(
            id=f"o{i+1}",
            type=norm_label,
            confidence=float(d["confidence"]),
            bbox=[float(x) for x in d["bbox"]]
        ))

    persons_all = [o for o in objects if o.type == "person"]
    valid_persons = [p for p in persons_all if float(p.confidence) >= PERSON_MIN_CONF_RAW]

    helmets = [o for o in objects if o.type == "helmet"]
    vests = [o for o in objects if o.type == "safety_vest"]
    masks = [o for o in objects if o.type == "mask"]
    machines = [o for o in objects if o.type in ["machinery", "vehicle"]]

    # Floor estimation should use valid persons only
    floor_y = compute_scene_floor_y([p.bbox for p in valid_persons], default_floor_y=h)

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
        helmet_present = False

        for h_obj in helmets:
            ov = helmet_overlap_score(p.bbox, h_obj.bbox)
            if ov > best_helmet_overlap:
                best_helmet_overlap = ov
                best_helmet_id = h_obj.id

            if ov >= PPE_PRESENT_THRESH:
                helmet_present = True
            elif _ppe_present_fallback(h_obj.bbox, head_box):
                helmet_present = True

        # Vest
        best_vest_overlap = 0.0
        best_vest_id = None
        vest_present = False

        for v_obj in vests:
            ov = vest_overlap_score(p.bbox, v_obj.bbox)
            if ov > best_vest_overlap:
                best_vest_overlap = ov
                best_vest_id = v_obj.id

            if ov >= PPE_PRESENT_THRESH:
                vest_present = True
            elif _ppe_present_fallback(v_obj.bbox, torso_box):
                vest_present = True

        # Mask
        best_mask_overlap = 0.0
        best_mask_id = None
        mask_present = False

        for m_obj in masks:
            ov = mask_overlap_score(p.bbox, m_obj.bbox)
            if ov > best_mask_overlap:
                best_mask_overlap = ov
                best_mask_id = m_obj.id

            if ov >= PPE_PRESENT_THRESH:
                mask_present = True
            elif _ppe_present_fallback(m_obj.bbox, face_box):
                mask_present = True

        score = (int(helmet_present) + int(vest_present) + int(mask_present)) / 3.0

        per_person_ppe.append({
            "person_id": p.id,
            "helmet": helmet_present,
            "vest": vest_present,
            "mask": mask_present,
            "score": round(score, 3),
            "helmet_overlap": round(best_helmet_overlap, 3),
            "vest_overlap": round(best_vest_overlap, 3),
            "mask_overlap": round(best_mask_overlap, 3)
        })

        overlap_cache[p.id] = {
            "helmet_overlap": float(best_helmet_overlap),
            "helmet_id": best_helmet_id,
            "vest_overlap": float(best_vest_overlap),
            "vest_id": best_vest_id,
            "mask_overlap": float(best_mask_overlap),
            "mask_id": best_mask_id
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

    # -----------------------------
    # Risk: person-at-height
    # -----------------------------
    for p in valid_persons:
        person_conf_raw = float(p.confidence)
        person_conf_used = min(1.0, person_conf_raw + PERSON_CONF_BOOST)

        floor_score, floor_dist_px = distance_to_floor_score(
            person_bbox=p.bbox,
            floor_y=floor_y,
            max_allowed=250.0
        )

        triggered, prob, evidence = person_at_height_risk(
            person_conf=person_conf_used,
            floor_distance_px=floor_dist_px,
            floor_score=floor_score
        )

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

    # -----------------------------
    # Risk: person-person proximity
    # -----------------------------
    for i in range(len(valid_persons)):
        for j in range(i + 1, len(valid_persons)):
            a = valid_persons[i]
            b = valid_persons[j]

            dist_px = pairwise_pixel_distance(a.bbox, b.bbox)

            triggered, prob, evidence = person_person_proximity_risk(
                dist_px=dist_px,
                conf_a=float(a.confidence),
                conf_b=float(b.confidence),
                near_thresh_px=350.0
            )

            if prob <= 0.0:
                continue

            relations.append(RelationEdge(
                from_id=a.id,
                to_id=b.id,
                relation="person_person_distance_px",
                value=float(dist_px),
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

    # -----------------------------
    # Risk: person near machine
    # -----------------------------
    for p in valid_persons:
        person_conf_raw = float(p.confidence)
        person_conf_used = min(1.0, person_conf_raw + PERSON_CONF_BOOST)

        for m in machines:
            dist_px = pairwise_pixel_distance(p.bbox, m.bbox)

            triggered, prob, evidence = person_near_machine_risk(
                dist_px=dist_px,
                person_conf=person_conf_used,
                machine_conf=float(m.confidence),
                near_thresh_px=100.0
            )

            person_machine_prob[(p.id, m.id)] = prob
            person_machine_dist[(p.id, m.id)] = dist_px

            if prob <= 0.0:
                continue

            relations.append(RelationEdge(
                from_id=p.id,
                to_id=m.id,
                relation="person_machine_distance_px",
                value=float(dist_px),
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

    # -----------------------------
    # Risk: person extremely close to machine (context)
    # -----------------------------
    for (pid, mid), base_prob in person_machine_prob.items():
        dist_px = person_machine_dist[(pid, mid)]

        triggered, prob, evidence = person_machine_close_context_risk(
            person_near_machine_prob=base_prob,
            dist_px=dist_px,
            danger_dist_px=25.0
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

    # -----------------------------
    # PPE missing risks (helmet/vest/mask)
    # -----------------------------
    for p in valid_persons:
        person_conf_raw = float(p.confidence)
        cached = overlap_cache.get(p.id, {})

        best_helmet_overlap = float(cached.get("helmet_overlap", 0.0))
        best_helmet_id = cached.get("helmet_id", None)

        best_vest_overlap = float(cached.get("vest_overlap", 0.0))
        best_vest_id = cached.get("vest_id", None)

        best_mask_overlap = float(cached.get("mask_overlap", 0.0))
        best_mask_id = cached.get("mask_id", None)

        # Helmet
        triggered, prob, evidence = person_no_helmet_risk(
            person_conf=person_conf_raw,
            best_helmet_overlap=best_helmet_overlap
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

        # Vest
        triggered, prob, evidence = person_no_vest_risk(
            person_conf=person_conf_raw,
            best_vest_overlap=best_vest_overlap
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

        # Mask
        triggered, prob, evidence = person_no_mask_risk(
            person_conf=person_conf_raw,
            best_mask_overlap=best_mask_overlap
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

    # Final score
    math_transparency = noisy_or_steps(risk_probs)
    final_score = float(math_transparency["final_score"])
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
            "method": "Noisy-OR",
            "scene_floor_y": float(floor_y),
            **math_transparency
        }
    }

    try:
        output["llm_explanation"] = explain_risk_with_ollama(output, model="llama3.1:8b")
    except Exception as e:
        output["llm_explanation_error"] = str(e)

    print(json.dumps(output, indent=2))

    with open("outputs/result.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()