import cv2
import json

from vision.detector import ObjectDetector
from vision.geometry import compute_scene_floor_y, distance_to_floor_score, pairwise_pixel_distance
from risk.rules import person_at_height_risk, person_person_proximity_risk
from risk.risk_graph import ObjectNode, RelationEdge, RiskFactor, build_risk_graph

from llm.explain import explain_risk_with_ollama


def noisy_or(probabilities):
    p_not = 1.0
    for p in probabilities:
        p_not *= (1.0 - p)
    return 1.0 - p_not


def risk_level(score: float) -> str:
    if score < 0.35:
        return "LOW"
    if score < 0.70:
        return "MEDIUM"
    return "HIGH"


def main():
    image_path = "samples/image1.jpg"

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or failed to load: {image_path}")

    h, w, _ = img.shape

    detector = ObjectDetector()
    detections = detector.detect(image_path)

    objects = []
    relations = []
    factors = []
    risk_probs = []

    # Build object nodes
    for i, d in enumerate(detections):
        objects.append(ObjectNode(
            id=f"o{i+1}",
            type=d["label"],
            confidence=float(d["confidence"]),
            bbox=[float(x) for x in d["bbox"]]
        ))

    # Step 5: Estimate floor/work-surface line from detected persons
    person_bboxes = [o.bbox for o in objects if o.type == "person"]
    floor_y = compute_scene_floor_y(person_bboxes, default_floor_y=h)

    # Step 5: Person-at-height indicator per person
    for obj in objects:
        if obj.type != "person":
            continue

        floor_score, floor_dist_px = distance_to_floor_score(
            person_bbox=obj.bbox,
            floor_y=floor_y,
            max_allowed=250.0
        )

        triggered, prob, evidence = person_at_height_risk(
            person_conf=obj.confidence,
            floor_distance_px=floor_dist_px,
            floor_score=floor_score
        )

        factors.append(RiskFactor(
            name="person_at_height_indicator",
            triggered=triggered,
            probability=prob,
            evidence=[f"object_id={obj.id}"] + evidence
        ))

        relations.append(RelationEdge(
            from_id=obj.id,
            to_id="scene_floor_estimate",
            relation="distance_to_floor_line_px",
            value=float(floor_dist_px),
            prob=float(floor_score)
        ))

        risk_probs.append(prob)

    # Step 6: Person-to-person proximity contextual risk
    persons = [o for o in objects if o.type == "person"]

    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            a = persons[i]
            b = persons[j]

            dist_px = pairwise_pixel_distance(a.bbox, b.bbox)

            triggered, prob, evidence = person_person_proximity_risk(
                dist_px=dist_px,
                conf_a=a.confidence,
                conf_b=b.confidence,
                near_thresh_px=350.0
            )

            # Keep JSON clean
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

            risk_probs.append(prob)

    # Final scoring AFTER all factors are included
    final_score = noisy_or(risk_probs)
    final_level = risk_level(final_score)

    output = {
        "risk_level": final_level,
        "confidence": round(final_score, 3),
        "factors": [
            f"{f.name} (p={f.probability:.2f}) for {', '.join(f.evidence[:2])}"
            for f in factors if f.triggered
        ],
        "risk_graph": build_risk_graph(objects, relations, factors),
        "math_transparency": {
            "method": "Noisy-OR",
            "scene_floor_y": float(floor_y),
            "per_factor_probs": [round(p, 3) for p in risk_probs],
            "final_score": round(final_score, 3)
        }
    }

    # Step 7: LLM Explanation (Ollama Local)
    try:
        output["llm_explanation"] = explain_risk_with_ollama(output, model="llama3.1:8b")
    except Exception as e:
        output["llm_explanation_error"] = str(e)

    print(json.dumps(output, indent=2))

    with open("outputs/result.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()