from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class ObjectNode:
    id: str
    type: str
    confidence: float
    bbox: list

@dataclass
class RelationEdge:
    from_id: str
    to_id: str
    relation: str
    value: float
    prob: float

@dataclass
class RiskFactor:
    name: str
    triggered: bool
    probability: float
    evidence: List[str]

def build_risk_graph(objects: List[ObjectNode],
                     relations: List[RelationEdge],
                     risk_factors: List[RiskFactor]) -> Dict[str, Any]:
    return {
        "objects": [asdict(o) for o in objects],
        "relations": [asdict(r) for r in relations],
        "risk_factors": [asdict(f) for f in risk_factors]
    }