from typing import Optional, Dict, List, Literal
from pydantic import BaseModel


class Observation(BaseModel):
    invoice_text: str
    extracted_fields: Dict[str, Optional[str]]
    validation_errors: List[str]
    fraud_detected: bool
    step_count: int
    goal: str


class Action(BaseModel):
    action_type: Literal["extract_field", "validate", "flag_fraud", "finish"]
    field_name: Optional[str] = None
    value: Optional[str] = None


class Reward(BaseModel):
    value: float
    reason: str
    cumulative: float