from __future__ import annotations

import json
from typing import List

from pydantic import BaseModel, Field, validator

class ScenarioTestCase(BaseModel):
    anchor_utterance_index: int = Field(..., ge=0)
    suggested_tests: List[str] = Field(..., min_items=1)
    customer_text: str
    seller_action: str
    reason: str = Field(default="", description="LLM-generated explanation for why this case was identified")
    
    ## actually said
    @validator("suggested_tests")
    def _validate_tests(cls, value: List[str]) -> List[str]:
        uniq = []
        for test in value:
            test = str(test).strip()
            if not test:
                continue
            if test not in uniq:
                uniq.append(test)
        if not uniq:
            raise ValueError("suggested_tests must contain at least one non-empty test type")
        return uniq
