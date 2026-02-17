from __future__ import annotations

import json

from parallel import Parallel
from pydantic import BaseModel


class ParallelClient:
    """Thin wrapper around the Parallel AI Task API."""

    def __init__(self, api_key: str):
        self.client = Parallel(api_key=api_key)

    def deep_research(
        self,
        prompt: str,
        processor: str = "pro",
        output_schema: dict | None = None,
    ) -> dict:
        task_spec = None
        if output_schema is not None:
            task_spec = {
                "output_schema": {"type": "json", "json_schema": output_schema}
            }

        task_run = self.client.task_run.create(
            input=prompt,
            processor=processor,
            task_spec=task_spec,
        )
        result = self.client.task_run.result(task_run.run_id, api_timeout=3600)
        output = result.output

        # SDK may return pydantic models — convert to plain dict
        if isinstance(output, BaseModel):
            output = output.model_dump()
        elif isinstance(output, str):
            output = json.loads(output)

        return output
