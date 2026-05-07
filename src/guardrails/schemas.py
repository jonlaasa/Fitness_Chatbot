from __future__ import annotations

from pydantic import BaseModel, Field


class ScopeDecision(BaseModel):
    in_scope: bool = Field(
        description="True if the user question is within the project scope."
    )
    reason: str = Field(
        description="Short justification for the scope decision."
    )


class GuardedRagAnswer(BaseModel):
    answer: str = Field(
        description="Final answer for the user based only on the retrieved context."
    )
    grounded_in_context: bool = Field(
        description="True if the answer can be supported by the retrieved context."
    )
    fallback_reason: str = Field(
        default="",
        description="Reason to explain why the answer is limited or insufficient.",
    )


class GuardedAgentAnswer(BaseModel):
    answer: str = Field(
        description="Final cleaned answer for the user."
    )
    safe_and_in_scope: bool = Field(
        description="True if the final answer stays within the project scope and is safe to return."
    )
    fallback_reason: str = Field(
        default="",
        description="Reason to explain why the answer should be limited or replaced.",
    )
