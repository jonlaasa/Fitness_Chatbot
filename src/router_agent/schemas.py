from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    domain: Literal["fitness", "nutrition"] = Field(
        description="Domain selected for the user query."
    )
    reason: str = Field(description="Short explanation for the routing decision.")


class RouterState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    domain: Literal["fitness", "nutrition"]
    route_reason: str
    documents: list[Document]
    answer: str


class RouterInput(TypedDict):
    user_query: str


class RouterOutput(TypedDict):
    domain: Literal["fitness", "nutrition"]
    route_reason: str
    documents: list[Document]
    answer: str
