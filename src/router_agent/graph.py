from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.embeddings.factory import build_embedding_model
from src.llm.local_model import resolve_ollama_model
from src.llm.prompt_strategies import get_prompt_strategy
from src.retrieval.vector_store import load_chroma_index
from src.router_agent.schemas import RouteDecision, RouterInput, RouterOutput, RouterState
from src.utils.paths import DB_DIR
from langchain_ollama import ChatOllama


ROUTER_PROMPT = SystemMessage(
    "You are a router for a local fitness and nutrition assistant. "
    "Choose exactly one domain for each question:\n"
    "- fitness: exercises, training, routines, muscles, movement, workout technique\n"
    "- nutrition: meals, ingredients, calories, protein, carbs, fat, supplements, diet\n"
    "Return only the structured route decision."
)

FITNESS_PROMPT = SystemMessage(
    "You are a helpful fitness assistant. Answer only from the retrieved fitness chunks. "
    "Do not invent details beyond the retrieved context."
)

NUTRITION_PROMPT = SystemMessage(
    "You are a helpful nutrition assistant. Answer only from the retrieved nutrition chunks. "
    "Do not invent details beyond the retrieved context."
)


def build_router_graph(
    model_name: str | None = None,
    db_path: str | None = None,
    prompt_strategy: str = "few-shot",
):
    # Este grafo separa el flujo en dos dominios:
    # - fitness: ejercicios, técnica, entrenamiento
    # - nutrition: comidas, ingredientes, macros, suplementos
    vector_store = load_chroma_index(db_path or DB_DIR, build_embedding_model())
    routing_model = _build_chat_model(model_name=model_name, temperature=0)
    generation_model = _build_chat_model(model_name=model_name, temperature=0.1)

    fitness_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"record_type": "exercise"},
        }
    )
    nutrition_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"$or": [{"record_type": "dish"}, {"record_type": "diet_pdf"}]},
        }
    )

    router_chain = routing_model.with_structured_output(RouteDecision)
    prompt_builder = get_prompt_strategy(prompt_strategy).builder

    def router_node(state: RouterState) -> RouterState:
        # Decide a qué rama debe ir la consulta antes de recuperar contexto.
        user_message = HumanMessage(state["user_query"])
        decision = router_chain.invoke([ROUTER_PROMPT, *state["messages"], user_message])
        return {
            "domain": decision.domain,
            "route_reason": decision.reason,
            "messages": [user_message],
        }

    def pick_retriever(state: RouterState) -> Literal["retrieve_fitness", "retrieve_nutrition"]:
        # Edge condicional de LangGraph: el router decide el siguiente nodo.
        return "retrieve_fitness" if state["domain"] == "fitness" else "retrieve_nutrition"

    def retrieve_fitness(state: RouterState) -> RouterState:
        # Retrieval especializado solo sobre chunks de ejercicios.
        return {"documents": fitness_retriever.invoke(state["user_query"])}

    def retrieve_nutrition(state: RouterState) -> RouterState:
        # Retrieval especializado sobre platos y PDFs de dieta/suplementación.
        return {"documents": nutrition_retriever.invoke(state["user_query"])}

    def generate_answer(state: RouterState) -> RouterState:
        # La generación final reutiliza la misma lógica de prompting del proyecto,
        # pero adaptando el system prompt al dominio elegido.
        system_prompt = FITNESS_PROMPT if state["domain"] == "fitness" else NUTRITION_PROMPT
        prompt = prompt_builder(state["user_query"], state["documents"])
        response = generation_model.invoke([system_prompt, *state["messages"], HumanMessage(prompt)])
        return {
            "answer": response.content,
            "messages": [response],
        }

    builder = StateGraph(RouterState, input_schema=RouterInput, output_schema=RouterOutput)
    builder.add_node("router", router_node)
    builder.add_node("retrieve_fitness", retrieve_fitness)
    builder.add_node("retrieve_nutrition", retrieve_nutrition)
    builder.add_node("generate_answer", generate_answer)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", pick_retriever)
    builder.add_edge("retrieve_fitness", "generate_answer")
    builder.add_edge("retrieve_nutrition", "generate_answer")
    builder.add_edge("generate_answer", END)
    return builder.compile()


def _build_chat_model(model_name: str | None, temperature: float) -> ChatOllama:
    # Constructor común para los nodos del router y de generación.
    return ChatOllama(
        base_url="http://localhost:11434",
        model=model_name or resolve_ollama_model(),
        temperature=temperature,
        num_ctx=2048,
        num_predict=256,
        seed=42,
    )
