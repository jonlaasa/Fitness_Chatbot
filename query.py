import os
import subprocess

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# --- CONFIGURACION ---
DB_PATH = "db/"
DEFAULT_MODEL = "phi3:latest"


def resolve_ollama_model():
    model_from_env = os.getenv("OLLAMA_MODEL")
    if model_from_env:
        return model_from_env

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return DEFAULT_MODEL

    model_names = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            model_names.append(parts[0])

    if DEFAULT_MODEL in model_names:
        return DEFAULT_MODEL

    for model_name in model_names:
        if model_name.startswith("phi3:"):
            return model_name

    return DEFAULT_MODEL


def run_query():
    print("Cargando motor de busqueda...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists(DB_PATH):
        print(f"Error: No existe la carpeta '{DB_PATH}'.")
        return

    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    model_name = resolve_ollama_model()
    print(f"Conectando con Ollama usando el modelo: {model_name}")
    llm = OllamaLLM(model=model_name)

    print("\nIA lista. Preguntame sobre tus documentos.\n")

    while True:
        user_query = input("Tu pregunta: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        docs = db.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = (
            f"<|user|>\n"
            f"Responde SIEMPRE en ESPANOL.\n"
            f"Usa SOLO el siguiente contexto para responder.\n"
            f"NO uses conocimiento externo.\n\n"
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA: {user_query}<|end|>\n"
            f"<|assistant|>"
        )

        answer = llm.invoke(prompt)

        print(context)
        print(f"\nRespuesta del modelo:\n{answer}\n")
        print("-" * 30)


if __name__ == "__main__":
    run_query()
