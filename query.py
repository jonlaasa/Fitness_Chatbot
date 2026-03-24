import os
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURACIÓN ---
DB_PATH = "db/"

def run_query():
    print("🧠 Cargando motor de búsqueda...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: No existe la carpeta '{DB_PATH}'.")
        return

    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    print(f"🚀 Conectando con Microsoft Phi-3 (el segundo modelo)...")
    # Usamos el nombre que Ollama le dio en tu PC
    llm = Ollama(model="phi3:latest")

    print("\n✅ ¡IA Lista! Pregúntame sobre tus documentos.\n")

    while True:
        query = input("❓ Tu pregunta: ")
        if query.lower() in ['exit', 'quit']: break

        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        # PROMPT BLINDADO: Castellano + No Inventar
        prompt = (
            f"<|user|>\n"
            f"Responde SIEMPRE en ESPAÑOL.\n"
            f"Usa SOLO el siguiente contexto para responder.\n"
            f"NO uses conocimiento externo. Dame del contexto lo que mas se parezca pero luego ajustalo. .\n\n"
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA: {query}<|end|>\n"
            f"<|assistant|>"
        )

        answer = llm.invoke(prompt)
        
        print(context)
        print(f"\n🤖 Phi-3 responde:\n{answer}\n")
        print("-" * 30)

if __name__ == "__main__":
    run_query()