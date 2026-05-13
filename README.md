# Local Fitness RAG

Lightweight local RAG prototype for a university project focused on sports science, exercise, training, basic nutrition, and healthy habits.

This repository is intentionally small, CPU-friendly, and easy to explain in an academic defense. It is not a production system. It is a first iteration designed to run locally on a laptop with 16 GB RAM, an Intel Core i5, and no dedicated GPU.

## 1. Project goal

The goal is to build a local assistant that:

- answers questions about exercises, training, and basic nutrition
- uses external knowledge instead of relying only on the language model
- retrieves relevant documents before generating an answer
- stays simple enough to justify every design decision in a university report

This prototype is inspired by products such as Whoop, Freeletics, and Fitbod, but it does not attempt to reproduce them fully. Instead, it focuses on a clean and modular RAG baseline.

## 2. Why RAG

RAG, or Retrieval-Augmented Generation, is used because a local LLM alone can produce generic or hallucinated answers. By retrieving relevant domain documents before generation, the system improves grounding and makes the answer path easier to explain:

1. the user asks a question
2. the system converts the question into an embedding
3. the vector database retrieves the most semantically similar documents
4. the LLM answers using that retrieved context

This is more academically defensible than asking a local model to answer from parametric memory only.

## 3. Data sources

Primary exercise source:

- [longhaul-fitness/exercises](https://github.com/longhaul-fitness/exercises)

According to the repository README, the main exercise fields include `name`, `slug`, `primaryMuscles`, `secondaryMuscles`, `steps`, and `notes`. That makes it a good structured source for a first RAG prototype.

Secondary nutrition source:

- [google-research-datasets/Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)

The Nutrition5k README describes dish metadata CSVs containing dish-level fields such as `dish_id`, calories, fat, carbohydrate, protein, and repeated ingredient fields. This project uses only that metadata and ignores images, videos, depth maps, and multimodal signals. The metadata files are fetched from the official Google Cloud bucket referenced by the dataset README.

Important exclusions:

- `sanidad.gob.es` is intentionally excluded
- no multimodal features are used
- no large-scale crawling is used
- no additional heavy datasets are included

Additional PDF source:

- a small Google Drive folder with diet and supplement PDFs used as an extra local knowledge source

## 4. Why only Nutrition5k metadata is used

Only metadata is used for three reasons:

1. it keeps the system lightweight and realistic for a laptop without GPU
2. it keeps the project easy to explain and reproduce
3. it allows nutrition-related retrieval without introducing image processing or multimodal complexity

This is a deliberate scope decision, not a limitation of the general RAG idea.

## 5. Why this is a first iteration

This repository is the first iteration of a larger academic direction. The objective is not maximum performance. The objective is to create:

- a correct baseline
- a transparent pipeline
- a modular codebase
- a clear foundation for later TFG extensions

That is why the system favors simple retrieval, a small embedding model, and a local persistent vector database.

## 6. Repository structure

```text
app/
data/
  raw/
    exercises/
    diets/
    nutrition5k/
      metadata/
  processed/
db/
conversations/
notebooks/
  02_llm_fit_model_selection.md
  llm_fit_results_template.csv
  03_prompt_engineering_experiments.md
  04_agent_tools.md
scripts/
  fetch_datasets.py
  ingest_diets.py
  ingest_exercises.py
  ingest_nutrition.py
  ingest_all.py
  build_index.py
  compare_prompting.py
  query_rag.py
  run_agent.py
src/
  agent/
    executor.py
    tools.py
  evaluation/
    dataset.py
    ragas_runner.py
    runner.py
    variants.py
  ingestion/
    diets.py
    exercises.py
    nutrition.py
  guarded_agent/
    executor.py
  guardrails/
    schemas.py
    service.py
    vendor.py
  processing/
    schemas.py
    jsonl.py
    documents.py
  embeddings/
    factory.py
  retrieval/
    vector_store.py
    pipeline.py
  router_agent/
    graph.py
    schemas.py
  llm/
    prompts.py
    local_model.py
  utils/
    paths.py
.env.example
README.md
requirements.txt
```

## 7. System design

### 7.1 Ingestion process

The ingestion layer reads both datasets and maps them into one shared schema called `NormalizedRecord`.

Exercise normalization:

- input: JSON files from `longhaul-fitness/exercises`
- output fields:
  - `id`
  - `source`
  - `exercise_name`
  - `category`
  - `primary_muscles`
  - `secondary_muscles`
  - `instructions`
  - `notes`
  - `tags`

Nutrition normalization:

- input: dish metadata CSVs from Nutrition5k
- output fields:
  - `id`
  - `source`
  - `ingredients`
  - `calories`
  - `protein`
  - `carbs`
  - `fat`

Both are stored as JSONL in `data/processed/` for transparent inspection and debugging.

Diet PDF normalization:

- input: local PDF files from the Google Drive folder
- output fields:
  - `id`
  - `source`
  - `title`
  - `category`
  - `notes`
  - `document_text`

PDF titles are cleaned before indexing so labels such as `Copia de ...` do not appear in the retrieved metadata or in the generated answer.

### 7.2 Document design

This project follows the explicit semantic unit strategy at the source-record level:

- 1 exercise = 1 document
- 1 dish = 1 document

Those semantic documents are then split into short retrieval chunks for the vector index. This keeps the original academic unit easy to justify while improving retrieval precision. Each chunk stores metadata about its parent document, so the system can show both the best-matching chunk and the source document it came from.

The document builder converts normalized records into structured natural-language documents for embedding, while preserving metadata separately for retrieval output and later filtering.

The same principle is applied to the PDF source before chunking:

- 1 PDF = 1 document

This keeps the implementation simple and makes the source traceable during retrieval.

### 7.3 Embeddings

Embedding model:

- `sentence-transformers/all-MiniLM-L6-v2`

Why this model:

- lightweight
- widely used
- CPU-friendly
- good enough for a small semantic retrieval baseline

This choice matches the project constraint of running locally on a modest laptop.

### 7.4 Vector database

Vector store:

- ChromaDB

Why Chroma:

- local and persistent
- simple Python integration
- suitable for small and medium academic experiments
- easy to explain compared with heavier infrastructure

The database is stored locally in `db/`.

### 7.5 Semantic similarity

The retrieval stage uses semantic similarity in embedding space. The Chroma collection is configured with cosine similarity (`hnsw:space=cosine`) so that the nearest vectors represent semantically closer documents according to the MiniLM embedding space.

This is a core concept in the project:

- lexical matching looks for exact words
- semantic similarity looks for meaning-level closeness

That is why a query such as "exercises for glutes" can retrieve documents like `Glute Bridge` even if the wording is not identical.

### 7.6 Top-k retrieval

The system retrieves:

- `k = 3`

Why `k=3`:

- it keeps the context small
- it reduces prompt noise
- it is easy to justify and debug
- it is appropriate for a first lightweight baseline

The repository now supports two retrieval modes:

- `similarity`
  - standard cosine-similarity retrieval
- `mmr`
  - Maximal Marginal Relevance retrieval, which tries to balance relevance and diversity among the retrieved chunks

For MMR, the current CLI exposes:

- `k = 3`
- `fetch_k = 20`
- `lambda_mult = 1.0`

When documents are shown in the CLI, the system prints the parent title, relevant metadata, and a short preview of the retrieved chunk content. This makes retrieval behavior easier to inspect and aligns with the didactic style used in class notebooks.

### 7.7 LLM layer

The generation layer is designed for a local Ollama model. The default recommendation is:

- Phi-3 Mini or another lightweight local model

The code also auto-detects a locally available Ollama model if the preferred one is not installed.

Auxiliary note:

- LLM-Fit can be used as a preliminary model-selection aid, but it is not part of the runtime pipeline

### 7.7.1 Agent tools

Besides the standard RAG pipeline, the repository includes a small local agent inspired by the class example. The goal is not to replace retrieval, but to show how a local LLM can decide when to use project-specific tools.

The agent uses:

- `ChatOllama`
- `create_agent` from LangChain
- two lightweight tools connected to the project domain

Included tools:

- `search_fitness_knowledge`
  - searches the local Chroma knowledge base and returns the most relevant documents
- `fitness_calculator`
  - evaluates simple expressions such as BMI, calorie arithmetic, macro totals, or training volume

This design makes sense for the project because one tool retrieves domain knowledge and the other performs useful numeric support tasks. It stays small, explainable, and aligned with a first local academic iteration.

### 7.7.2 Guardrails

The repository also includes a modular guardrails layer designed for later evaluation experiments. This layer is inspired by the class notebook approach, but it is integrated directly into the local RAG pipeline using the current modular structure.

The first guardrail version includes:

- an input scope guardrail based on Guardrails AI `RestrictToTopic`, combined with lightweight heuristic checks
- a structured output guardrail based on Pydantic models so the final answer follows a controlled schema

For future evaluation, the intended baseline is `few-shot`. That means guardrail experiments are run on top of the same retrieval pipeline while keeping `few-shot` as the reference prompting strategy.

Besides the standard RAG pipeline, the repository includes a simple local agent inspired by the class example. This agent is built with:

- `ChatOllama` for the local chat model
- `create_agent` from LangChain
- two local tools connected to the project domain

The two tools are:

- `search_fitness_knowledge`
- `fitness_calculator`

The first tool queries the same local Chroma knowledge base used by the RAG system, which allows the agent to retrieve exercise or nutrition information on demand. The second tool performs simple calculations useful in the domain, such as BMI, calorie arithmetic, macro totals, or training volume formulas.

This agent layer does not replace the main RAG flow. It complements it and makes it possible to demonstrate a second interaction mode based on tool use, which is especially useful for the practical and academic part of the project.

### 7.7.2 Conversation evidence

The query and agent scripts save simple conversation evidence under `conversations/`.

Each saved conversation includes:

- `conversation.json`
- `conversation.txt`

This is useful for annexes, screenshots, and documenting practical tests with the LLM.

### 7.8 Prompting strategy

The prompt is intentionally restrictive. It tells the model to:

- answer only from retrieved context
- avoid invention
- say `"Not enough information in the retrieved documents."` when needed
- keep the answer concise and factual

This is important in a university setting because it makes the reasoning path more controllable and the limitations more explicit.

### 7.9 Prompt engineering experiments

On top of the main RAG pipeline, the repository now includes a small prompt engineering layer for controlled experiments.

Available strategies:

- `zero-shot`
- `one-shot`
- `few-shot`
- `chain-of-thought`

These strategies do not modify ingestion, embeddings, Chroma, or retrieval. They only modify how the retrieved context is presented to the LLM, which makes them suitable for academic comparison.

## 8. Resource-aware design

This project is intentionally resource-aware:

- CPU-first design
- no GPU assumption
- lightweight embedding model
- local persistent vector DB
- no multimodal pipeline
- no reranking stage
- no heavy orchestration framework

A smaller system is better for this first local version because it is faster to understand, easier to debug, and easier to defend academically.

## 9. Limitations

Current limitations include:

- retrieval quality depends strongly on metadata quality
- no reranker is used
- no evaluation benchmark is implemented yet
- nutrition data is limited to metadata and not broader dietary guidance
- the local LLM can still produce generic phrasing even when grounded
- no explicit filtering by muscle group, category, or macronutrient has been added yet

## 10. Risks

Main risks in this first iteration:

- retrieval errors can surface irrelevant but semantically nearby documents
- the model can still overgeneralize from a small set of retrieved records
- dish identifiers are less interpretable than human-friendly meal names
- answers may be correct but too generic if the retrieved context is sparse

## 11. Future work

Planned future improvements:

- multimodal extension with images
- stronger retrievers
- reranking
- explicit filtering by metadata
- quantitative evaluation
- prompt comparison
- multiple local LLM comparison
- better nutrition-specific sources
- conversational memory and user personalization

## 12. Setup

### 12.1 Create the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### 12.2 Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 12.3 Install and check Ollama

```powershell
ollama list
```

If needed:

```powershell
ollama pull phi3
```

### 12.4 Optional guardrails validator setup

If you want to use the Guardrails AI topic validator exactly as configured in this project, run:

```powershell
.\.venv\Scripts\python scripts\install_guardrails_validator.py
```

This installs the validator in a short external path to avoid Windows + OneDrive path-length issues.

## 13. How to run locally

### Option A: download all source files automatically

```powershell
.\.venv\Scripts\python scripts\fetch_datasets.py
```

This downloads:

- `strength.json`
- `cardio.json`
- `flexibility.json`
- `dish_metadata_cafe1.csv`
- `dish_metadata_cafe2.csv`
- the shared Google Drive diet PDFs into `data/raw/diets/`

### Option B: place the raw files manually

Expected locations:

- `data/raw/exercises/`
- `data/raw/nutrition5k/metadata/`
- `data/raw/diets/`

### Quick run in one command

If you want to do everything in one go:

```powershell
.\.venv\Scripts\python scripts\run_pipeline.py
```

That command:

- downloads the lightweight dataset files
- normalizes both sources
- builds the Chroma index
- runs a final validation query

You can also skip the download step if the raw files are already present:

```powershell
.\.venv\Scripts\python scripts\run_pipeline.py --skip-fetch
```

And you can choose the validation question:

```powershell
.\.venv\Scripts\python scripts\run_pipeline.py --question "Which exercises target the chest?"
```

### Step 1: normalize raw data

```powershell
.\.venv\Scripts\python scripts\ingest_all.py
```

Or separately:

```powershell
.\.venv\Scripts\python scripts\ingest_diets.py
.\.venv\Scripts\python scripts\ingest_exercises.py
.\.venv\Scripts\python scripts\ingest_nutrition.py
```

### Step 2: build the vector database

```powershell
.\.venv\Scripts\python scripts\build_index.py
```

### Step 3: query the local RAG system

Single question:

```powershell
.\.venv\Scripts\python scripts\query_rag.py --question "Which exercises target the glutes?"
```

Single question using the PDF source:

```powershell
.\.venv\Scripts\python scripts\query_rag.py --question "What do the diet PDFs say about supplement risks?"
```

Interactive mode:

```powershell
.\.venv\Scripts\python scripts\query_rag.py
```

Interactive mode with a specific prompt strategy:

```powershell
.\.venv\Scripts\python scripts\query_rag.py --strategy few-shot
```

Single question with a specific prompt strategy:

```powershell
.\.venv\Scripts\python scripts\query_rag.py --strategy zero-shot --question "Which exercises target the glutes?"
```

Single question with guardrails enabled. If no strategy is passed, the system uses `few-shot` as the guardrail baseline:

```powershell
.\.venv\Scripts\python scripts\query_rag.py --guardrails --question "Which exercises target the glutes?"
```

Single question with MMR retrieval:

```powershell
.\.venv\Scripts\python scripts\query_rag.py --strategy few-shot --retrieval-mode mmr --question "Which exercises target the glutes?"
```

Example of an out-of-scope query blocked by guardrails:

```powershell
.\.venv\Scripts\python scripts\query_rag.py --guardrails --question "Who won the last football world cup?"
```

Compare all prompt strategies for the same question:

```powershell
.\.venv\Scripts\python scripts\compare_prompting.py --question "Which exercises target the glutes?"
```

Run the local agent with tools:

```powershell
.\.venv\Scripts\python scripts\run_agent.py --question "Find exercises for glutes and calculate the BMI for 80 kg and 1.78 m"
```

Run the guarded agent variant:

```powershell
.\.venv\Scripts\python scripts\run_guarded_agent.py --question "Find exercises for glutes and calculate the BMI for 80 kg and 1.78 m"
```

Run the router agent variant:

```powershell
.\.venv\Scripts\python scripts\run_router_agent.py --question "Which exercises target the glutes?"
```

Run a RAGAS evaluation pass over the prepared CSV:

```powershell
.\.venv\Scripts\python scripts\evaluate_ragas.py --max-rows 1
```

Notes about evaluation:

- the evaluation CSV is `rag_eval.csv`
- the default baseline is `few_shot_baseline`
- the script evaluates the currently integrated variants: baseline, guardrails, MMR, router agent, and guarded agent
- results are saved under `evaluation_results/` as both `ragas_summary.csv` and `ragas_summary.md`
- the detailed per-question outputs are stored inside each variant folder
- the script first attempts RAGAS scoring with a local Ollama evaluator and, if a metric times out or fails to parse, it fills that score with a lightweight local fallback so the final comparison table is always complete on a CPU-only laptop

Recommended incremental evaluation workflow:

1. start with one variant, one metric, and one row

```powershell
.\.venv\Scripts\python scripts\evaluate_ragas.py --variants few_shot_baseline --metrics answer_relevancy --max-rows 1
```

2. continue on the same run directory and add another metric

```powershell
.\.venv\Scripts\python scripts\evaluate_ragas.py --variants few_shot_baseline --metrics factual_correctness --max-rows 1 --resume-run-dir evaluation_results/ragas_eval_YYYYMMDD_HHMMSS
```

3. evaluate one complete variant before moving to the next one

```powershell
.\.venv\Scripts\python scripts\evaluate_ragas.py --variants few_shot_mmr --metrics answer_relevancy faithfulness context_recall factual_correctness --max-rows 1
```

Practical time note:

- on a CPU-only laptop, one metric for one variant and one row can still take a few minutes
- one full pass over all variants, all metrics, and multiple rows can take a long time
- for that reason, the recommended workflow is to evaluate in parts and progressively complete the same summary table

Saved conversation evidence:

- every query creates a folder inside `conversations/`
- each folder contains `conversation.txt` and `conversation.json`
- this works for both the RAG script and the local agent
- the saved files include retrieved document metadata and a short preview of each retrieved document

You can also run:

```powershell
.\.venv\Scripts\python app\main.py
```

## 14. Environment variables

Example values are included in `.env.example`:

- `OLLAMA_MODEL`
- `EMBEDDING_MODEL`
- `TOP_K`
- `CHROMA_DB_PATH`
- `EXERCISES_DATA_DIR`
- `NUTRITION_DATA_DIR`
- `DIETS_DATA_DIR`
- `DIETS_DRIVE_FOLDER_URL`

## 15. What has been validated in this workspace

This repository was executed locally in this workspace with:

- real exercise JSON files from `longhaul-fitness/exercises`
- real Nutrition5k dish metadata CSVs from the official dataset storage
- real diet and supplement PDFs downloaded from the shared Google Drive folder
- normalized outputs for both sources
- a persisted Chroma database
- successful retrieval and answer generation with a local Ollama model

Observed local validation results:

- 404 exercise records normalized
- 5006 nutrition records normalized
- 5 diet PDF records normalized
- 5415 documents indexed in Chroma
- successful example queries for chest and glute exercises

The repository also includes two academic support artifacts:

- `notebooks/02_llm_fit_model_selection.md` for documenting the LLM selection process
- `notebooks/03_prompt_engineering_experiments.md` for documenting prompt engineering comparisons
- `notebooks/04_agent_tools.md` for documenting the local agent and its tools

## 16. Notes

- Hugging Face may warn about unauthenticated downloads if `HF_TOKEN` is not set. This does not block execution.
- On Windows, Hugging Face may warn about symlink cache behavior. This also does not block execution.
- The root files `ingest.py` and `query.py` belong to the earlier prototype and are kept only as legacy reference.
