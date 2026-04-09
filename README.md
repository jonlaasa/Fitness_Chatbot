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
    nutrition5k/
      metadata/
  processed/
db/
notebooks/
  02_llm_fit_model_selection.md
  llm_fit_results_template.csv
  03_prompt_engineering_experiments.md
scripts/
  fetch_datasets.py
  ingest_exercises.py
  ingest_nutrition.py
  ingest_all.py
  build_index.py
  compare_prompting.py
  query_rag.py
src/
  ingestion/
    exercises.py
    nutrition.py
  processing/
    schemas.py
    jsonl.py
    documents.py
  embeddings/
    factory.py
  retrieval/
    vector_store.py
    pipeline.py
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

### 7.2 Document design

This project follows the explicit semantic unit strategy:

- 1 exercise = 1 document
- 1 dish = 1 document

No generic chunking is used. This is important academically because the document boundaries are easy to justify. Each record is already a meaningful unit of knowledge, so chunking would add complexity without clear benefit in this first iteration.

The document builder converts normalized records into structured natural-language documents for embedding, while preserving metadata separately for retrieval output and later filtering.

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

### 7.7 LLM layer

The generation layer is designed for a local Ollama model. The default recommendation is:

- Phi-3 Mini or another lightweight local model

The code also auto-detects a locally available Ollama model if the preferred one is not installed.

Auxiliary note:

- LLM-Fit can be used as a preliminary model-selection aid, but it is not part of the runtime pipeline

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

## 13. How to run locally

### Option A: download the lightweight source files automatically

```powershell
.\.venv\Scripts\python scripts\fetch_datasets.py
```

This downloads:

- `strength.json`
- `cardio.json`
- `flexibility.json`
- `dish_metadata_cafe1.csv`
- `dish_metadata_cafe2.csv`

### Option B: place the raw files manually

Expected locations:

- `data/raw/exercises/`
- `data/raw/nutrition5k/metadata/`

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

Compare all prompt strategies for the same question:

```powershell
.\.venv\Scripts\python scripts\compare_prompting.py --question "Which exercises target the glutes?"
```

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

## 15. What has been validated in this workspace

This repository was executed locally in this workspace with:

- real exercise JSON files from `longhaul-fitness/exercises`
- real Nutrition5k dish metadata CSVs from the official dataset storage
- normalized outputs for both sources
- a persisted Chroma database
- successful retrieval and answer generation with a local Ollama model

Observed local validation results:

- 404 exercise records normalized
- 5006 nutrition records normalized
- 5410 documents indexed in Chroma
- successful example queries for chest and glute exercises

The repository also includes two academic support artifacts:

- `notebooks/02_llm_fit_model_selection.md` for documenting the LLM selection process
- `notebooks/03_prompt_engineering_experiments.md` for documenting prompt engineering comparisons

## 16. Notes

- Hugging Face may warn about unauthenticated downloads if `HF_TOKEN` is not set. This does not block execution.
- On Windows, Hugging Face may warn about symlink cache behavior. This also does not block execution.
- The root files `ingest.py` and `query.py` belong to the earlier prototype and are kept only as legacy reference.
