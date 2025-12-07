# Multi-Hop Question Answering with Entailment Verification

This project implements a multi-hop question answering (QA) system over a news corpus, combining dense retrieval, entailment verification, and answer generation using LLMs. It supports multi-hop reasoning, evidence chain construction, and evaluation against a gold dataset.

## Features
- **Dense Retrieval:** Finds relevant documents using bi-encoder embeddings.
- **Multi-Hop Reasoning:** Expands queries over multiple hops for complex questions.
- **Entailment Verification:** Uses a cross-encoder NLI model to verify evidence chains.
- **LLM Answer Generation:** Generates concise answers using a LLaMA-based model.
- **API Interface:** FastAPI endpoint for querying the system.
- **Evaluation Metrics:** Computes exact match, evidence recall, and entailment scores.

## Setup Instructions
1. **Clone the repository** and place your corpus and dataset files in the root directory:
   - `corpus.json` (news articles)
   - `MultiHopRAG.json` (multi-hop QA dataset)
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include: `torch`, `transformers`, `fastapi`, `uvicorn`, `numpy`, `pydantic`.
3. **Set environment variables:**
   - `HF_TOKEN`: Your HuggingFace token for model downloads.
   - `LLAMA_MODEL_NAME`: (Optional) Override default LLaMA model.
4. **Run the server:**
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`.

## API Usage
### Endpoint: `/ask`
- **Method:** POST
- **Request Body:**
  ```json
  {
    "query": "<your question>",
    "top_k": 5,
    "chain_length": 2,
    "entailment_threshold": 0.7,
    "eval_mode": true,
    "num_hops": 2
  }
  ```
- **Response:**
  - `predicted_answer`: The system's answer
  - `chains`: Evidence chains with entailment scores
  - `metrics`: Evaluation metrics (if gold answer exists)
  - `retrieval_steps`: Details of each retrieval hop

## Example Workflow
1. **Ask a question:**
   - The system retrieves relevant documents over multiple hops.
   - Constructs evidence chains and verifies them via entailment.
   - Generates a concise answer using LLM.
   - Returns supporting evidence and metrics.

## Evaluation
- If `MultiHopRAG.json` contains gold answers/evidence, metrics are computed automatically.
- Metrics include exact match, evidence recall, and average entailment score.

## Screenshots & Demo
Below are sample screenshots and a demo video illustrating the workflow:

| Screenshot | Description |
|------------|-------------|
| ![Screenshot 1](Screenshot 2025-12-07 at 4.55.55 PM.png) | API request example |
| ![Screenshot 2](Screenshot 2025-12-07 at 4.56.14 PM.png) | Retrieval results |
| ![Screenshot 3](Screenshot 2025-12-07 at 4.56.36 PM.png) | Evidence chains |
| ![Screenshot 4](Screenshot 2025-12-07 at 4.56.44 PM.png) | Metrics and answer |

**Demo Video:**
[Project Demo - Video.mp4](Project Demo - Video.mp4)

## File Structure
- `main.py`: Main API and logic
- `corpus.json`: News corpus
- `MultiHopRAG.json`: Multi-hop QA dataset
- `doc_embeddings.npy`: Cached document embeddings
- `Project Demo - Video.mp4`: Demo video
- `Screenshot ... .png`: Screenshots

## Citation
If you use this project, please cite appropriately.

---
For questions or issues, please contact the project maintainer.
