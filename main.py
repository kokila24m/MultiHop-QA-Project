import os
import json
import math
from itertools import combinations, islice
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
)

# ===========================
# Config
# ===========================

CORPUS_PATH = "corpus.json"          # your news corpus
MULTIHOP_PATH = "MultiHopRAG.json"   # your multihop dataset
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
ENTAILMENT_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"  # MNLI-style

LLAMA_MODEL_NAME = os.environ.get(
    "LLAMA_MODEL_NAME",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # small & CPU-friendly
)

# IMPORTANT: don't hardcode your HF token in code
HF_TOKEN = os.environ.get("HF_TOKEN", None)

EMBEDDINGS_PATH = "doc_embeddings.npy"   # cached dense embeddings

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_CHARS = 3000          # to keep LLM prompt manageable
MAX_CHAIN_TEXT_CHARS = 3000
MAX_CHAINS_PER_QUERY = 64         # safety limit to avoid explosion


# ===========================
# Data models for API
# ===========================

class QuestionRequest(BaseModel):
    query: str
    top_k: int = 5
    chain_length: int = 2         # for ablation: 1, 2, 3, ...
    entailment_threshold: float = 0.7
    eval_mode: bool = True        # if True, compute accuracy if gold exists
    num_hops: int = 1             # how many retrieval hops (1 = old behavior)


class EvidenceChain(BaseModel):
    doc_indices: List[int]
    titles: List[str]
    entailment_score: float
    supporting: bool
    concatenated_text: Optional[str] = None


class RetrievalStep(BaseModel):
    hop: int
    queries: List[str]
    retrieved_indices: List[int]
    retrieved_titles: List[str]


class AnswerResponse(BaseModel):
    query: str
    predicted_answer: str
    gold_answer: Optional[str]
    chains: List[EvidenceChain]
    metrics: Dict[str, float]
    # FIXED: List[RetrievalStep], not List<RetrievalStep>
    retrieval_steps: Optional[List[RetrievalStep]] = None  # show intermediate hops


# ===========================
# Utility: text normalization
# ===========================

def normalize_answer(s: str) -> str:
    """Very simple normalization for EM-style evaluation."""
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ===========================
# Load corpus and dataset
# ===========================

print("Loading corpus...")
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus: List[Dict[str, Any]] = json.load(f)

doc_texts: List[str] = []
for d in corpus:
    title = d.get("title", "")
    body = d.get("body", "")
    full_text = (title or "") + "\n\n" + (body or "")
    doc_texts.append(full_text)

print(f"Loaded {len(doc_texts)} documents from corpus.")

print("Loading MultiHopRAG dataset...")
multihop_data: List[Dict[str, Any]] = []
if os.path.exists(MULTIHOP_PATH):
    with open(MULTIHOP_PATH, "r", encoding="utf-8") as f:
        multihop_data = json.load(f)
else:
    print(f"Warning: {MULTIHOP_PATH} not found. Eval will be limited.")

# index MultiHopRAG by query string for quick lookup
query2gold: Dict[str, Dict[str, Any]] = {}
for item in multihop_data:
    q = item.get("query", "").strip()
    if q:
        query2gold[q] = item

print(f"Indexed {len(query2gold)} MultiHopRAG queries.")


# ===========================
# Embedding model (bi-encoder)
# ===========================

print("Loading embedding model...")
embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(DEVICE)
embed_model.eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # (batch, seq, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


@torch.no_grad()
def encode_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """
    Encode texts into dense embeddings with simple progress logging.
    Also used for query encoding (len(texts) == 1).
    """
    all_embeddings = []
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    print(f"[Embeddings] Total texts: {total}, batch_size: {batch_size}, total_batches: {total_batches}")

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = texts[start:end]

        print(f"[Embeddings] Processing batch {batch_idx + 1}/{total_batches} "
              f"(items {start + 1}–{end})")

        encoded_input = embed_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)
        model_output = embed_model(**encoded_input)
        emb = mean_pooling(model_output, encoded_input["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().numpy())

    print("[Embeddings] Finished encoding all batches.")
    return np.vstack(all_embeddings)


# ===========================
# Prepare / cache corpus embeddings
# ===========================

print("Preparing corpus embeddings...")

if os.path.exists(EMBEDDINGS_PATH):
    # Try to load precomputed embeddings
    doc_embeddings = np.load(EMBEDDINGS_PATH)
    if doc_embeddings.shape[0] != len(doc_texts):
        print(
            f"Warning: embedding file has {doc_embeddings.shape[0]} rows, "
            f"but corpus has {len(doc_texts)} docs. Recomputing embeddings..."
        )
        doc_embeddings = encode_texts(doc_texts)
        np.save(EMBEDDINGS_PATH, doc_embeddings)
        print(f"Recomputed and saved embeddings to {EMBEDDINGS_PATH}:", doc_embeddings.shape)
    else:
        print(f"Loaded precomputed embeddings from {EMBEDDINGS_PATH}:", doc_embeddings.shape)
else:
    # First run: compute and save
    print("No precomputed embeddings found. Encoding now (one-time cost)...")
    doc_embeddings = encode_texts(doc_texts)
    np.save(EMBEDDINGS_PATH, doc_embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_PATH}:", doc_embeddings.shape)


# ===========================
# Cross-encoder for entailment
# ===========================

print("Loading cross-encoder NLI model...")
nli_tokenizer = AutoTokenizer.from_pretrained(ENTAILMENT_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(ENTAILMENT_MODEL_NAME).to(DEVICE)
nli_model.eval()
ENTAILMENT_LABEL_INDEX = 2  # typical MNLI ordering: [contradiction, neutral, entailment]


@torch.no_grad()
def entailment_probability(premise: str, hypothesis: str) -> float:
    inputs = nli_tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)
    logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return float(probs[ENTAILMENT_LABEL_INDEX].item())


# ===========================
# LLaMA for answer generation
# ===========================

print("Loading LLaMA model (may be large)...")
llama_tokenizer = AutoTokenizer.from_pretrained(
    LLAMA_MODEL_NAME,
    use_auth_token=HF_TOKEN
)
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    use_auth_token=HF_TOKEN,
)
llama_pipeline = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)


def generate_answer_with_llama(query: str, context_docs: List[str]) -> str:
    # Concatenate top documents as context
    context = "\n\n".join(context_docs)
    context = context[:MAX_CONTEXT_CHARS]

    prompt = (
        "You are an expert multi-hop question answering system.\n"
        "Use the provided news/article context to answer the question concisely.\n"
        "If the answer is not clearly stated in the context, reply with the single word 'unknown'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer (few words, no explanation):"
    )

    outputs = llama_pipeline(
        prompt,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
        eos_token_id=llama_tokenizer.eos_token_id,
        truncation=True,
    )
    full_text = outputs[0]["generated_text"]
    generated_part = full_text[len(prompt):]
    # take first line as answer
    answer = generated_part.strip().split("\n")[0].strip()
    # FIXED: strip(), not trip()
    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()
    return answer


# ===========================
# Retrieval helpers
# ===========================

def dense_retrieval(query: str, top_k: int = 5) -> List[int]:
    """
    Single-hop dense retrieval.
    """
    q_emb = encode_texts([query])[0]  # (dim,)
    scores = np.dot(doc_embeddings, q_emb)  # cosine (since normalized)
    top_idx = np.argsort(-scores)[:top_k]
    return top_idx.tolist()


def rerank_indices(query: str, candidate_indices: List[int], top_m: int) -> List[int]:
    """
    Given a set of candidate doc indices, rerank them by similarity to query.
    """
    if not candidate_indices:
        return []
    q_emb = encode_texts([query])[0]
    sub_embs = doc_embeddings[candidate_indices]
    sims = np.dot(sub_embs, q_emb)
    order = np.argsort(-sims)
    ordered = [candidate_indices[i] for i in order]
    return ordered[:top_m]


def multi_hop_retrieval(
    query: str,
    num_hops: int,
    top_k: int,
    expansion_per_hop: int = 3,
) -> Tuple[List[int], List[RetrievalStep]]:
    """
    Simple iterative multi-hop retrieval:
    - Hop 1: use original query
    - Later hops: use titles of top docs from previous hop as new queries
    - Returns:
        - union of all retrieved doc indices
        - retrieval_steps: what happened at each hop
    """
    num_hops = max(1, num_hops)
    retrieval_steps: List[RetrievalStep] = []
    all_doc_indices: set = set()

    current_queries = [query]

    for hop in range(num_hops):
        hop_indices_set: set = set()
        # retrieve for each query in this hop
        for q in current_queries:
            idxs = dense_retrieval(q, top_k=top_k)
            hop_indices_set.update(idxs)

        hop_indices = list(hop_indices_set)
        all_doc_indices.update(hop_indices)
        hop_titles = [corpus[i].get("title", "") or "" for i in hop_indices]

        retrieval_steps.append(
            RetrievalStep(
                hop=hop + 1,
                queries=current_queries,
                retrieved_indices=hop_indices,
                retrieved_titles=hop_titles,
            )
        )

        # build next-hop queries from titles of top-k docs of this hop
        if hop == num_hops - 1:
            break

        expansion_queries: List[str] = []
        ranked_for_expansion = rerank_indices(query, hop_indices, expansion_per_hop)
        for i in ranked_for_expansion:
            t = corpus[i].get("title", "")
            if t:
                expansion_queries.append(t)

        if not expansion_queries:
            break
        current_queries = expansion_queries

    return list(all_doc_indices), retrieval_steps


# ===========================
# Chain construction
# ===========================

def build_evidence_chains(
    top_indices: List[int],
    chain_length: int,
    max_chains: int = MAX_CHAINS_PER_QUERY,
) -> List[List[int]]:
    if chain_length <= 1:
        return [[i] for i in top_indices[:max_chains]]

    all_combos = combinations(top_indices, chain_length)
    chains = list(islice(all_combos, max_chains))
    return [list(c) for c in chains]


def concat_chain_text(doc_indices: List[int]) -> str:
    parts = []
    for idx in doc_indices:
        d = corpus[idx]
        title = d.get("title", "")
        body = d.get("body", "")
        parts.append(f"[{title}]\n{body}")
    text = "\n\n".join(parts)
    return text[:MAX_CHAIN_TEXT_CHARS]


# ===========================
# Evaluation metrics
# ===========================

def compute_metrics(
    query: str,
    predicted_answer: str,
    chains: List[EvidenceChain],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    gold_item = query2gold.get(query.strip())
    if not gold_item:
        # no gold: adhoc inference mode
        return metrics

    gold_answer = gold_item.get("answer", "")
    norm_pred = normalize_answer(predicted_answer)
    norm_gold = normalize_answer(gold_answer)

    metrics["exact_match"] = 1.0 if norm_pred == norm_gold and norm_gold != "" else 0.0

    # evidence recall based on titles
    gold_titles = set()
    for ev in gold_item.get("evidence_list", []):
        t = ev.get("title", "")
        if t:
            gold_titles.add(t.lower().strip())

    sys_titles = set()
    for ch in chains:
        if not ch.supporting:
            continue
        for t in ch.titles:
            if t:
                sys_titles.add(t.lower().strip())

    if gold_titles:
        metrics["evidence_title_recall"] = len(gold_titles & sys_titles) / len(gold_titles)
    else:
        metrics["evidence_title_recall"] = 0.0

    # average entailment of supporting chains
    support_scores = [c.entailment_score for c in chains if c.supporting]
    if support_scores:
        metrics["avg_support_entailment"] = float(sum(support_scores) / len(support_scores))
    else:
        metrics["avg_support_entailment"] = 0.0

    return metrics


# ===========================
# FastAPI app
# ===========================

app = FastAPI(title="Multi-Hop QA with Entailment Verification")


@app.post("/ask", response_model=AnswerResponse)
def answer_question(req: QuestionRequest):
    query = req.query.strip()
    print(f"\n=== New query ===\n{query}\n")

    # 1. Multi-hop retrieval
    all_indices, retrieval_steps = multi_hop_retrieval(
        query=query,
        num_hops=req.num_hops,
        top_k=req.top_k,
    )
    print(f"Multi-hop union doc indices: {all_indices}")

    # 2. Rerank union for context + chains
    top_indices = rerank_indices(query, all_indices, req.top_k)
    print(f"Top indices after rerank: {top_indices}")

    # 3. Use top docs as context for LLaMA answer
    context_docs = [doc_texts[i] for i in top_indices]
    predicted_answer = generate_answer_with_llama(query, context_docs)
    print(f"Predicted answer: {predicted_answer}")

    # 4. Chain-of-evidence construction
    chains_indices = build_evidence_chains(top_indices, chain_length=req.chain_length)

    # 5. Entailment verification for each chain
    hypoth = f"The answer to the question '{query}' is '{predicted_answer}'."
    chains_out: List[EvidenceChain] = []

    for chain_idx_list in chains_indices:
        c_text = concat_chain_text(chain_idx_list)
        score = entailment_probability(c_text, hypoth)
        titles = [corpus[i].get("title", "") for i in chain_idx_list]
        supporting = score >= req.entailment_threshold

        chains_out.append(
            EvidenceChain(
                doc_indices=chain_idx_list,
                titles=titles,
                entailment_score=score,
                supporting=supporting,
                concatenated_text=c_text,
            )
        )

    # 6. Evaluation metrics (if gold available)
    gold_item = query2gold.get(query)
    gold_answer = gold_item.get("answer") if gold_item else None

    metrics = compute_metrics(query, predicted_answer, chains_out) if req.eval_mode else {}

    return AnswerResponse(
        query=query,
        predicted_answer=predicted_answer,
        gold_answer=gold_answer,
        chains=chains_out,
        metrics=metrics,
        retrieval_steps=retrieval_steps,
    )


@app.get("/")
def root():
    return {
        "message": "Multi-Hop QA with Entailment Verification",
        "endpoints": ["/ask"],
    }


if __name__ == "__main__":
    # Run: python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
