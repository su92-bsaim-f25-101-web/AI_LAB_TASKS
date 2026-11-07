import re
import csv
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
    EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
except Exception:
    EMB_MODEL = None

class ContextRetriever:
    def __init__(self, items, mode="tfidf"):
        if not items or any(not x["text"].strip() for x in items):
            raise ValueError("Records must contain text")
        self.items = items
        self.mode = mode
        raw_texts = [x["text"] for x in items]
        if mode == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.vectorizer.fit_transform(raw_texts)
        elif mode == "embedding":
            if EMB_MODEL is None:
                raise RuntimeError("Embedding backend unavailable")
            self.matrix = EMB_MODEL.encode(raw_texts, normalize_embeddings=True)
        else:
            raise ValueError("Mode must be tfidf or embedding")

    def query_search(self, prompt, k=3):
        if not prompt.strip() or k <= 0:
            return []
        if self.mode == "tfidf":
            vec = self.vectorizer.transform([prompt])
            if vec.nnz == 0:
                return []
            scores = (vec @ self.matrix.T).toarray()[0]
        else:
            vec = EMB_MODEL.encode([prompt], normalize_embeddings=True)[0]
            scores = self.matrix @ vec
        sorted_indices = sorted(range(len(self.items)), key=lambda i: (-float(scores[i]), i))
        return [(self.items[i], float(scores[i])) for i in sorted_indices[:k]]

def word_boundaries(content):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", content)]

def sliding_windows(content, window_size=40, step_overlap=10, base_offset=0):
    if window_size <= 0 or not 0 <= step_overlap < window_size:
        raise ValueError("Invalid window or overlap parameters")
    positions = word_boundaries(content)
    chunk_spans = []
    step = window_size - step_overlap
    for i in range(0, len(positions), step):
        j = min(i + window_size, len(positions))
        chunk_spans.append((base_offset + positions[i][0], base_offset + positions[j - 1][1]))
        if j == len(positions):
            break
    return chunk_spans

def sentence_aware_split(content, max_limit=40):
    if max_limit <= 0:
        raise ValueError("Limit must be positive")
    break_points = [0] + [m.end() for m in re.finditer(r"[.!?]\s+", content)] + [len(content)]
    blocks, current_chunk, current_count = [], None, 0
    for start, end in zip(break_points, break_points[1:]):
        phrase = content[start:end]
        token_len = len(phrase.split())
        if not token_len:
            continue
        if token_len > max_limit:
            if current_chunk is not None:
                blocks.append(current_chunk)
            blocks.extend(sliding_windows(phrase, max_limit, 0, start))
            current_chunk, current_count = None, 0
        elif current_chunk is not None and current_count + token_len <= max_limit:
            current_chunk = (current_chunk[0], end)
            current_count += token_len
        else:
            if current_chunk is not None:
                blocks.append(current_chunk)
            current_chunk, current_count = (start, end), token_len
    if current_chunk is not None:
        blocks.append(current_chunk)
    return blocks

def generate_chunks_from_docs(documents, scheme):
    output_chunks = []
    for doc_key, body in documents.items():
        if scheme == "sentence":
            spans = sentence_aware_split(body)
        else:
            spans = sliding_windows(body, 40, int(scheme))
        for idx, (a, b) in enumerate(spans):
            output_chunks.append({
                "id": f"{doc_key}:{idx}",
                "doc": doc_key,
                "start": a,
                "end": b,
                "text": body[a:b]
            })
    return output_chunks

CORPUS_DATA = {
    "late": "Late submission of course assignments is penalized by ten percent daily. Students facing illness may apply for an extension if supported by verified medical evidence before the deadline. Late requests without documents are rejected.",
    "password": "IT Help Desk Guide: Accounts. Reset a forgotten password using the portal recovery link. Accounts are locked after five failed sign-in attempts and unlock automatically after 30 minutes. Never share credentials with anyone.",
    "assignment": "Assignments must be submitted in PDF format via the student portal before 11:59 PM. Late uploads trigger automated penalty point deductions. Group projects require a single submission from the team lead.",
    "contractor": "Contractors may work remotely only if their explicit signed agreement allows remote access. Standard employee remote working policies do not apply to external contractors.",
    "printer": "Printer Troubleshooting Sheet. For blank printer pages, check ink cartridge levels and run the automatic head cleaning cycle. Ensure protective tape is removed before installation."
}

QUERY_ITEMS = [
    {"query": "What is required for illness extensions?", "doc": "late", "evidence": "extension if supported by verified medical evidence", "split": "dev"},
    {"query": "How do I reset a forgotten password?", "doc": "password", "evidence": "Reset a forgotten password using the portal recovery link", "split": "dev"},
    {"query": "Where do I upload assignments?", "doc": "assignment", "evidence": "submitted in PDF format via the student portal", "split": "dev"},
    {"query": "Can contractors work remotely?", "doc": "contractor", "evidence": "Contractors may work remotely only if their explicit signed agreement", "split": "dev"},
    {"query": "What do I do for blank printer pages?", "doc": "printer", "evidence": "check ink cartridge levels and run the automatic head cleaning cycle", "split": "test"},
    {"query": "How can I recover my login access?", "doc": "password", "evidence": "Reset a forgotten password using the portal recovery link", "split": "test"},
    {"query": "How should I hand in my coursework?", "doc": "assignment", "evidence": "submitted in PDF format via the student portal", "split": "test"},
    {"query": "What proof is needed when I was ill?", "doc": "late", "evidence": "extension if supported by verified medical evidence", "split": "test"},
    {"query": "What are tomorrow's flight prices?", "doc": None, "evidence": None, "split": "test"},
    {"query": "Who won the football match?", "doc": None, "evidence": None, "split": "test"}
]

def task1_execute():
    print("============================================================")
    print("TASK 1: TOKENIZATION & FAQ CHATBOT ANALYSIS")
    print("Student Name: Abdul Moiz | Roll No: BSAIM-F25-101")
    print("============================================================")
    test_strings = [
        "AI, search!",
        "I can't log in.",
        "help@example.com",
        "CS4944 version 2.1",
        "",
        "یہ ایک مثال ہے"
    ]
    for text in test_strings:
        sp_toks = text.split()
        rx_toks = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
        print(f"Raw Text    : {repr(text)}")
        print(f"Whitespace  : {sp_toks} ({len(sp_toks)})")
        print(f"Word Regex  : {rx_toks} ({len(rx_toks)})\n")

    faq_db = [
        {"id": "faq:password", "question": "How do I reset my password?", "answer": "Use the recovery link on the portal page.", "source": "it_help.txt"},
        {"id": "faq:submit", "question": "Where do I submit my assignment?", "answer": "Upload to the learning portal.", "source": "assessment.txt"},
        {"id": "faq:late", "question": "Can illness justify a late submission?", "answer": "Request extension with medical proof.", "source": "handbook.txt"}
    ]
    records = [dict(r, text=r["question"] + " " + r["answer"]) for r in faq_db]
    search_engine = ContextRetriever(records, "tfidf")

    def faq_respond(user_input):
        if not user_input.strip():
            return "Please enter a valid question."
        for item in faq_db:
            if user_input.strip().casefold() == item["question"].casefold():
                return item["answer"] + f"}]"
        hits = search_engine.query_search(user_input)
        if not hits:
            return "No matching FAQ found."
        response_lines = ["Possible matching questions:"]
        for r, score in hits:
            response_lines.append(f" - {r['question']} (similarity: {score:.3f})")
        return "\n".join(response_lines)

    sample_prompts = ["How do I reset my password?", "password recovery", "", "volcano geology"]
    for prompt in sample_prompts:
        print(f"User Question: {repr(prompt)}")
        print(f"Bot Output:\n{faq_respond(prompt)}\n" + "-" * 50)

def task2_execute():
    print("============================================================")
    print("TASK 2: CHUNKING & SEMANTIC SUGGESTIONS")
    print("Student Name: Abdul Moiz | Roll No: BSAIM-F25-101")
    print("============================================================")
    for scheme in ["0", "10", "sentence"]:
        generated = generate_chunks_from_docs(CORPUS_DATA, scheme)
        print(f"Strategy [{scheme}]: Produced {len(generated)} chunks.")

def calculate_metrics(similarity_matrix, chunk_list, query_list):
    hits, reciprocal_ranks = [], []
    for row, item in zip(similarity_matrix, query_list):
        if item["doc"] is None:
            continue
        top_ranks = np.argsort(-row, kind="stable")[:3]
        matched = [r for r, idx in enumerate(top_ranks, 1) if chunk_list[idx]["doc"] == item["doc"] and item["evidence"] in chunk_list[idx]["text"]]
        hits.append(bool(matched))
        reciprocal_ranks.append(1 / matched[0] if matched else 0)
    return float(np.mean(hits)), float(np.mean(reciprocal_ranks))

def task3_execute():
    print("============================================================")
    print("TASK 3: EVALUATION & BENCHMARKING")
    print("Student Name: Abdul Moiz | Roll No: BSAIM-F25-101")
    print("============================================================")
    prompts = [q["query"] for q in QUERY_ITEMS]
    log_records = []
    for scheme in ["0", "10", "sentence"]:
        chunks = generate_chunks_from_docs(CORPUS_DATA, scheme)
        chunk_texts = [c["text"] for c in chunks]
        for mode in ["tfidf", "embedding"]:
            if mode == "embedding" and EMB_MODEL is None:
                continue
            t_start = time.perf_counter()
            if mode == "tfidf":
                vec = TfidfVectorizer()
                C_mat = vec.fit_transform(chunk_texts)
                Q_mat = vec.transform(prompts)
            else:
                C_mat = EMB_MODEL.encode(chunk_texts, normalize_embeddings=True)
                Q_mat = EMB_MODEL.encode(prompts, normalize_embeddings=True)
            enc_time = time.perf_counter() - t_start
            
            t_start = time.perf_counter()
            sim_scores = (Q_mat @ C_mat.T).toarray() if mode == "tfidf" else Q_mat @ C_mat.T
            search_time = time.perf_counter() - t_start

            for split_name in ["dev", "test"]:
                subset_indices = [i for i, q in enumerate(QUERY_ITEMS) if q["split"] == split_name]
                h_val, mrr_val = calculate_metrics(sim_scores[subset_indices], chunks, [QUERY_ITEMS[i] for i in subset_indices])
                print(f"Strategy: {scheme:<8} | Mode: {mode:<9} | Split: {split_name:<4} | Hit@3: {h_val:.3f} | MRR@3: {mrr_val:.3f} | Chunks: {len(chunks)}")

            for row_scores, q_item in zip(sim_scores, QUERY_ITEMS):
                for rank_pos, c_idx in enumerate(np.argsort(-row_scores, kind="stable")[:3], 1):
                    target_chunk = chunks[c_idx]
                    log_records.append({
                        "strategy": scheme, "method": mode, "query": q_item["query"],
                        "split": q_item["split"], "rank": rank_pos, "score": round(float(row_scores[c_idx]), 4),
                        "chunk_id": target_chunk["id"], "doc": target_chunk["doc"],
                        "start": target_chunk["start"], "end": target_chunk["end"],
                        "answerable": q_item["doc"] is not None, "text": target_chunk["text"]
                    })

    with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_records[0].keys()))
        writer.writeheader()
        writer.writerows(log_records)
    print("\nSaved evaluation results to retrieval_log.csv successfully.")

def main():
    task1_execute()
    task2_execute()
    task3_execute()

if __name__ == "__main__":
    main()