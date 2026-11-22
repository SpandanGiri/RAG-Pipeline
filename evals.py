import json
import time
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from main import load_rag, PROMPT


EVAL_PATH = Path("eval.json")
RESULTS_PATH = Path("eval_results.json")
MAX_GEN_EVALS = 10   # Start with 10 to stay comfortably under free-tier limits
SLEEP_BETWEEN_LLM = 4  # Seconds between Gemini calls

JUDGE_PROMPT = """You are grading a RAG system's answer against a ground-truth reference.
Question: {question}
Expected Answer: {expected}
Generated Answer: {generated}

Respond ONLY with valid JSON:
{{"verdict": "CORRECT" or "INCORRECT", "reason": "one short sentence"}}
"""


def evaluate():
    eval_data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    retriever, llm = load_rag()

    judge_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


    hits = 0
    mrr_total = 0.0
    correct_answers = 0
    gen_tested = 0
    results = []

    for i,item in enumerate(eval_data):
        qt = item['question']
        expected_file = item["expected_source"]
        print(qt,expected_file)

        retrieved_docs = retriever.invoke(qt)
        retrieved_files = [Path(d.metadata.get("source", "")).name for d in retrieved_docs]

        hit = expected_file in retrieved_files

        hits+=int(hit)

        rank = (retrieved_files.index(expected_file) + 1) if hit else None
        mrr_total += (1.0 / rank) if rank else 0.0

        record = {
                    "id": item["id"],
                    "question": qt,
                    "expected_source": expected_file,
                    "retrieved_sources": retrieved_files,
                    "retrieval_hit": hit,
                    "rank": rank,
                }

        if gen_tested < MAX_GEN_EVALS:
            context = "\n\n---\n\n".join(d.page_content for d in retrieved_docs)
            messages = PROMPT.format_messages(context=context, question=qt)

            gen_answer = llm.invoke(messages).text

            judge_resp = judge_llm.invoke(
                            JUDGE_PROMPT.format(
                                question=qt,
                                expected=item["expected_answer"],
                                generated=gen_answer,
                            )
                        ).text



            cleaned = judge_resp.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                verdict_data = json.loads(cleaned)
            except json.JSONDecodeError:
                verdict_data = {"verdict": "PARSE_ERROR", "reason": judge_resp}

            if verdict_data.get("verdict") == "CORRECT":
                correct_answers += 1

            record["generated_answer"] = gen_answer
            record["expected_answer"] = item["expected_answer"]
            record["judge"] = verdict_data
            gen_tested += 1
            time.sleep(SLEEP_BETWEEN_LLM)

        results.append(record)
        print(f"[{i+1}/{len(eval_data)}] Hit: {hit} (Rank: {rank})")


        total = len(eval_data)
        print("\n=== BASELINE EVAL RESULTS ===")
        print(f"Retrieval Recall@k : {hits}/{total} ({hits/total:.1%})")
        print(f"Retrieval MRR      : {mrr_total/total:.3f}")
        if gen_tested:
            print(f"Answer Correctness : {correct_answers}/{gen_tested} ({correct_answers/gen_tested:.1%})")

        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nDetailed breakdown saved to {RESULTS_PATH}")



if __name__=="__main__":
    evaluate()