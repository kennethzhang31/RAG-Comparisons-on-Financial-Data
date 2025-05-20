import json
from byaldi import RAGMultiModalModel

# IMPORTANT, 
# eval data 的架構為
# eval_data
#       |---|-qid1/(*.pdf)  <- pdf 是source的PDF
#           |-qid2/(*.pdf)
# 這個架構與 rag_cohere、rag_gemini、rag_me5Large 不相同

# TODO: 需要重新設計indexing流程，使其符合 rag_cohere、rag_gemini、rag_me5Large 的index格式

def load_data(questions_path: str, ground_truth_path: str):
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)['questions']
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)["ground_truths"]

    questions = {q['qid']: q for q in questions_data}
    gts = {gt['qid']: gt for gt in ground_truth_data}

    return questions, gts

def log_result(qid, question, gt_id, retrieved_ids, is_correct, filepath="./results_byaldi.jsonl"):
    result = {
        "qid": qid,
        "query": question,
        "gt_id": gt_id,
        "retrieved_ids": retrieved_ids,
        "is_correct": is_correct
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

def main():
    # Paths to evaluation data
    gt_path = "/Users/kennethzhang/rag_testing_master/ground_truths.json"
    q_path = "/Users/kennethzhang/rag_testing_master/questions.json"

    # Load questions and ground truths
    questions, ground_truths = load_data(q_path, gt_path)

    # Evaluate each question
    for qid in range(1, 101):
        # Initialize RAG model
        RAG_index = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0")
        
        # Index the documents
        RAG_index.index(
            input_path=f"/Users/kennethzhang/largit_work/byaldi_implementation/eval_data/{qid}/",
            index_name=f"qid{qid}",
            store_collection_with_index=False,
            overwrite=True
        )

        # Get query and search
        query = questions[qid]['query']
        results = RAG_index.search(query=query, k=3)

        # Process results
        doc_to_ids = RAG_index.get_doc_ids_to_file_names()
        res_ids = [doc_to_ids[res.doc_id] for res in results]
        res = []

        for res_id in res_ids:
            res_id = res_id.split(".")[0]
            res_id = res_id.split("/")[-1]
            res.append(res_id)

        # Check if correct and log result
        is_correct = str(ground_truths[qid]['retrieve']) in res
        print(f"{qid}: gt = {ground_truths[qid]['retrieve']}, result = {res}")
        log_result(qid, query, ground_truths[qid]['retrieve'], res, is_correct)
        
        # Clean up
        del RAG_index

if __name__ == "__main__":
    main() 