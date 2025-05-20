from google import genai
import os
import pickle
import chromadb
import json
from chromadb.config import Settings
import uuid
from transformers import AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Tuple
from geminiRetrieve import GeminiRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JinaReranker:
    """A class to handle document reranking using Jina's reranker model."""
    
    def __init__(self):
        """Initialize the reranker with the Jina model."""
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'jinaai/jina-reranker-v2-base-multilingual',
            torch_dtype="auto",
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def rerank(self, query: str, docs: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Rerank documents based on their relevance to the query."""
        sentence_pairs = [[query, doc] for doc in docs]
        scores = self.model.compute_score(sentence_pairs, max_length=1024)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

def main():
    """Main function to run the retrieval and reranking evaluation."""
    # Initialize retriever and reranker
    retriever = GeminiRetriever()
    reranker = JinaReranker()
    
    # Load questions and ground truths
    with open("./questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open("./ground_truths.json", "r", encoding="utf-8") as f:
        gt = json.load(f)

    # Create ground truth mapping
    gt_map = {str(gt_['qid']): str(gt_["retrieve"]) for gt_ in gt['ground_truths']}

    count = 0
    correct = 0
    mistakes = []

    # Process questions in range 50-99
    for idx, q in enumerate(data['questions']):
        if idx < 50 or idx > 99:
            continue

        qid = str(q['qid'])
        sources = q['source']
        query = q['query']

        try:
            # Get initial retrieval results
            ranked = retriever.retrieve_documents(query, sources)
            maxsim_len = len(ranked)
            texts_for_rerank = [r["text"] for r in ranked]

            # Rerank the results
            reranked = reranker.rerank(query, texts_for_rerank, top_k=3)

            # Map reranked texts back to sources
            reranked_sources = []
            for text, _ in reranked:
                for r in ranked:
                    if r["text"] == text:
                        reranked_sources.append(str(r["source"]))
                        break

            ground_truth = gt_map.get(qid)

            logger.info(f"\nQID {qid} — {query}")
            logger.info(f"MaxSim original length: {maxsim_len}")
            logger.info(f"Top 3 Reranked Matches: {reranked_sources} | GT: {ground_truth}")
            logger.info(f"RESULT: {ground_truth in reranked_sources}")

            count += 1
            if ground_truth in reranked_sources:
                correct += 1
            else:
                mistakes.append(qid)
        except Exception as e:
            logger.error(f"Error processing question {qid}: {e}")
            continue

    print()
    logger.info(f"\nAccuracy: {correct/count * 100:.2f}%")
    logger.info(f"Mistakes: {mistakes}")

if __name__ == "__main__":
    main()