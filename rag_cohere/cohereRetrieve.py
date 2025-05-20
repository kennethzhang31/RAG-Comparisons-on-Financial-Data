import os
import pickle
import chromadb
import json
from chromadb.config import Settings
import uuid
import cohere
from tqdm import tqdm
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class CohereRetriever:
    """A class to handle document retrieval using Cohere embeddings and ChromaDB."""
    
    def __init__(self, chroma_storage: str = None, pkl_dir: str = None):
        """Initialize the retriever with necessary configurations."""
        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise EnvironmentError("Error: COHERE_API_KEY not found")

        self.client = cohere.ClientV2(api_key=self.api_key)
        self.embed_model_name = "embed-v4.0"
        
        # Configure storage paths
        self.chroma_storage = chroma_storage or os.path.join(os.getenv("CHROMA_STORAGE", "./"), "cohere_collection")
        self.pkl_dir = pkl_dir or os.path.join(os.getenv("EMBEDDING_STORAGE", "./"), "cohere_embeddings")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_storage,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.is_indexed = os.path.exists(self.chroma_storage) and len(os.listdir(self.chroma_storage)) > 1

    def index_documents(self) -> None:
        """Index documents from pickle files into ChromaDB collections."""
        if self.is_indexed:
            logger.info("Collection already indexed.")
            return

        logger.info("Collection empty...")
        for fname in tqdm(os.listdir(self.pkl_dir), desc="Indexing collections"):
            if not fname.endswith(".pkl"):
                continue
                
            doc_id = fname.replace(".pkl", "")
            try:
                collection = self.chroma_client.get_or_create_collection(
                    name=f"{str(doc_id)}_db",
                    metadata={"hnsw:space": "cosine"}
                )

                filepath = os.path.join(self.pkl_dir, fname)
                with open(filepath, "rb") as f:
                    pkl_data = pickle.load(f)

                embeddings = pkl_data["embeddings"]
                meta = pkl_data["metadata"]
                filename = meta["filename"]
                num_pages = meta["num_pages"]

                for i in range(num_pages):
                    collection.add(
                        embeddings=[embeddings[i]],
                        metadatas=[{
                            "filename": filename,
                            "page": i
                        }],
                        ids=[str(uuid.uuid4())]
                    )
            except Exception as e:
                logger.error(f"Error indexing document {fname}: {e}")

        logger.info("All pickle files loaded into individual Chroma collections.")

    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query using Cohere model."""
        try:
            response = self.client.embed(
                texts=[query],
                model=self.embed_model_name,
                input_type="classification",
                embedding_types=["float"]
            )
            return response.embeddings.float_[0]
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise

    def retrieve_documents(self, query: str, sources: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        query_embedding = self.get_query_embedding(query)
        results = []

        for source_id in sources:
            collection_name = str(source_id)
            try:
                collection = self.chroma_client.get_collection(
                    name=f"{collection_name}_db"
                )
                res = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    include=["metadatas", "distances", "documents"]
                )
                if res["metadatas"]:
                    results.append({
                        "source": source_id,
                        "filename": res["metadatas"][0][0].get("filename", ""),
                        "page": res["metadatas"][0][0].get("page", 0),
                        "score": res["distances"][0][0],
                        "metadata": res["metadatas"][0][0]
                    })
            except Exception as e:
                logger.error(f"Could not load collection {collection_name}: {e}")
                continue

        return sorted(results, key=lambda x: x['score'])[:top_k]

def main():
    """Main function to run the retrieval evaluation."""
    retriever = CohereRetriever()
    retriever.index_documents()

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
            ranked = retriever.retrieve_documents(query, sources)
            top_sources = [str(r["source"]) for r in ranked]
            ground_truth = gt_map.get(qid)

            logger.info(f"\nQID {qid} — {query}")
            logger.info(f"Top 3 Matches: {top_sources} | GT: {ground_truth}")
            logger.info(f"RESULT: {ground_truth in top_sources}")

            count += 1
            if ground_truth in top_sources:
                correct += 1
            else:
                mistakes.append(qid)
        except Exception as e:
            logger.error(f"Error processing question {qid}: {e}")
            continue

    print()
    logger.info(f"Accuracy: {correct/count * 100:.2f}%")
    logger.info(f"Mistakes: {mistakes}")

if __name__ == "__main__":
    main()