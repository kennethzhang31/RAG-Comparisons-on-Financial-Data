import os
import time
import chromadb
from chromadb.config import Settings
import cohere
import base64
from pdf2image import convert_from_path
import io
import uuid
import pickle
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import argparse

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for the embedding process."""
    model_name: str = "embed-v4.0"
    rate_limit_delay: float = 0.3
    input_type: str = "image"
    embedding_types: List[str] = None

    def __post_init__(self):
        if self.embedding_types is None:
            self.embedding_types = ["float"]

class CohereEmbedder:
    """A class to handle document embedding using Cohere."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedder with configuration."""
        self.config = config or EmbeddingConfig()
        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise EnvironmentError("Error: COHERE_API_KEY not found")
        
        self.client = cohere.ClientV2(api_key=self.api_key)
        self.total_tokens = 0

    def get_embedding(self, image: str, metadata: Dict[str, Any]) -> Tuple[List[float], int]:
        """Get embedding for an image."""
        try:
            response = self.client.embed(
                model=self.config.model_name,
                input_type=self.config.input_type,
                embedding_types=self.config.embedding_types,
                images=[image]
            )
            
            embeddings = response.embeddings.float_[0]
            tokens = response.meta.billed_units.image_tokens or 0
            
            logger.info(f"Successfully indexed {metadata['filename']} Page: {metadata['page']}")
            return embeddings, tokens
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    def process_pdf(self, pdf_path: str, qid: str, output_dir: str) -> Dict[str, Any]:
        """Process a single PDF and generate embeddings."""
        try:
            filename = Path(pdf_path).stem
            save_path = os.path.join(output_dir, f"{filename}.pkl")

            if os.path.exists(save_path):
                logger.info(f"{save_path} already exists. Skipping...")
                return {"status": "skipped", "reason": "file_exists"}

            logger.info(f"Processing PDF: {pdf_path}")
            images = convert_from_path(pdf_path)
            pdf_embeddings = []
            total_tokens = 0

            for idx, image in enumerate(images):
                try:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    stringified_buffer = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    image_base64 = f"data:image/png;base64,{stringified_buffer}"
                    
                    embedding, tokens = self.get_embedding(
                        image_base64, 
                        {"filename": filename, "page": idx}
                    )
                    
                    pdf_embeddings.append(embedding)
                    total_tokens += tokens
                    time.sleep(self.config.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"Failed to process page {idx}: {e}")
                    continue

            if pdf_embeddings:
                with open(save_path, "wb") as f:
                    pickle.dump({
                        "embeddings": pdf_embeddings,
                        "metadata": {
                            "filename": filename,
                            "qid": qid,
                            "num_pages": len(pdf_embeddings)
                        },
                        "embed_model": f"cohere-{self.config.model_name}"
                    }, f)
                logger.info(f"✅ Saved {filename}.pkl with {len(pdf_embeddings)} pages")
                return {"status": "success", "total_tokens": total_tokens}
            
            return {"status": "error", "reason": "no_embeddings_generated"}

        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            return {"status": "error", "error": str(e)}

def main():
    """Main function to process all PDFs."""
    parser = argparse.ArgumentParser(description="Embed PDFs with Cohere")
    parser.add_argument("-s", "--silent", action="store_true", help="Run in silent mode (no print)")
    args = parser.parse_args()

    if args.silent:
        logger.setLevel(logging.WARNING)

    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        raise EnvironmentError("Error: DATA_DIR not found")

    output_dir = os.path.join(os.getenv("EMBEDDING_STORAGE", "./"), "cohere_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    embedder = CohereEmbedder()
    total_stats = {"total_tokens": 0, "processed": 0, "errors": 0}

    for qid in os.listdir(data_dir):
        qid_path = os.path.join(data_dir, qid)
        for pdf in os.listdir(qid_path):
            if not pdf.endswith(".pdf"):
                continue

            pdf_path = os.path.join(qid_path, pdf)
            result = embedder.process_pdf(pdf_path, qid, output_dir)
            
            if result["status"] == "success":
                total_stats["total_tokens"] += result["total_tokens"]
                total_stats["processed"] += 1
            else:
                total_stats["errors"] += 1

    logger.info("\nProcessing Complete!")
    logger.info(f"Total Documents Processed: {total_stats['processed']}")
    logger.info(f"Total Errors: {total_stats['errors']}")
    logger.info(f"Total Tokens: {total_stats['total_tokens']}")
    logger.info(f"Estimated Cost: ${total_stats['total_tokens']/3014 * 0.00148:.4f}")

if __name__ == "__main__":
    main()