from google import genai
import os
import time
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from google.api_core import exceptions
import pickle
import json
import re
# from langchain.text_splitter import MarkdownHeaderTextSplitter
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownElementNodeParser
)
import tiktoken
from llama_index.core.schema import Document 
from llama_index.core.settings import Settings
import logging
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import argparse

load_dotenv()
Settings.llm = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for the embedding process."""
    chunk_size: int = 1024
    chunk_overlap: int = 100
    rate_limit_delay: float = 0.5
    model_name: str = 'models/gemini-embedding-exp-03-07'

class GeminiEmbedder:
    """A class to handle document embedding using Gemini."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedder with configuration."""
        self.config = config or EmbeddingConfig()
        self.api_key = os.getenv("VERTEX_GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("Error: API KEY NOT FOUND")
        
        self.client = genai.Client(api_key=self.api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.parser = MarkdownElementNodeParser(
            nested_node_parser=SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        )

    def get_embedding(self, content: str) -> List[float]:
        """Get embedding for a piece of content."""
        try:
            response = self.client.models.embed_content(
                model=self.config.model_name,
                contents=[content]
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoding.encode(text))

    def process_document(self, md_path: str, meta_path: str, output_path: str) -> Dict[str, Any]:
        """Process a single document and generate embeddings."""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            toc = metadata.get("computed_toc")
            has_valid_toc = isinstance(toc, list) and len(toc) > 0

            if not content.strip():
                logger.warning(f"Skipping empty file: {md_path}")
                return {"status": "skipped", "reason": "empty_file"}

            embeddings = []
            total_tokens = 0
            max_tokens = 0

            if has_valid_toc:
                nodes = self.parser.extract_elements(content)
                logger.info(f"Total Chunks: {len(nodes)}")

                for i, node in enumerate(nodes):
                    section_text = self._extract_section_text(node)
                    if not section_text:
                        continue

                    try:
                        embedding = self.get_embedding(section_text)
                        estimated_tokens = self.count_tokens(section_text)
                        
                        embeddings.append({
                            "filename": Path(md_path).stem,
                            "metadata": {"node": i},
                            "embedding": embedding,
                            "text": section_text
                        })
                        
                        total_tokens += estimated_tokens
                        max_tokens = max(max_tokens, estimated_tokens)
                        
                        logger.info(f"Embedded {node.type} chunk {i+1}/{len(nodes)} | Tokens: {estimated_tokens}")
                        time.sleep(self.config.rate_limit_delay)

                    except Exception as e:
                        logger.error(f"Embedding failed on chunk {i+1}: {e}")
                        raise

            else:
                embedding = self.get_embedding(content)
                estimated_tokens = self.count_tokens(content)
                
                embeddings.append({
                    "filename": Path(md_path).stem,
                    "metadata": {"node": 0},
                    "embedding": embedding,
                    "text": content
                })
                
                total_tokens = estimated_tokens
                max_tokens = estimated_tokens
                logger.info(f"Embedded full document (no TOC) | Tokens: {estimated_tokens}")
                time.sleep(self.config.rate_limit_delay)

            with open(output_path, "wb") as f:
                pickle.dump(embeddings, f)

            return {
                "status": "success",
                "total_tokens": total_tokens,
                "max_tokens": max_tokens,
                "chunks": len(embeddings)
            }

        except Exception as e:
            logger.error(f"Failed to process {md_path}: {e}")
            return {"status": "error", "error": str(e)}

    def _extract_section_text(self, node) -> Optional[str]:
        """Extract text from a node based on its type."""
        if node.type == "text":
            return node.element.strip()
        elif node.type == "table" and node.table is not None:
            return node.table.to_string(index=False)
        elif node.type == "table_text":
            return node.element.strip()
        return None

def main():
    """Main function to process all documents."""
    parser = argparse.ArgumentParser(description="Embed Markdown files with Gemini")
    parser.add_argument("-s", "--silent", action="store_true", help="Run in silent mode (no print)")
    args = parser.parse_args()

    if args.silent:
        logger.setLevel(logging.WARNING)

    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        raise EnvironmentError("Error: DATA_DIR not found")

    output_dir = os.path.join(os.getenv("EMBEDDING_STORAGE", "./"), "gemini_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    embedder = GeminiEmbedder()
    folder_list = [f for f in os.listdir(data_dir) if f != ".DS_Store"]
    total_stats = {"total_tokens": 0, "max_tokens": 0, "processed": 0, "errors": 0}

    for idx, folder in enumerate(folder_list, 1):
        md_folder = os.path.join(data_dir, folder)
        md_path = os.path.join(md_folder, f"{folder}.md")
        meta_path = os.path.join(md_folder, "metadata.json")
        output_path = os.path.join(output_dir, f"{folder}.pkl")

        if not os.path.exists(md_path) or not os.path.exists(meta_path):
            logger.warning(f"{md_path} does not exist! Skipping")
            continue

        if os.path.exists(output_path):
            logger.info(f"{output_path} already exists. skipping...")
            continue

        logger.info(f"Processing: {md_path:<40} | {idx}/{len(folder_list)}")
        
        result = embedder.process_document(md_path, meta_path, output_path)
        
        if result["status"] == "success":
            total_stats["total_tokens"] += result["total_tokens"]
            total_stats["max_tokens"] = max(total_stats["max_tokens"], result["max_tokens"])
            total_stats["processed"] += 1
        else:
            total_stats["errors"] += 1

    logger.info("\nProcessing Complete!")
    logger.info(f"Total Documents Processed: {total_stats['processed']}")
    logger.info(f"Total Errors: {total_stats['errors']}")
    logger.info(f"Total Tokens: {total_stats['total_tokens']}")
    logger.info(f"Max Tokens per Document: {total_stats['max_tokens']}")

if __name__ == "__main__":
    main()