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
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import asyncio
from dotenv import load_dotenv
import argparse

Settings.llm = None

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
    chunk_size: int = 512
    chunk_overlap: int = 50
    model_name: str = 'intfloat/multilingual-e5-large-instruct'
    max_concurrent_tasks: int = 5
    device: str = 'mps'

class ME5LargeEmbedder:
    """A class to handle document embedding using ME5 Large model."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedder with configuration."""
        self.config = config or EmbeddingConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name).to(self.config.device)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.parser = MarkdownElementNodeParser(
            nested_node_parser=SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        )
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Average pool the hidden states."""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    async def get_embedding_async(self, text: str) -> List[List[float]]:
        """Get embedding for a piece of content with async processing."""
        tokens = self.tokenizer.tokenize(text)
        token_count = len(tokens)
        
        if token_count > self.config.chunk_size:
            logger.info(f"[Split] {token_count} tokens. Splitting...")
            embeddings = []
            i = 0
            while i < token_count:
                chunk_tokens = tokens[i:i + self.config.chunk_size]
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                inputs = self.tokenizer(
                    chunk_text,
                    max_length=self.config.chunk_size,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    emb = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                    emb = F.normalize(emb, p=2, dim=1)
                    embeddings.append(emb[0].cpu().numpy().tolist())
                i += self.config.chunk_size - self.config.chunk_overlap
            return embeddings
        else:
            inputs = self.tokenizer(
                text,
                max_length=self.config.chunk_size,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                emb = F.normalize(emb, p=2, dim=1)
                return [emb[0].cpu().numpy().tolist()]

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoding.encode(text))

    async def process_document(self, md_path: str, meta_path: str, output_path: str) -> Dict[str, Any]:
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

            if has_valid_toc:
                nodes = self.parser.extract_elements(content)
                logger.info(f"Total Chunks: {len(nodes)}")

                async def embed_node(i: int, node) -> List[Dict[str, Any]]:
                    section_text = self._extract_section_text(node)
                    if not section_text:
                        return []

                    result = await self.get_embedding_async(section_text)
                    out = []
                    for j, emb in enumerate(result):
                        out.append({
                            "filename": Path(md_path).stem,
                            "metadata": {"node": i, "subchunk": j},
                            "embedding": emb,
                            "text": self.tokenizer.convert_tokens_to_string(
                                self.tokenizer.tokenize(section_text)[
                                    j * (self.config.chunk_size - self.config.chunk_overlap):
                                    (j + 1) * self.config.chunk_size
                                ]
                            )
                        })
                    logger.info(f"Embedded chunk {i+1}/{len(nodes)} | Tokens: {self.count_tokens(section_text)}")
                    return out

                embed_tasks = [embed_node(i, node) for i, node in enumerate(nodes)]
                results = await asyncio.gather(*embed_tasks)
                for group in results:
                    embeddings.extend(group)
            else:
                result = await self.get_embedding_async(content)
                for j, emb in enumerate(result):
                    embeddings.append({
                        "filename": Path(md_path).stem,
                        "metadata": {"node": 0, "subchunk": j},
                        "embedding": emb,
                        "text": self.tokenizer.convert_tokens_to_string(
                            self.tokenizer.tokenize(content)[
                                j * (self.config.chunk_size - self.config.chunk_overlap):
                                (j + 1) * self.config.chunk_size
                            ]
                        )
                    })
                logger.info(f"Embedded full document (no TOC) | Tokens: {self.count_tokens(content)}")

            with open(output_path, "wb") as f:
                pickle.dump(embeddings, f)

            return {
                "status": "success",
                "total_tokens": total_tokens,
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

async def main():
    """Main function to process all documents."""
    parser = argparse.ArgumentParser(description="Embed Markdown files with ME5 Large")
    parser.add_argument("-s", "--silent", action="store_true", help="Run in silent mode (no print)")
    args = parser.parse_args()

    if args.silent:
        logger.setLevel(logging.WARNING)

    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        raise EnvironmentError("Error: DATA_DIR not found")

    output_dir = os.path.join(os.getenv("EMBEDDING_STORAGE", "./"), "me5_large_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    embedder = ME5LargeEmbedder()
    folder_list = [f for f in os.listdir(data_dir) if f != ".DS_Store"]
    total_stats = {"total_tokens": 0, "processed": 0, "errors": 0}

    async def process_folder(folder: str, idx: int, n: int):
        async with embedder.semaphore:
            md_folder = os.path.join(data_dir, folder)
            md_path = os.path.join(md_folder, f"{folder}.md")
            meta_path = os.path.join(md_folder, "metadata.json")
            output_path = os.path.join(output_dir, f"{folder}.pkl")

            if not os.path.exists(md_path) or not os.path.exists(meta_path):
                logger.warning(f"{md_path} does not exist! Skipping")
                return

            if os.path.exists(output_path):
                logger.info(f"{output_path} already exists. skipping...")
                return

            logger.info(f"Processing: {md_path:<40} | {idx}/{n}")
            
            result = await embedder.process_document(md_path, meta_path, output_path)
            
            if result["status"] == "success":
                total_stats["total_tokens"] += result["total_tokens"]
                total_stats["processed"] += 1
            else:
                total_stats["errors"] += 1

    # Schedule all tasks
    tasks = [
        asyncio.create_task(process_folder(folder, idx + 1, len(folder_list)))
        for idx, folder in enumerate(folder_list)
    ]

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    logger.info("\nProcessing Complete!")
    logger.info(f"Total Documents Processed: {total_stats['processed']}")
    logger.info(f"Total Errors: {total_stats['errors']}")
    logger.info(f"Total Tokens: {total_stats['total_tokens']}")

if __name__ == "__main__":
    asyncio.run(main())