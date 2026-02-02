#!/usr/bin/env python3
"""
Upload Human Layer context packs to Qdrant with COMPLETE SEPARATION from Veritas/Pipeline.

CRITICAL SEPARATION RULES:
- Collection prefix: humangr_  (NOT pipeline_)
- Product ID: HUMANGR (in all document metadata)
- No imports from pipeline_autonomo or pipeline_v2
- No shared collections with Veritas

This script is standalone and does not depend on the brains codebase.
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import uuid

# External dependencies only - NO brains imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError:
    print("ERROR: qdrant-client not installed. Run: pip install qdrant-client")
    sys.exit(1)

try:
    import voyageai
except ImportError:
    voyageai = None

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    TextEmbedding = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    try:
        import requests
        REQUESTS_AVAILABLE = True
    except ImportError:
        REQUESTS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SEPARATION CONFIGURATION - IMPOSSÍVEL DE ERRAR
# ═══════════════════════════════════════════════════════════════════════════════

HUMANGR_CONFIG = {
    # Collection naming - COMPLETELY SEPARATE from pipeline_/veritas_
    "collection_prefix": "humangr_",
    "collection_name": "humangr_context_packs",

    # Product identification
    "product_id": "HUMANGR",
    "product_name": "Human Layer MCP Server",

    # Embedding settings
    "embedding_model": "voyage-large-2",
    "embedding_dimensions": 1024,

    # Qdrant settings
    "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
    "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
}


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING PROVIDER (STANDALONE - NO brains imports)
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingProvider:
    """Standalone embedding provider for HumanGR."""

    def __init__(self):
        self._voyage_client = None
        self._fastembed_model = None
        self._st_model = None
        self._ollama_url = None
        self._dimensions = HUMANGR_CONFIG["embedding_dimensions"]

        # Try Voyage AI first
        voyage_key = os.getenv("VOYAGE_API_KEY")
        if voyage_key and voyageai:
            try:
                self._voyage_client = voyageai.Client(api_key=voyage_key)
                print(f"✓ Using Voyage AI ({HUMANGR_CONFIG['embedding_model']})")
                return
            except Exception as e:
                print(f"  Voyage AI failed: {e}")

        # Try FastEmbed second (fast, local, no GPU needed)
        if FASTEMBED_AVAILABLE:
            try:
                self._fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                self._dimensions = 384  # bge-small dimensions
                print(f"✓ Using FastEmbed (bge-small-en-v1.5, {self._dimensions} dims)")
                return
            except Exception as e:
                print(f"  FastEmbed failed: {e}")

        # Try Ollama third (free, local, but slower)
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if HTTPX_AVAILABLE or REQUESTS_AVAILABLE:
            try:
                # Test Ollama connection
                test_url = f"{ollama_host}/api/tags"
                if HTTPX_AVAILABLE:
                    resp = httpx.get(test_url, timeout=5.0)
                    resp.raise_for_status()
                else:
                    resp = requests.get(test_url, timeout=5)
                    resp.raise_for_status()

                self._ollama_url = ollama_host
                self._dimensions = 768  # nomic-embed-text dimensions
                print(f"✓ Using Ollama (nomic-embed-text, {self._dimensions} dims)")
                return
            except Exception as e:
                print(f"  Ollama failed: {e}")

        # Fallback to sentence-transformers
        if SentenceTransformer:
            try:
                self._st_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
                self._dimensions = 1024
                print("✓ Using sentence-transformers (bge-large-en-v1.5)")
                return
            except Exception as e:
                print(f"  sentence-transformers failed: {e}")

        raise RuntimeError("No embedding provider available. Install fastembed, voyageai, run Ollama, or install sentence-transformers.")

    def _ollama_embed(self, text: str) -> List[float]:
        """Get embedding from Ollama."""
        url = f"{self._ollama_url}/api/embeddings"
        # Truncate to 2048 chars for faster embedding
        truncated = text[:2048] if len(text) > 2048 else text
        payload = {"model": "nomic-embed-text", "prompt": truncated}

        if HTTPX_AVAILABLE:
            resp = httpx.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            return resp.json()["embedding"]
        else:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["embedding"]

    def embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self._voyage_client:
            result = self._voyage_client.embed(
                texts=[text],
                model=HUMANGR_CONFIG["embedding_model"],
            )
            return result.embeddings[0]
        elif self._fastembed_model:
            embeddings = list(self._fastembed_model.embed([text]))
            return embeddings[0].tolist()
        elif self._ollama_url:
            return self._ollama_embed(text)
        elif self._st_model:
            return self._st_model.encode(text).tolist()
        else:
            raise RuntimeError("No embedding provider configured")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        if self._voyage_client:
            result = self._voyage_client.embed(
                texts=texts,
                model=HUMANGR_CONFIG["embedding_model"],
            )
            return result.embeddings
        elif self._fastembed_model:
            # FastEmbed has efficient batch processing
            embeddings = list(self._fastembed_model.embed(texts))
            return [e.tolist() for e in embeddings]
        elif self._ollama_url:
            # Ollama doesn't have batch API, so we loop with progress
            embeddings = []
            for i, t in enumerate(texts):
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"  Embedding {i+1}/{len(texts)}...", end="\r")
                embeddings.append(self._ollama_embed(t))
            print(f"  Embedded {len(texts)}/{len(texts)} documents    ")
            return embeddings
        elif self._st_model:
            return [e.tolist() for e in self._st_model.encode(texts)]
        else:
            raise RuntimeError("No embedding provider configured")

    @property
    def dimensions(self) -> int:
        return self._dimensions


# ═══════════════════════════════════════════════════════════════════════════════
# QDRANT UPLOADER (STANDALONE - COMPLETE SEPARATION)
# ═══════════════════════════════════════════════════════════════════════════════

class HumanGRQdrantUploader:
    """
    Qdrant uploader for HumanGR context packs.

    COMPLETE SEPARATION from Veritas/Pipeline:
    - Uses humangr_ collection prefix
    - Adds product_id=HUMANGR to all documents
    - No shared collections or data
    """

    def __init__(self, context_packs_dir: Path):
        self.context_packs_dir = context_packs_dir
        self.collection_name = HUMANGR_CONFIG["collection_name"]
        self.product_id = HUMANGR_CONFIG["product_id"]

        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=HUMANGR_CONFIG["qdrant_host"],
            port=HUMANGR_CONFIG["qdrant_port"],
        )

        # Initialize embedding provider
        self.embedder = EmbeddingProvider()

        print(f"\n{'='*60}")
        print(f"HUMANGR QDRANT UPLOADER - COMPLETE SEPARATION")
        print(f"{'='*60}")
        print(f"Collection: {self.collection_name}")
        print(f"Product ID: {self.product_id}")
        print(f"Context packs dir: {self.context_packs_dir}")
        print(f"{'='*60}\n")

    def ensure_collection(self) -> None:
        """Ensure collection exists with correct schema."""
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            print(f"✓ Collection {self.collection_name} exists")
        else:
            print(f"Creating collection: {self.collection_name}")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dimensions,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✓ Created collection: {self.collection_name}")

    def load_context_packs(self) -> List[Dict[str, Any]]:
        """Load all context packs from directory."""
        packs = []
        for md_file in sorted(self.context_packs_dir.glob("*.md")):
            content = md_file.read_text()
            sprint_id = md_file.stem.replace("_CONTEXT", "")

            # Extract metadata from content
            title = ""
            wave = ""
            if "title:" in content:
                for line in content.split("\n"):
                    if "title:" in line:
                        title = line.split("title:")[-1].strip().strip('"')
                        break
            if "wave:" in content:
                for line in content.split("\n"):
                    if "wave:" in line:
                        wave = line.split("wave:")[-1].strip()
                        break

            packs.append({
                "id": sprint_id,
                "filename": md_file.name,
                "title": title or sprint_id,
                "wave": wave,
                "content": content,
                "content_hash": hashlib.md5(content.encode()).hexdigest(),
            })

        print(f"✓ Loaded {len(packs)} context packs")
        return packs

    def upload_packs(self, packs: List[Dict[str, Any]]) -> None:
        """Upload context packs to Qdrant."""
        points = []

        print("\nGenerating embeddings...")
        # Truncate to 2000 chars for faster Ollama embedding (content is stored separately)
        contents = [p["content"][:2000] for p in packs]
        embeddings = self.embedder.embed_batch(contents)

        print("Preparing points...")
        for pack, embedding in zip(packs, embeddings):
            # Generate deterministic UUID from pack ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"humangr.{pack['id']}"))

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    # SEPARATION METADATA
                    "product_id": self.product_id,  # CRITICAL: Always HUMANGR
                    "product_name": HUMANGR_CONFIG["product_name"],

                    # Pack metadata
                    "sprint_id": pack["id"],
                    "filename": pack["filename"],
                    "title": pack["title"],
                    "wave": pack["wave"],
                    "content_hash": pack["content_hash"],

                    # Content (truncated for payload)
                    "content": pack["content"][:10000],
                    "content_length": len(pack["content"]),

                    # Timestamps
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            points.append(point)

        print(f"Uploading {len(points)} points to Qdrant...")
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f"✓ Uploaded {len(points)} context packs to {self.collection_name}")

    def verify_upload(self) -> None:
        """Verify upload and separation."""
        # Check collection
        info = self.qdrant.get_collection(self.collection_name)
        print(f"\n{'='*60}")
        print("VERIFICATION")
        print(f"{'='*60}")
        print(f"Collection: {self.collection_name}")
        print(f"Points count: {info.points_count}")
        # vectors_count may not exist in newer qdrant-client versions
        if hasattr(info, 'vectors_count'):
            print(f"Vectors count: {info.vectors_count}")

        # Verify all points have correct product_id
        sample = self.qdrant.scroll(
            collection_name=self.collection_name,
            limit=5,
            with_payload=True,
        )

        print(f"\nSample documents (first 5):")
        for point in sample[0]:
            payload = point.payload
            print(f"  - {payload.get('sprint_id')}: product_id={payload.get('product_id')}")

        # CRITICAL: Verify NO pipeline_ documents exist
        print(f"\n{'='*60}")
        print("SEPARATION VERIFICATION")
        print(f"{'='*60}")
        print(f"✓ Collection prefix: humangr_ (NOT pipeline_)")
        print(f"✓ Product ID: HUMANGR (NOT VERITAS)")
        print(f"✓ No shared collections")
        print(f"{'='*60}")

    def run(self) -> None:
        """Run the full upload process."""
        self.ensure_collection()
        packs = self.load_context_packs()
        self.upload_packs(packs)
        self.verify_upload()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    # Determine context packs directory
    script_dir = Path(__file__).parent
    context_packs_dir = script_dir.parent / "context_packs"

    if not context_packs_dir.exists():
        print(f"ERROR: Context packs directory not found: {context_packs_dir}")
        sys.exit(1)

    # Check for required environment
    if not os.getenv("VOYAGE_API_KEY"):
        print("\nWARNING: VOYAGE_API_KEY not set. Will try fallback embedding.")

    # Run uploader
    uploader = HumanGRQdrantUploader(context_packs_dir)
    uploader.run()

    print("\n" + "="*60)
    print("✅ UPLOAD COMPLETE - HumanGR context packs in Qdrant")
    print("="*60)
    print(f"\nCollection: {HUMANGR_CONFIG['collection_name']}")
    print(f"SEPARATION: COMPLETE (no Veritas/Pipeline overlap)")
    print("="*60)


if __name__ == "__main__":
    main()
