# Standard lib
from dataclasses import dataclass, field, asdict
from typing import Any
import time
from pathlib import Path
import uuid

# 3rd lib: Docling
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    TripletTableSerializer,
)
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.serializer.markdown import MarkdownParams
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

# 3rd: Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, UpdateStatus

# 3rd: LiteLLM
import litellm

# Custom lib
from src.utils import logger


@dataclass
class Chunk:
    """
    Chunk class: result of chunking document, that holds the serialized text
    and its metadata.
    """

    serialized: str
    metadata: dict[str, Any]


@dataclass
class VectorPayload:
    """
    Vector payload class.

    Attributes:
        file_id: File ID. A file ID can be either an absolute local file path (maybe the path of an temp file),
        or ID from a local/cloud blob storage, basically anything that can uniquely identify that document.

        disease: Disease. The disease to be associated with the document.

        chunk_content: Chunk content. The content of the chunk, in serialized format.

        chunk_metadata: Chunk metadata. The metadata of the chunk, in serialized (JSON) format.
        Currently have two fields: pages (list[int]) contains all the pages where the current chunk reference,
        and filename (str): document filename
    """

    file_id: str
    disease: str
    chunk_content: str = ""
    chunk_metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"""
        --- 

        Disease: {self.disease}

        ## Document content

        {self.chunk_content}

        ## Document metadata

        {self.chunk_metadata}

        ---
        """


class SerializerProvider(ChunkingSerializerProvider):
    """
    Custom serializer provider for chunker.
    """

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=TripletTableSerializer(),  # Better support for LLM and RAG
            params=MarkdownParams(image_placeholder="<!-- image -->"),
        )


class Embedder:
    """
    Embedder class that hold the workflow for embedding document
    """

    # Qdrant client and configurations
    client: QdrantClient
    collection_name: str

    # Docling components
    converter: DocumentConverter
    tokenizer: BaseTokenizer
    chunker: HybridChunker

    # Embedding model
    embedding_model: str

    def __init__(
        self,
        qdrant_conn: str,
        collection_name: str,
        embedding_model: str,
        tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the embedder. This will also create Qdrant collection if not exists

        Args:
            qdrant_conn: Qdrant connection string.
            collection_name: Qdrant collection name.
            embedding_model: Embedding model name.
            tokenizer_model: Tokenizer model name.

        """

        # Initialize Qdrant client
        logger.info("Initialize Qdrant client")
        self.client = QdrantClient(qdrant_conn)
        self.collection_name = collection_name
        if not self.client.collection_exists(collection_name):
            logger.info("Qdrant collection does not exist, creating...")
            success = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )

            if not success:
                msg = "Failed to create Qdrant collection"
                logger.error(msg)
                raise Exception(msg)

            logger.info("Qdrant collection created successfully")

        # Initialize Docling components
        self.converter = DocumentConverter()
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(tokenizer_model),
            max_tokens=512,
        )
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True,
            serializer_provider=SerializerProvider(),
        )

        self.embedding_model = embedding_model

        logger.info("Initialized embedder successfully")

    def chunk_document(
        self,
        filepath: str,
        filename: str | None = None,
        allow_log: bool = False,
    ) -> list[Chunk]:
        """
        Chunk document. Returns a list of serialized chunks.

        Args:
            converter: Docling document converter.
            chunker: Docling hybrid chunker
            tokenizer: Docling tokenizer
            filepath: Path to the document file.
            filename: Document file name
            allow_log: Whether to log the chunk text.

        Returns:
            List of serialized chunks.
        """

        # Validation: check if filepath exists
        if not Path(filepath).exists():
            msg = f"File {filepath} does not exist"
            logger.warning(msg)
            raise FileNotFoundError(msg)

        # Read document into Docling document
        doc = self.converter.convert(Path(filepath).absolute()).document

        # Chunk markdown
        base_chunks = self.chunker.chunk(doc)
        chunks: list[Chunk] = []

        for i, chunk in enumerate(base_chunks):
            # Serialize chunk
            serialized = self.chunker.contextualize(chunk=chunk)
            chunks.append(
                Chunk(
                    serialized=serialized,
                    metadata=Embedder.extract_chunk_metadata(
                        chunk.meta.export_json_dict(),
                        filename,
                    ),
                )
            )

            # Log (for development)
            if allow_log:
                # Print the chunk text
                serialized_token_count = self.tokenizer.count_tokens(serialized)
                print(f"=== {i} === ({serialized_token_count} tokens)")
                print(f"(text):\n{serialized}")
                print(f"(metadata):\n{chunk.meta.export_json_dict()}")
                print()

        return chunks

    @staticmethod
    def extract_chunk_metadata(
        metadata: dict[str, Any],
        filename: str | None = None,
    ) -> dict[str, Any]:
        """
        Extract useful metdata from chunk metadata extracted from Docling

        Args:
            metadata: chunk metadata after export to JSON
            filename: optional filename, if None then it will use the metadata filename

        Returns:
            dict[str, Any]: Useful metadata
        """

        pages: list[int] = []
        filename = filename if filename else metadata["origin"]["filename"]

        # Extract reference from metadata
        for item in metadata["doc_items"]:
            for prov in item["prov"]:
                pages.append(prov["page_no"])

        # Remove duplicates
        pages = list(set(pages))

        return {"pages": pages, "filename": filename}

    @staticmethod
    async def embed(content: str, model: str) -> list[float]:
        """
        Embed a string.

        Args:
            content: Content to embed.
            model: Embedding model name.

        Returns:
            Embedding vector.
        """

        content = content.strip()
        if content == "":
            raise ValueError("Content is empty")

        # Send request to embedding model
        resp = await litellm.aembedding(model, content)

        # Check if response data is empty
        if len(resp.data) == 0:
            msg = "Unexpected failure when embedding: response data is empty"
            logger.error(msg)
            raise Exception(msg)

        return list(resp.data[0].embedding)

    @staticmethod
    async def embed_batch(contents: list[str], model: str) -> list[list[float]]:
        """
        Embed a list of strings.

        Args:
            contents: List of strings to embed.
            model: Embedding model name.

        Returns:
            List of embedding vectors.
        """

        if len(contents) == 0:
            raise ValueError("Contents is empty")

        # Validation: since some embedding model would throw error if
        # any string in the batch is empty, we'll filter all of them
        # before passing to an LLM
        contents = [content for content in contents if content.strip() != ""]

        # Send request to embedding model
        resp = await litellm.aembedding(model, contents)

        # Return result
        embeddings = [list(data.embedding) for data in resp.data]
        return embeddings

    def add_vectors(
        self,
        payload: VectorPayload,
        embeddings: list[Chunk],
        vectors: list[list[float]],
    ) -> bool:
        """
        Add vectors to Qdrant. Internally, we use upsert, so chunk that has the same id
        will get overwriten (although id collision is unlikely).

        Args:
            payload: Vector payload.
            embeddings: Embeddings chunks.
            vectors: Vectors list corresponding to each embedding.

        Returns:
            True if vectors are added successfully, False otherwise.
        """

        # Construct the point struct list
        points = []
        for i, vector in enumerate(vectors):
            # Generate point ID
            point_id = Embedder.generate_id(payload.file_id, embeddings[i].serialized)

            # Set serialized chunk content and metdata to payload
            payload.chunk_content = embeddings[i].serialized
            payload.chunk_metadata = embeddings[i].metadata

            # Append point to list
            points.append(
                PointStruct(id=point_id, vector=vector, payload=asdict(payload))
            )

        # Upsert points to Qdrant
        result = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,  # Wait for all points to be upserted before return operation status
        )

        # Check operation status
        if result.status != UpdateStatus.COMPLETED:
            logger.error(f"Failed to upsert points: {result}")
            return False

        logger.info("Embed document success")
        return True

    @staticmethod
    def generate_id(document_id: str, content: str) -> str:
        """
        Generate UUIDv5 for Qdrant point struct.
        """

        return str(uuid.uuid5(uuid.NAMESPACE_DNS, document_id + content))

    async def embed_document(
        self,
        disease: str,
        filepath: str,
        filename: str | None = None,
    ) -> bool:
        """
        Embed a document and add it to Qdrant.

        Args:
            disease: Disease name.

            filepath: Path to the PDF file.

            fileame: Document filename, it can be optional. If not provided, it will automatically
            figure out filename
        """

        try:
            start_time = time.perf_counter()

            # Chunk document
            logger.info("Start chunking document")
            embeddings = self.chunk_document(filepath=filepath, filename=filename)
            logger.info(f"Finished chunking document: {len(embeddings)} chunks")

            # Embed chunks
            logger.info("Start embedding chunks")
            vectors = await Embedder.embed_batch(
                contents=[embedding.serialized for embedding in embeddings],
                model=self.embedding_model,
            )
            logger.info(f"Finished embedding chunks: {len(vectors)} vectors")

            # Add to vector store
            logger.info("Start adding vectors to vector store")
            payload = VectorPayload(filepath, disease)
            success = self.add_vectors(
                payload=payload,
                embeddings=embeddings,
                vectors=vectors,
            )

            if success:
                logger.info(f"Add {len(vectors)} vectors to vector store")
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                logger.info(f"Embedding document in {execution_time:.4f} seconds")
                return True

            logger.error("Failed to add vectors to vector store")
            return False
        except FileNotFoundError as e:
            logger.error(f"File {filepath} does not exist: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
