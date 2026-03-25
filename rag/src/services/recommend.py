# Standard lib
from typing import List
from collections import defaultdict
import math

# 3rd: Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint, FieldCondition, Filter, MatchValue

# 3rd: Instructor
from instructor import Instructor, Mode, from_litellm

# 3rd: LiteLLM
from litellm import completion

# 3rd: Pydantic
from pydantic import BaseModel

# Custom lib
from src.services.embed import Embedder, VectorPayload
from src.utils import logger, inject_llm_env


class MedicineResult(BaseModel):
    """
    Medicine result class.

    Attributes:
        dosage: Dosage.
        unit: Unit.
        route_of_administration: Route of administration.
        dosing_interval: Dosing interval.
    """

    dosage: float
    unit: str
    route_of_administration: str
    dosing_interval: float


class RecommendResult(BaseModel):
    """
    Recommend result class.

    Attributes:
        recommended_documents: List of recommended documents, which is sorted from best to worst
        medicine_result: Medicine result.
    """

    recommended_documents: list[str]
    treatment_site: str
    empiric_antibiotic: str
    medicine_result: MedicineResult


class Recommender:
    """
    Recommender class that hold the workflow for recommending medicine.
    """

    # Qdrant client and configurations
    qdrant_client: QdrantClient
    collection_name: str

    # LLM client and models
    llm_client: Instructor
    embedding_model: str

    # Queries and its vector embedding
    relevant_info_vector: list[float] = []

    def __init__(
        self,
        qdrant_conn: str,
        collection_name: str,
        embedding_model: str,
    ):
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(qdrant_conn)
        self.collection_name = collection_name
        if not self.qdrant_client.collection_exists(collection_name):
            # It shouldn't be the Recommender responsibility to handle Qdrant
            # collection initialization
            logger.info("Qdrant collection does not exist")
            raise Exception("Qdrant collection does not exist")

        # Connect to generative LLM
        self.llm_client = from_litellm(
            completion=completion,
            mode=Mode.JSON,
        )

        self.embedding_model = embedding_model

    def search_top_k(
        self,
        vector: list[float],
        filter: Filter,
        payload_includes: list[str] | bool,
        top_k: int,
    ) -> List[ScoredPoint]:
        """
        Search for top_k points with the given query vector and filters.

        Args:
            vector: Query vector.
            filter: Filter to apply.
            payload_includes: Payload fields to include in the response.
            top_k: Number of top results to return.

        Returns:
            List of ScoredPoint objects.
        """

        if top_k < 0:
            raise ValueError("top_k must be greater than or equal to 0")
        if top_k > 100:
            logger.warning(f"top_k too large ({top_k}) may affect performance")

        resp = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=filter,
            with_vectors=False,
            with_payload=payload_includes,
        )

        return resp.points

    @staticmethod
    def rerank(points: List[ScoredPoint]) -> List[str]:
        """
        Rerank the points to get the most suitable documents list.
        We'll use the idea of TF-IDF for this rerank algorithm, where:
        TF: the frequency in which a document contains the chunk
        IDF: the cosine similarity score, which show how closely
        the query vector to the stored chunk -> can be considered
        "important" factor in TF-IDF. The score will be accumulated (add up)

        Args:
            points: List of ScoredPoint objects.

        Returns:
            List of file IDs.
        """

        doc_scores: dict[str, float] = defaultdict(float)
        doc_chunk_count: dict[str, int] = defaultdict(int)

        # Calculate score and frequency for each chunk/point
        for point in points:
            if point.payload is None:
                message = f"point payload unexpectedly None: id = {point.id}"
                logger.error(message)
                raise Exception(message)

            # Get file_id from point payload
            file_id = str(point.payload.get("file_id"))
            doc_scores[file_id] += point.score
            doc_chunk_count[file_id] += 1

        total_chunks = len(points)

        return sorted(
            doc_scores.keys(),
            key=lambda file_id: (
                Recommender.tfidf(
                    doc_scores[file_id],
                    doc_chunk_count[file_id],
                    total_chunks,
                ),
            ),
            reverse=True,
        )

    @staticmethod
    def tfidf(score: float, count: int, total: int) -> float:
        """
        Calculate the TF-IDF score.
        TF = count / total
        IDF = score

        Args:
            score: Score
            count: Count.
            total: Total.
        """

        return math.log(1 + score) * (count / total)

    @staticmethod
    def generate_document_info_query() -> str:
        """
        Helper method: return the second query in the recommendation flow:
        get relevant information like medicine usage from the most relevant document
        """

        return """
        Medicine information about disease: dosage, route of administration, dosing interval,
        treatment site (inpatient or outpatience), information about empiric antibiotic
        """

    @staticmethod
    def prepare_prompt(
        clinical_picture: str,
        payloads: list[VectorPayload],
    ) -> str:
        """
        Prepare the prompt for the LLM.

        Args:
            clinical_picture: Clinical picture.
            relevant_info: Relevant information.

        Returns:
            Prompt string.
        """
        # Flatten the payloads into a string
        relevant_info = "\n".join([payload.__str__() for payload in payloads])

        return f"""
        You are a doctor, and you are helping patients diagnose their diseases based 
        on the knowledge from the hospital internal treatment protocols.

        Here is the patient's clinical picture:
        {clinical_picture}

        This is the relevant information about medicine:
        {relevant_info}

        Now, from these information, answer the question:
        1. What is the empiric antibiotic of this disease?
        2. What is the client treatment site (inpatient or outpatient)?
        3. What is the medicine informatio (dosage, route of administration, dosing interval)?

        Return the result as a JSON with this format:
        {{
            "treatment_site": "string",
            "empiric_antibiotic": "string",
            "medicine_result": {{
                "dosage": float,
                "unit": "string",
                "route_of_administration": "string",
                "dosing_interval": float
            }}
        }}

        """

    async def recommend_medicine(
        self,
        disease: str,
        clinical_picture: str,
        model: str,
        provider: str | None = None,
        api_key: str | None = None,
    ) -> RecommendResult | None:
        """
        Recommend medicine for a given disease and clinical picture.
        This workflow consist of 3 main steps:
        1. From the clinical picture -> get the list of most relevent document (treatment regimen)
        2. From the recommended list, use the most relevant document as filter, query chunks
        that contains medicine dosage (or whatever we may need in the future)
        3. With chunks fetched from the second step, make a call to LLM to produce our result
        based on our structured (JSON response), since chunk store in vector store would be fragmented
        and not structured as we like (they will be used as document reference when display to user)

        Args:
            disease: Disease name.
            clinical_picture: Clinical picture.

        Returns:
            RecommendResult object.

        """

        try:
            # Embed the clinical picture
            embedded_clinical_picture = await Embedder.embed(
                content=clinical_picture,
                model=self.embedding_model,
            )
            logger.info("Embedded clinical picture successfully")

            # Query to find the most relevant documents. Since we only fetch only the file id,
            # we can set a somewhat high top_k for more accurate result
            recommended_docs = self.search_top_k(
                vector=embedded_clinical_picture,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="disease",
                            match=MatchValue(value=disease),
                        )
                    ]
                ),
                payload_includes=["file_id"],
                top_k=200,
            )

            # Rerank the documents
            recommended_docs = Recommender.rerank(recommended_docs)
            if len(recommended_docs) == 0:
                # This could happen if we set the minimal scoring metric to avoid
                # giving unrelevant document when the query didn't semantically match
                # any documents.
                logger.warning("No recommended documents found")
                return None

            # Second route: get relevant information from the most relevant document
            if self.relevant_info_vector is None or len(self.relevant_info_vector) == 0:
                # If the second query is not embedded, we embed it and store them
                # for future use
                logger.info("Embedding second query")
                self.relevant_info_vector = await Embedder.embed(
                    content=Recommender.generate_document_info_query(),
                    model=self.embedding_model,
                )
            relevant_info_resp = self.search_top_k(
                vector=self.relevant_info_vector,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_id",
                            match=MatchValue(value=recommended_docs[0]),
                        )
                    ]
                ),
                payload_includes=True,
                top_k=10,
            )

            # Prompt injection with relevant_info found
            relevant_info: list[VectorPayload] = []
            for point in relevant_info_resp:
                if point.payload is None:
                    message = f"point payload unexpectedly None: id = {point.id}"
                    logger.error(message)
                    continue
                payload = VectorPayload(**point.payload)
                relevant_info.append(payload)
            prompt = Recommender.prepare_prompt(clinical_picture, relevant_info)

            # Called to generative LLM for answer
            if provider is not None and api_key is not None:
                inject_llm_env(provider=provider, api_key=api_key)
            result = self.llm_client.chat.completions.create(
                model=model,
                response_model=RecommendResult,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            # Pass the recommended_docs to result
            result.recommended_documents = recommended_docs

            return result
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
