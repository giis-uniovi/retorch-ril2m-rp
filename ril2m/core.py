# -*- coding: utf-8 -*-
import json
from pathlib import Path
import chromadb
import logging

import os

from jinja2 import Template

from ril2m.helpers import setup_logging, loadfile, save_output_to_file
from ril2m.helpers import OllamaClient

TEMPERATURE = 0.5
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_MODEL_NAME = DEFAULT_MODEL.replace(":", "-")
EMBED_MODEL = "nomic-embed-text:v1.5"  # ollama pull nomic-embed-text
COLLECTION_NAME = "example_embeedings2"
TOP_K = 5  # default value for the number of test cases

URI = "http://" + ("ollama-gpu:11434" if os.getenv("CI_ENV") else os.getenv("OLLAMAIP"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(BASE_DIR, 'input', 'prompts')
CONTEXTS = os.path.join(BASE_DIR, 'input', 'context')

setup_logging()
logger = logging.getLogger(__name__)


class TestCaseVectorStore:
    def __init__(self, persist_dir: str = "./knowledge_base"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.ollama = OllamaClient(base_url=URI, model=DEFAULT_MODEL, embed_model=EMBED_MODEL,
                                   temperature=TEMPERATURE)

    def add_test_cases(self, test_cases: list[dict]):
        """
        Index test cases into ChromaDB.

        Each dict must have: id, testname, annotations, code.
        Embeddings are computed exclusively from the ``code`` field so that
        similarity search is based on test logic, not on existing annotations.
        Annotations are stored only in the metadata for later retrieval.
        """
        if not test_cases:
            logger.warning("add_test_cases called with an empty list — nothing indexed.")
            return

        ids, embeddings, documents, metadatas = [], [], [], []

        for tc in test_cases:
            tc_id = tc["id"]
            code = tc["code"]
            annotations = tc.get("annotations", "")
            test_name = tc.get("testname", "")

            text_to_embed = code.strip()  # annotations intentionally excluded

            logger.debug("  Embedding: %s", tc_id)
            embedding = self.ollama.embed(text_to_embed)

            ids.append(tc_id)
            embeddings.append(embedding)
            documents.append(text_to_embed)
            metadatas.append({
                "id": tc_id,
                "testname": test_name,
                "annotations": annotations,
                "code": code,
                **(tc.get("metadata") or {}),
            })

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("  Indexed %d test cases into ChromaDB.", len(ids))

    def search(self, test_provided: str, top_k: int = TOP_K) -> list[dict]:
        """Search for the top_k test cases more similar to the provided one (test_provided).

        If the collection has fewer documents than *top_k*, n_results is clamped
        to the collection size.  An empty collection returns an empty list (zero-shot).
        """
        available = self.count()
        if available == 0:
            logger.debug("Collection is empty — returning no similar cases (zero-shot).")
            return []
        effective_top_k = min(top_k, available)
        query_embedding = self.ollama.embed(test_provided)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_top_k,
            include=["metadatas", "distances", "documents"],
        )
        similar = []
        for i in range(len(results["ids"][0])):
            similar.append({
                "id": results["ids"][0][i],
                "testname": results["metadatas"][0][i]["testname"],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "code": results["metadatas"][0][i]["code"],
                "annotations": results["metadatas"][0][i]["annotations"],
            })
        logger.debug("Retrieved %d similar test cases: %s", len(similar), [s["id"] for s in similar])
        return similar

    def count(self) -> int:
        return self.collection.count()


class JavaTestRAG:
    """
    JavaTestRAG is responsible for handling Retrieval-Augmented Generation (RAG)
    operations for Java test-related workflows.

    The vector store (ChromaDB + embeddings) is model-agnostic — embeddings are
    always produced by ``EMBED_MODEL`` and can be shared across LLM experiments.
    Only the chat LLM changes when *model* / *temperature* differ.

    Attributes:
       persist_dir (str): directory where the ChromaDB embeddings are stored.
       model (str): Ollama model used for annotation generation.
       temperature (float): sampling temperature for the LLM.
    """

    def __init__(
        self,
        persist_dir: str = "./knowledge_base",
        model: str = DEFAULT_MODEL,
        temperature: float = TEMPERATURE,
    ):
        self.store = TestCaseVectorStore(persist_dir)
        self.ollama = OllamaClient(
            temperature=temperature,
            base_url=URI,
            model=model,
            embed_model=EMBED_MODEL,
        )

    def index_from_file(self, path: str):
        """Load and indexes the embeedings from a file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.store.add_test_cases(data)

    def index_test_cases(self, test_cases: list[dict]):
        """Index directly from a list of test cases."""
        self.store.add_test_cases(test_cases)

    def retrieve(self, test_provided: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieves the top_k test cases more similar to the provided one (test_provided)."""
        return self.store.search(test_provided, top_k)

    def build_prompt(
        self,
        test_provided: str,
        top_k: int = TOP_K,
        resource_file: str | None = None,
    ) -> str:
        """Retrieve similar cases and render the Jinja2 prompt without querying the LLM."""
        similar = self.retrieve(test_provided, top_k)

        context_parts = []
        for i, tc in enumerate(similar, 1):
            sim_score = 1 - tc["distance"]  # cosine distance → similarity
            context_parts.append(
                f"--- Example {i} (similarity: {sim_score:.2%}) ---\n"
                f"ID: {tc['id']}\n"
                f"TestName: {tc['testname']}\n"
                f"Annotations: {tc['annotations']}\n"
                f"Code:\n```java\n{tc['code']}\n```"
            )
        context = "\n\n".join(context_parts)

        with open(os.path.join(PROMPTS, 'generate_annotations.j2')) as f:
            template = Template(f.read())

        if resource_file is None:
            resource_file = os.path.join(CONTEXTS, 'resourcefile.json')

        return template.render(
            resourcefile=loadfile(resource_file),
            testcase=test_provided,
            examples=context,
        )

    def query(
        self,
        test_provided: str,
        top_k: int = TOP_K,
        resource_file: str | None = None,
        seed: int | None = None,
    ) -> str:
        """
        RAG Pipeline
          1. Retrieve the top_k most similar training test cases
          2. Build a context-enriched prompt via the Jinja2 template
          3. Query the LLM and return the response

        Parameters
        ----------
        test_provided:
            Source code of the test case to annotate.
        top_k:
            Number of similar examples to include in the prompt context.
        resource_file:
            Path to the resource definitions JSON injected into the prompt.
            Defaults to ``ril2m/input/context/resourcefile.json``.
        seed:
            Ollama RNG seed for reproducibility.  When ``None`` the OllamaClient
            picks a random seed.
        """
        prompt = self.build_prompt(test_provided, top_k, resource_file)
        logger.debug("Prompt sent to LLM:\n%s", prompt)
        return self.ollama.chat(prompt, seed=seed)


if __name__ == "__main__":
    from ril2m.input.fetch_testcases import generate_ragtestcases, load_all_ragtestcases
    logger.info("Fetching test cases from repositories...")
    generate_ragtestcases()

    id_a_excluir = "TC-016"

    # 1. Load and merge all per-repo test case JSONs
    full_data = load_all_ragtestcases(CONTEXTS)

    # 2. Filter and remove the test case that we are going to use to query
    sample_test_cases = [tc for tc in full_data if tc["id"] != id_a_excluir]

    logger.info("Starting RAG...")
    rag = JavaTestRAG(persist_dir=os.path.join(BASE_DIR, 'chroma_db'))
    # 3. If there's no previous embeedings created (stored in our database) query the model to create it.
    if rag.store.count() == 0:
        logger.info("Indexing test cases...")
        rag.index_test_cases(sample_test_cases)
    else:
        logger.info(f"Using a existant database, the number of indexed test cases is: {rag.store.count()}.")

    # 4. Retrieve the excluded test case
    nuevo_test = next((item for item in full_data if item["id"] == id_a_excluir), None).get("code")

    # 5. Query the model with the test case to annotate or perform any action
    logger.info("\nQuerying the model with the test case\n")
    result = rag.query(
        test_provided=nuevo_test,
        top_k=4,
    )

    logger.info(result)
    save_output_to_file(f"rag-result-{DEFAULT_MODEL_NAME}.txt", result)
    logger.info(f"Metrics results saved to metrics_results-{DEFAULT_MODEL_NAME}.csv")

