from pathlib import Path
import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from backend.config import config
import pandas as pd
from typing import List, Dict, Any
from backend.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_data"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

class VectorStoreManager:
    def __init__(self):
        # Lazy init: avoid downloading embedding models on import/startup.
        # Embeddings and vectorstore will be created only when needed (ingest=True).
        self.embeddings = None
        self.vectorstore = None

    def _init_embeddings_and_vectorstore(self):
        if self.embeddings is None:
            logger.info("Initializing HuggingFaceEmbeddings...")
            self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        if self.vectorstore is None:
            logger.info("Initializing Chroma vectorstore...")
            self.vectorstore = Chroma(
                collection_name=config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=config.CHROMA_PERSIST_DIR
            )
    
    def fetch_huggingface_data(self, offset: int = 0, length: int = 100) -> List[Dict]:
        """Fetch data from Hugging Face datasets API"""
        url = f"https://datasets-server.huggingface.co/rows?dataset=bitext%2FBitext-customer-support-llm-chatbot-training-dataset&config=default&split=train&offset={offset}&length={length}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            rows = []
            for item in data.get('rows', []):
                row_data = item.get('row', {})
                rows.append({
                    'instruction': row_data.get('instruction', ''),
                    'category': row_data.get('category', ''),
                    'intent': row_data.get('intent', ''),
                    'response': row_data.get('response', ''),
                    'flags': row_data.get('flags', '')
                })
            
            logger.info(f"Fetched {len(rows)} records from Hugging Face (offset={offset})")
            return rows
            
        except Exception as e:
            logger.info(f"❌ Error fetching from Hugging Face: {e}")
            return []

    def save_raw_data(self, rows: List[Dict[str, Any]], filename: str = "huggingface_raw_data.csv"):
        """Save raw Hugging Face data to CSV in data/raw_data."""
        if not rows:
            logger.warning("No raw rows to save to CSV")
            return

        csv_path = RAW_DATA_DIR / filename
        try:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"Saved raw data to {csv_path}")
        except Exception as e:
            logger.exception(f"Failed to save raw data to CSV: {e}")

    def has_vectorstore_data(self) -> bool:
        # Avoid initializing embeddings/vectorstore just to check for presence.
        persist_dir = Path(config.CHROMA_PERSIST_DIR)
        if not persist_dir.exists():
            return False
        # Heuristic: if the persist directory contains files, assume data exists.
        try:
            files = list(persist_dir.rglob("*"))
            return any(f.is_file() for f in files)
        except Exception:
            return False

    def load_raw_data_from_csv(self, max_records: int = None, filename: str = "huggingface_raw_data.csv") -> int:
        csv_path = RAW_DATA_DIR / filename
        if not csv_path.exists():
            logger.warning("No raw Hugging Face CSV exists to load.")
            return 0

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            logger.exception(f"Failed to read raw CSV from {csv_path}: {e}")
            return 0

        rows = df.to_dict(orient="records")
        if max_records is not None:
            rows = rows[:max_records]

        documents = []
        for row in rows:
            content = f"""
            Customer Question: {row.get('instruction', '')}
            Category: {row.get('category', '')}
            Intent: {row.get('intent', '')}
            Support Response: {row.get('response', '')}
            """
            documents.append(Document(
                page_content=content,
                metadata={
                    "category": row.get('category', ''),
                    "intent": row.get('intent', ''),
                    "source": "huggingface_csv",
                    "flags": row.get('flags', '')
                }
            ))

        if documents:
            # Ensure embeddings/vectorstore exist before ingest
            self._init_embeddings_and_vectorstore()
            self.vectorstore.add_documents(documents)
            try:
                self.vectorstore.persist()
            except Exception:
                logger.warning("Chroma persistence is deprecated or unsupported by this version.")
            logger.info(f"Loaded {len(documents)} documents from raw CSV into ChromaDB")
            return len(documents)

        logger.info("No documents loaded from CSV")
        return 0

    def add_huggingface_to_vectorstore(self, max_records: int = 30000, force: bool = True, ingest: bool = True) -> int:
        """Fetch data from Hugging Face and add to vector store.

        If `ingest` is False, the method will only fetch the dataset rows and
        save them as CSV in `data/raw_data` without computing embeddings or
        adding documents to the vector store. This is useful when embedding
        model downloads are failing or should be deferred.
        """
        raw_csv = RAW_DATA_DIR / "huggingface_raw_data.csv"
        if raw_csv.exists() and not force:
            if self.has_vectorstore_data():
                logger.info(
                    "📌 Existing raw Hugging Face CSV found and vector store already has data. Skipping fetch. "
                    "Use force=True to refresh data."
                )
                return 0

            logger.info(
                "📌 Existing raw Hugging Face CSV found but vector store appears empty. "
                "Loading data from CSV into vector store."
            )
            if ingest:
                return self.load_raw_data_from_csv(max_records=max_records)
            logger.info("Ingest disabled; skipping load from CSV into vector store.")
            return 0

        all_rows = []
        offset = 0
        batch_size = 100
        
        # Fetch in batches until we reach max_records
        while len(all_rows) < max_records:
            rows = self.fetch_huggingface_data(offset=offset, length=batch_size)
            if not rows:
                break
            all_rows.extend(rows)
            offset += batch_size
            logger.info(f"Fetched {len(all_rows)} records so far...")
        
        # Convert to LangChain Documents
        documents = []
        for row in all_rows[:max_records]:
            # Create rich content combining instruction and response
            content = f"""
            Customer Question: {row['instruction']}
            Category: {row['category']}
            Intent: {row['intent']}
            Support Response: {row['response']}
            """
            
            doc = Document(
                page_content=content,
                metadata={
                    "category": row['category'],
                    "intent": row['intent'],
                    "source": "huggingface",
                    "flags": row.get('flags', '')
                }
            )
            documents.append(doc)
        
        # Save raw data to CSV before optionally adding to the vector store
        self.save_raw_data(all_rows)

        if not ingest:
            logger.info("Ingest disabled: raw CSV saved but embeddings not created.")
            return len(all_rows[:max_records])

        # Add to vector store
        if documents:
            # Ensure embeddings/vectorstore initialized
            self._init_embeddings_and_vectorstore()
            self.vectorstore.add_documents(documents)
            try:
                self.vectorstore.persist()
            except Exception:
                logger.warning("Chroma persistence is deprecated or unsupported by this version.")
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return len(documents)

        logger.info("No documents to add")
        return 0
    
    def fetch_and_save_csv_only(self, max_records: int = 30000) -> int:
        """Phase 1: Only fetch raw data and save to CSV. No embeddings."""
        raw_csv = RAW_DATA_DIR / "huggingface_raw_data.csv"
        if raw_csv.exists():
            logger.info("Raw CSV already exists. Skipping fetch.")
            return 0

        all_rows = []
        offset = 0
        batch_size = 100

        while len(all_rows) < max_records:
            rows = self.fetch_huggingface_data(offset=offset, length=batch_size)
            if not rows:
                break
            all_rows.extend(rows)
            offset += batch_size
            logger.info(f"Fetched {len(all_rows)} records so far...")

        if all_rows:
            self.save_raw_data(all_rows[:max_records])
            logger.info(f"Phase 1 complete: {len(all_rows[:max_records])} rows saved to CSV.")
            return len(all_rows[:max_records])

        logger.warning("No rows fetched from Hugging Face.")
        return 0

    def build_embeddings_from_csv(self, max_records: int = 30000) -> int:
        """Phase 2: Build embeddings from existing CSV. No fetching."""
        raw_csv = RAW_DATA_DIR / "huggingface_raw_data.csv"
        if not raw_csv.exists():
            logger.error("Cannot build embeddings: raw CSV not found. Run Phase 1 first.")
            return 0

        if self.has_vectorstore_data():
            logger.info("Vector store already has data. Skipping embedding creation.")
            return 0

        return self.load_raw_data_from_csv(max_records=max_records)