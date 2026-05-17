from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from backend.config import config
import json
from backend.utils.logger import get_logger
from backend.utils.retry import async_retry

logger = get_logger(__name__)

class KnowledgeBaseRetrievalAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )
        
        self.vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR
        )
        
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.MODEL_NAME,
            temperature=0.2
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.MAX_RETRIEVAL_DOCS}
        )
    
    async def retrieve_knowledge(self, query: str, intent: str) -> List[Dict[str, Any]]:
        # Enhance query with intent context
        enhanced_query = f"Intent: {intent}. Customer question: {query}"
        # Attempt to use an async retriever if available, otherwise fall back to
        # sync retrieval methods or a direct vectorstore search.
        docs = []
        if hasattr(self.retriever, "aget_relevant_documents"):
            logger.debug("Using async retriever: aget_relevant_documents")
            docs = await self.retriever.aget_relevant_documents(enhanced_query)
        elif hasattr(self.retriever, "get_relevant_documents"):
            logger.debug("Using sync retriever: get_relevant_documents")
            docs = self.retriever.get_relevant_documents(enhanced_query)
        else:
            # Fallback to vectorstore similarity search
            logger.debug("Falling back to vectorstore.similarity_search_with_relevance_scores")
            docs = self.vectorstore.similarity_search_with_relevance_scores(
                enhanced_query, k=config.MAX_RETRIEVAL_DOCS
            )

        retrieved_docs = []
        for doc in docs:
            # Some vectorstore APIs return tuples (Document, score)
            if isinstance(doc, tuple) and len(doc) == 2:
                document, score = doc
                content = getattr(document, "page_content", str(document))
                metadata = getattr(document, "metadata", {})
                relevance = score
            else:
                content = getattr(doc, "page_content", str(doc))
                metadata = getattr(doc, "metadata", {})
                relevance = metadata.get("score", 1.0)

            retrieved_docs.append({
                "content": content,
                "metadata": metadata,
                "relevance_score": relevance
            })

        logger.info(f"retrieve_knowledge returned {len(retrieved_docs)} docs for query")

        return retrieved_docs
    
    async def answer_with_context(self, query: str, intent: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        context = "\n\n".join([doc["content"] for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable customer support agent.
            Use the following context from the knowledge base to answer the customer's question.
            If the answer is not in the context, say so clearly.
            
            Context:
            {context}
            
            Provide:
            1. Main answer
            2. Confidence level (0-1)
            3. Sources used
            4. Missing information (if any)"""),
            ("human", "Intent: {intent}\nQuestion: {query}")
        ])
        
        chain = prompt | self.llm
        response = await async_retry(
            lambda: chain.ainvoke({
                "context": context,
                "intent": intent,
                "query": query
            }),
            retries=3,
            initial_delay=0.5,
            backoff_factor=2.0,
        )
        
        return {
            "answer": response.content,
            "retrieved_docs": docs,
            "num_docs": len(docs)
        }