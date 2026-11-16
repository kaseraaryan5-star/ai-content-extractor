import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self, collection_name="tech_content"):
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        print("Loading Embedding model (local)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self._setup_collection()

    def _setup_collection(self):
        """Collection create karta hai agar nahi exist karti"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )

    def upsert_transcript(self, transcript, metadata):
        """Text ko chunks mein tod kar Qdrant mein save karta hai"""
        print("Chunking transcript...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(transcript)
        
        print(f"Upserting {len(chunks)} chunks to Qdrant...")
        points = []
        for i, chunk in enumerate(chunks):
            vector = self.embeddings.embed_query(chunk)
            point_id = abs(hash(f"{metadata['url']}_{i}")) % (10**12)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "url": metadata['url'],
                        "title": metadata.get('title', 'Unknown')
                    }
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("Successfully stored in Qdrant!")

    def search(self, query, limit=3):
        """Semantic search query with compatibility fix for qdrant-client"""
        query_vector = self.embeddings.embed_query(query)
        
        if hasattr(self.client, "query_points"):
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            ).points
        else:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
        return [hit.payload for hit in search_result]

if __name__ == "__main__":
    db = VectorStore()
    test_text = "Python is a great language for AI. Qdrant is a powerful vector database for fast similarity search."
    db.upsert_transcript(test_text, {"url": "test_url", "title": "Test Title"})
    print("\n--- SEARCH TEST RESULT ---")
    results = db.search("Tell me about vector database")
    for r in results:
        print("-", r['text'])
