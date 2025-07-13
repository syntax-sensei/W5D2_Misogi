import config
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from loader import all_splits

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=config.OPENAI_API_KEY,
    model=config.EMBEDDING_MODEL
)

print(f"Processing {len(all_splits)} chunks...")

# Create and populate Chroma vector store
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory=config.CHROMA_PERSIST_DIRECTORY
)

# Persist the vector store to disk
vectorstore.persist()

print(f"✅ Successfully stored {len(all_splits)} chunks in ChromaDB")
print(f"Location: {config.CHROMA_PERSIST_DIRECTORY}")

# Test retrieval
print("\n🧪 Testing retrieval...")
try:
    results = vectorstore.similarity_search("customer health scoring", k=2)
    print("Sample search results:")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc.page_content[:100]}...")
except Exception as e:
    print(f"Search test failed: {e}")