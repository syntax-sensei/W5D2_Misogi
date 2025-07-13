import os

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY = "data/embeddings"
COLLECTION_NAME = "gitlab_knowledge_base"

# Company Context
COMPANY_NAME = "GitLab"
with open(os.path.join(os.path.dirname(__file__), "company_context.txt"), "r", encoding="utf-8") as f:
    COMPANY_CONTEXT = f.read().strip()

