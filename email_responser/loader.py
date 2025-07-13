from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Load Markdown files
loader = DirectoryLoader("C:/Users/Aditya Sebastian/Desktop/MISOGI/W5D2/email_responser/data", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
docs = loader.load()

# print(f"Loaded {len(docs)} documents")
# for doc in docs:
#     print(f"Document: {doc.metadata.get('source', 'Unknown')}")

# Initialize splitter
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

# Split documents
all_splits = []

for doc in docs:
    print(f"\nProcessing: {doc.metadata.get('source', 'Unknown')}")
    splits = markdown_splitter.split_text(doc.page_content)
    print(f"Created {len(splits)} chunks")
    
    # # Show first few chunks from each document
    # for i, split in enumerate(splits[:3]):  # Show first 3 chunks
    #     print(f"  Chunk {i+1}: {split.page_content[:]}...")
    
    # Add enhanced metadata to each chunk
    for split in splits:
        split.metadata.update({
            'source_file': doc.metadata.get('source', 'Unknown'),
            'chunk_type': 'markdown_section',
            'total_chunks': len(splits)
        })
    
    all_splits.extend(splits)

# all_splits now contains Document chunks from all files

print(f"\nTotal chunks created: {len(all_splits)}")

