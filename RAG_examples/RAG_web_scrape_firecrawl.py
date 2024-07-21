import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import FireCrawlLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()  # loading the environment variables
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, 'db')
persistent_directory = os.path.join(db_dir, "chroma_db_firecrawl")


def create_vector_store():
    """
    crawl the website, split the content, create embeddings, and persist the vector store.
    """
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL API key not found.")

    # Step-1: Crawl the website using FireCrawlLoader
    print("Begin Crawling the website...")
    loader = FireCrawlLoader(api_key=api_key, url="https://apple.com", mode="scrape")
    docs = loader.load()
    print("Finished Crawling the website.")

    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # Step-2: Split the crawled content into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    print("\n---Document Chunk Information---\n")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk: \n{docs[0].page_content}")

    # Step-3: Create Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

    # Step-4: Create the vector store

    print(f"\n---Creating vector store in {persistent_directory}---\n")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print(f"\n---Finished creating vector store in  {persistent_directory}\n")


if not os.path.exists(persistent_directory):
    create_vector_store()

else:
    print(f"Vector store in {persistent_directory} already exists!\nNo need to initialize.")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
db =Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Step-5: Query the vector store
def query_vector_store(query):
    """
    Query the vector store with the specified question.
    """
    # Create a retriever for querying the vector store
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # Retrieve relevant documents based on query
    relevant_docs = retriever.invoke(query)
    # Display the relevant results with metadata
    print("\n---Relevant Documents---\n")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")



query = "What are the new products announced on Apple.com?"
query_vector_store(query)