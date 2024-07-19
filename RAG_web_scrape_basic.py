import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()  # loading the environment variables
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, 'db')
persistent_directory = os.path.join(db_dir, "chroma_db_apple") # Already existing

# Step-1: Scrape the content from apple.com using WebBasedLoader
# WebBasedLoader loads web pages and extracts their content
urls = ["https://www.apple.com/"]

# Create a loader for web content
loader = WebBaseLoader(urls) # instead of text loader in the previous scripts, here web based loader is used for url
documents = loader.load()

# Step-2: Split the scraped content into chunks
# CharacterTextSplitter splits the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print("\n---Document Chunk Information---\n")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk: \n{docs[0].page_content}")

# Step-3: Create Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

# Step-4: Create the vector store
if not os.path.exists(persistent_directory):
    print(f"\n---Creating vector store in {persistent_directory}---\n")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print(f"\n---Finished creating vector store in  {persistent_directory}\n")

else:
    print(f"Vector store in {persistent_directory} already exists!\nNo need to initialize.")
    db =Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Step-5: Query the vector store
# Create a retriever for querying the vector store
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

# define the query
query = "What are the new products announced on Apple.com?"

# Retrieve relevant documents based on query
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n---Relevant Documents---\n")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")







