import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()  # loading the environment variables
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'books', 'JungleBook.txt')
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')
# Read the text content from the file
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")

# Add more print statements to inspect the chunks if needed
for i, chunk in enumerate(docs):
    print(f"Chunk {i + 1}:\n{chunk.page_content[:500]}\n")  # Print the first 500 characters of each chunk to avoid excessive output

# Use only the first few chunks for testing
docs = docs[:50]

# Create embeddings for the document chunks
print("\n--- Creating Embeddings ---")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

# Create the vector store and persist it automatically
try:
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")
except Exception as e:
    print(f"Error during vector store creation: {e}")

print("Script completed successfully.")
