import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "db", "chroma_db")

# Define the Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

# Load the existing vector store with the embeddings
db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

# Define the user's question
query = "Who is Raksha?"

# Retrieve relevant documents based on the query

retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold":0.4})
relevant_docs = retriever.invoke(query)

# Check if relevant documents were retrieved
if not relevant_docs:
    print("No relevant documents were found.")
else:
    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
