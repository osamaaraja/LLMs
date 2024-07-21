'''
Exploring vairous types of embeddings in RAG
'''

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # loading the environment variables
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "JungleBook.txt")

db_dir = os.path.join(current_dir, 'db')

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)

    if not os.path.exists(persistent_directory):
        os.makedirs(persistent_directory)

        docs = docs[:80] # taking subsample of the chunks to avoid memory constraints
        print(f"\n--- Creating vector store {store_name} ---\n")
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"Vector store {store_name} created successfully.")

    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")


loader = TextLoader(file_path, encoding="utf-8") # Loading the document
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0) # Using character text splitter to split the doc into chunks
docs = text_splitter.split_documents(documents) # Chunked Document

print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}")



# Define Embeddings from OpenAI

embeddings_ada = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
create_vector_store(docs, embeddings_ada, "chroma_db_ada")

embeddings_small = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
create_vector_store(docs, embeddings_small , "chroma_db_small")

# Define Embedding from HuggingFace
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")

def query_vector_store(store_name, query, embeddings, db_dir):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Query vector store {store_name} ---")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2, "score_threshold": 0.2})

        try:
            relevant_docs = retriever.invoke(query)

            # Display the relevant results with metadata
            print(f"\n--- Relevant Documents for {store_name} ---")
            for i, doc in enumerate(relevant_docs, 1):
                print(f"Document {i}:\n{doc.page_content}\n")
                if doc.metadata:
                    print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        except Exception as e:
            print(f"An error occurred while querying the vector store: {e}")

    else:
        print(f"Vector store {store_name} does not exist.")

query = "who is Shere Khan?"

# Query the vector store
query_vector_store("chroma_db_ada", query, embeddings_ada, db_dir)
query_vector_store("chroma_db_small", query, embeddings_small, db_dir)
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings, db_dir)

