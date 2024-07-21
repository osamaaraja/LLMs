'''
different ways of searching inside a vector store
'''

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()  # loading the environment variables
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, 'db')
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata") # Already existing

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

def query_vector_store(store_name, query, embeddings, db_dir, search_type, params):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Query vector store {store_name} ---")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

        retriever = db.as_retriever(search_type=search_type, search_kwargs=params)

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

query = "who is Chief Archie?"

# Query the vector store

print("\n***** Using Similarity Search *****\n")
# finds the most similar documents based on cosine similarity. "k" sets the "k" most similar documents
query_vector_store("chroma_db_with_metadata", query, embeddings, db_dir, "similarity", {"k": 3})

print("\n***** Using Max Marginal Relevance *****\n")
# This method balances between selecting documents that are relevant to the query and diverse
# fetch_k specifies the number of documents to initially fetch based on similarity.
# lambda_multi controls the diversity of the results: 1 for minimum diversity and 0 for maximum diversity
# use when you want to avoid redundancy and retrieve diverse yet relevant documents.
query_vector_store("chroma_db_with_metadata", query, embeddings, db_dir, "mmr",
                   {"k": 3, "fetch_k": 20, "lambda_mult": 0.5})

print("\n***** Using Similarity Score Threshold *****\n")
# This method retrieves documents that exceed a certain similarity threshold.
# score_threshold sets the  minimum similarity score a document must have to be considered relevant.
query_vector_store("chroma_db_with_metadata", query, embeddings, db_dir, "similarity_score_threshold",
                   {"k": 3, "score_threshold": 0.4})


