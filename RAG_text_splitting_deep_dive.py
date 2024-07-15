'''
Going deep into different options when it comes to text splitter.
'''

import os
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter
     )

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        # custom logic for splitting text
        return text.split("\n\n")  # splits into chunks at the end of a paragraph.


def create_vector_store(docs, store_name, embeddings):
    persistent_directory = os.path.join('db', store_name)

    if os.path.exists(persistent_directory):
        print(f"Vector store {store_name} already exists. Skipping creation.")
        return

    if not os.path.exists(persistent_directory):
        os.makedirs(persistent_directory)

    docs = docs[:80] # taking subsample of the chunks to avoid memory constraints
    print(f"\n--- Creating vector store {store_name} ---\n")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print(f"Vector store {store_name} created successfully.")

def split_and_store_documents(docs, embeddings):
    # 1. Character-based Splitting
    print("\n--- Using Character based Splitting ---\n")
    char_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    char_docs = char_splitter.split_documents(docs)
    print("\n--- Document Chunks Information Using Character based Splitting ---")
    print(f"Number of document chunks: {len(char_docs)}")
    create_vector_store(char_docs, "chroma_db_char", embeddings)

    # 2. Sentence based Splitting
    print("\n--- Using Sentence based Splitting ---\n")
    sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_size=200)
    sentence_docs = sentence_splitter.split_documents(docs)
    print("\n--- Document Chunks Information Using Sentence based Splitting ---")
    print(f"Number of document chunks: {len(sentence_docs)}")
    create_vector_store(sentence_docs, "chroma_db_sentence", embeddings)

    # 3. Token based Splitting
    print("\n--- Using Token based Splitting ---\n")
    token_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=0)
    token_docs = token_splitter.split_documents(docs)
    print("\n--- Document Chunks Information Using Token based Splitting ---")
    print(f"Number of document chunks: {len(token_docs)}")
    create_vector_store(token_docs, "chroma_db_token", embeddings)

    # 4. Recursive Character based Splitting (THE MOST USED)
    print("\n--- Using Recursive Character based Splitting ---\n")
    rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    rec_char_docs = rec_char_splitter.split_documents(docs)
    print("\n--- Document Chunks Information Using Recursive Character based Splitting ---")
    print(f"Number of document chunks: {len(rec_char_docs)}")
    create_vector_store(rec_char_docs, "chroma_db_rec_char", embeddings)

    # 5. Custom Splitting
    print("\n--- Using Custom Splitting ---\n")
    custom_splitter = CustomTextSplitter()
    custom_docs = custom_splitter.split_documents(docs)
    print("\n--- Document Chunks Information Using Custom Splitting---")
    print(f"Number of document chunks: {len(custom_docs)}")
    create_vector_store(custom_docs, "chroma_db_custom", embeddings)

load_dotenv()  # loading the environment variables
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "JungleBook.txt")

db_dir = os.path.join(current_dir, 'db')  # not specifying the name of db, will be set dynamically later

if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

split_and_store_documents(documents, embeddings)

def query_vector_store(store_name, query, embeddings, db_dir):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Query vector store {store_name} ---")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 1, "score_threshold": 0.1})

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
query_vector_store("chroma_db_char", query, embeddings, db_dir)
query_vector_store("chroma_db_sentence", query, embeddings, db_dir)
query_vector_store("chroma_db_token", query, embeddings, db_dir)
query_vector_store("chroma_db_rec_char", query, embeddings, db_dir)
query_vector_store("chroma_db_custom", query, embeddings, db_dir)



