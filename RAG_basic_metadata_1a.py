'''
metadata is helpful in providing the source from where the information is coming from
'''
import os
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
books_dir = os.path.join(current_dir, 'books') # multiple books can be downloaded from https://www.gutenberg.org/
db_dir = os.path.join(current_dir, 'db')
persistent_dir = os.path.join(db_dir, 'chroma_db_with_metadata')

print('Books directory: ', books_dir)
print('persistent directory: ', persistent_dir)

if not os.path.exists(persistent_dir):
    print("Persistent directory does not exists. Initializing vector store ...")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"The directory {books_dir} does not exists. Please check the path.")

    # List all text files in the book directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        books_doc = loader.load()
        for doc in books_doc:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings for the document chunks
    print("\n--- Creating Embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

    # Use only the first few chunks for testing
    docs = docs[:80]
    #filtered_docs = filter_complex_metadata(docs)

    # Create the vector store and persist it automatically
    try:
        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)
        print("\n--- Finished creating vector store ---")
    except Exception as e:
        print(f"Error during vector store creation: {e}")

    print("Script completed successfully.")



