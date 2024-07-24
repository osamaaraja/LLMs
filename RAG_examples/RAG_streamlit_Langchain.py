import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import SequentialChain
from langchain_community.vectorstores import Chroma
from langchain.schema import SystemMessage

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPENAI_API_KEY in your .env file.")

# Initialize ChatOpenAI model
model = ChatOpenAI(model="gpt-4", api_key=openai_api_key)
system_message = SystemMessage(content="you are a helpful AI assistant")

# Set up paths
db_dir = os.path.join("../", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Function to query the vector store
def query_vector_store(query, db):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    try:
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        for i, doc in enumerate(relevant_docs, 1):
            st.write(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                st.write(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    except Exception as e:
        st.warning(f"An error occurred: {e}")

# Function to generate summary and metadata
def generate_summary(book_name):
    prompt_1 = PromptTemplate(
        input_variables=["book_name"],
        template="Summarize the book '{book_name}' in one concise paragraph based on the data provided in the book."
    )
    # Create an LLMChain with the prompt template and the model
    chain_name = LLMChain(llm=model, prompt=prompt_1, output_key="summary")

    # Define the second query template
    prompt_2 = PromptTemplate(
        input_variables=["book_name"],
        template="Based on the book name '{book_name}', provide the metadata related to the book if available."
    )
    # Create an LLMChain with the prompt template and the model
    chain_menu = LLMChain(llm=model, prompt=prompt_2, output_key="meta_data")

    # Define the sequential chain with input variables and output variables
    chain_sequence = SequentialChain(
        chains=[chain_name, chain_menu],
        input_variables=["book_name"],
        output_variables=["summary", "meta_data"]
    )

    response = chain_sequence({"book_name": book_name})

    # Extract only the necessary responses
    summary = response["summary"]
    meta_data = response["meta_data"]

    # Handle unwanted responses
    if "Without the details or the title of the book" in summary:
        summary = "Summary not available."

    return summary, meta_data
# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Streamlit UI
st.title("Pocket Library")
book_name = st.sidebar.selectbox("Book options", ("Eight Cousins", "Heart of Darkness", "Jungle Book", "Old Christmas", "The Haunted Book Shop"))

query = st.text_input("Write your query here")

if query:
    query_vector_store(query, db)
else:
    book_summary, meta_info = generate_summary(book_name)
    st.header("At a glance:\n\n")
    st.write(f"About the book:\n\n{book_summary}")
    st.write(f"Details:\n\n{meta_info}")
