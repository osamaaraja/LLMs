import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma

load_dotenv()  # loading the environment variables
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, 'db')
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata") # Already existing

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Defining user question
query = "How can I learn more about LangChain?"

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n***** Relevant Documents *****")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

#################################  COMBINE THE QUERY AND THE RELEVANT DOCUMENT CONTENT #################################
combined_input = ("Here are some document that might help answer the question: "+ query + "\n\nRelevant Documents:\n"
                  + "\n\n".join([doc.page_content for doc in relevant_docs])
                  + "\n\nPlease provide an answer based only on the provided documents."
                    " If the answer is not in the documents state it clearly.")

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

# Define the messages from the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

# Invoke the model
results = model.invoke(messages)
print("\n***** Generated Response *****")
print(f"Content Only:\n{results.content}")