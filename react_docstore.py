import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, 'db')
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

if os.path.exists(persistent_directory):
    print("Loading vector store...")
    # Define the embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
    # Load the existing vector store
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
else:
    raise FileNotFoundError(f"The directory {persistent_directory} does not exists. Please check the path.")

# Create a retriever for querying the vector store
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Initialize a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key, temperature=0)

# Contextualize question prompt
# this system prompt helps AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formluate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
     MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the question. "
    "If you don't know how to answer, just say that you don't know. Use three "
    "sentences maximum and keep the answer concise. \n\n{context}"
)

# create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [("system", qa_system_prompt),
     MessagesPlaceholder("chat_history"),
     ("human", "{input}")
     ]
)

# create a chain to combine documents for question-answering
# create_stuff_documents_chain feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

# create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain  = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# setup react agent with document store
# Load the react doctore prompt
react_docstore_prompt = hub.pull("hwchase17/react")

tools =[
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke({"input": input, "chat_history": kwargs.get("chat_history", [])}), # a custom tool that invokes the chain
        description="useful for when you need to answer questions about the context."
    )
]

# create the react agent with document store retriever
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_docstore_prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, handle_parsing_errors=True)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() =="exit":
        break
    response = agent_executor.invoke({"input": query, "chat_history": chat_history})
    print(f"AI {response["output"]}")

    # update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))



