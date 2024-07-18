'''
conversation with RAG application
'''

import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})

model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt =(
    "Given a chat history and the latest user question"
    "which might reference context in the chat history,"
    "formulate a standalone question which can be understood"
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# create prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"),
     ("human", "{input}")]
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers based on the retrieved context and
# indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use"
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you"
    "don't know. Use three sentences maximum and keep the answer"
    "concise.\n\n{context}"
)


qa_prompt = ChatPromptTemplate.from_messages(
    [("system", qa_system_prompt), MessagesPlaceholder("chat_history"),
     ("human", "{input}")]
)

# create_stuff_documents_chain feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

# Create history aware retriever
# this uses LLM to help reformulate the question based on the chat history
history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

# create a retrieval chain that combines the history aware retriever
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# function to simulate a continual chat
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit": break
        # process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # display the AI response
        print(f"AI: {result['answer']}")
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

if __name__ == "__main__":
    continual_chat()