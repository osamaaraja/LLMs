import os
from langchain_openai import OpenAIEmbeddings
import openai
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from taipy.gui import Gui, State

load_dotenv()  # loading the environment variables
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

# Set up paths
db_dir = os.path.join("../", "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")


# Function to query the vector store
def query_vector_store(query, db):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    try:
        relevant_docs = retriever.invoke(query)
        # Collect the relevant results with metadata
        results = []
        for i, doc in enumerate(relevant_docs, 1):
            result = f"Document {i}:\n{doc.page_content}\n"
            if doc.metadata:
                result += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            results.append(result)
        return "\n".join(results)
    except Exception as e:
        return f"An error occurred: {e}"


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


context= ("The following is a conversation with an AI assistant. The assistant is helpful, "
           "creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI."
           " How can I help you today? ")
conversation = {
    "Conversation": ["Who are you?", "Hi! I am GPT-4o. How can I help you today?"]
}
current_user_message = ""

client = openai.Client(api_key=openai_api_key)

def request(state: State, prompt: str) -> str:
    """
    Send a prompt to the GPT API and return the response.

    Args:
        - state: The current state.
        - prompt: The prompt to send to the API.

    Returns:
        The response from the API.
    """
    response = state.client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        model="gpt-4o",
    )
    return response.choices[0].message.content


def send_message(state: State) -> None:
    """
    Send the user's message to the API and update the conversation.

    Args:
        - state: The current state.
    """
    user_message = state.current_user_message.strip()

    keywords = ["Eight Cousins", "Jungle Book", "Heart of Darkness", "Old Christmas", "The Haunted Book Shop"]
    ai_initial_response = ""
    book_info = None

    for keyword in keywords:
        if keyword.lower() in user_message.lower():
            # Check if the message is related to a book in the Chroma DB
            book_info = query_vector_store(user_message, db)
            if book_info:
                # Form the initial part of the AI's response with the database information
                ai_initial_response = f"I found some information that might help:\n\n{book_info}\n\n"
            break

    # Add the user's message to the context
    state.context += f"Human: \n {user_message}\n\n AI:"

    # Send the user's message to the API and get the response
    ai_response = request(state, state.context).replace("\n", "")

    # Concatenate the database response with the AI's response
    full_response = ai_initial_response + ai_response if book_info else ai_response

    # Add the full response to the context for future messages
    state.context += full_response
    # Update the conversation
    conv = state.conversation._dict.copy()
    conv["Conversation"] += [state.current_user_message, full_response]
    state.conversation = conv
    # Clear the input field
    state.current_user_message = ""

def style_conv(state: State, idx: int, row: int) -> str:
    """
    Apply a style to the conversation table depending on the message's author.

    Args:
        - state: The current state of the app.
        - idx: The index of the message in the table.
        - row: The row of the message in the table.

    Returns:
        The style to apply to the message.
    """
    if idx is None:
        return None
    elif idx % 2 == 0:
        return "user_message"
    else:
        return "gpt_message"

page = """
<|{conversation}|table|show_all|style=style_conv|>
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|>
"""


if __name__ == "__main__":
    Gui(page).run(dark_mode=True, title="Taipy Chat")