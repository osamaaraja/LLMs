import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import SequentialChain

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPEN_API_KEY')

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables. Please set the OPEN_API_KEY in your .env file.")

# Initialize ChatOpenAI model
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key, temperature=0.6)

# Define the first query template
query_1 = "I want to open a {shop_type} shop. Suggest a fancy name for it."

prompt_1 = PromptTemplate(
    input_variables=["shop_type"],
    template=query_1
)

# Create an LLMChain with the prompt template and the model
chain_name = LLMChain(llm=model, prompt=prompt_1, output_key="shop_name")

# Define the second query template
query_2 = "Based on the shop name '{shop_name}', suggest some menu items for a {menu_type} shop."

prompt_2 = PromptTemplate(
    input_variables=["shop_name", "menu_type"],
    template=query_2
)

# Create an LLMChain with the prompt template and the model
chain_menu = LLMChain(llm=model, prompt=prompt_2, output_key="menu_suggestions")

# Define the sequential chain with input variables and output variables
chain_sequence = SequentialChain(
    chains=[chain_name, chain_menu],
    input_variables=["shop_type", "menu_type"],
    output_variables=["shop_name", "menu_suggestions"]
)

# Run the sequence with the specific inputs
response = chain_sequence.invoke({"shop_type": "coffee", "menu_type": "breakfast"})
print("Response:\n", response)

# Removing special characters from the response content
cleaned_content = ''.join(char for char in response['menu_suggestions'] if char.isalnum() or char.isspace() or char == '\n')

print("\nCleaned Content:\n")
print(cleaned_content)
