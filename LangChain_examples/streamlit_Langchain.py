import streamlit as st # a library that allows proof of concept application
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
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

st.title("Lets build your restaurant")
cuisine = st.sidebar.selectbox("Choose your Cuisine", ("Arabic", "Turkish", "German", "Pakistani", "Mexican"))


def generate_name(cuisine):
    # Chain 1: Restaurant name based on cuisine

    prompt_1 = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest a 10 fancy names for it."
    )

    # Create an LLMChain with the prompt template and the model
    chain_name = LLMChain(llm=model, prompt=prompt_1, output_key="restaurant_names")

    # Chain 2: Menu items based on cuisine

    # Define the second query template
    prompt_2 = PromptTemplate(
        input_variables=["cuisine"],
        template="Based on the cuisine '{cuisine}', suggest some menu items."
    )

    # Create an LLMChain with the prompt template and the model
    chain_menu = LLMChain(llm=model, prompt=prompt_2, output_key="menu_suggestions")

    # Define the sequential chain with input variables and output variables
    chain_sequence = SequentialChain(
        chains=[chain_name, chain_menu],
        input_variables=["cuisine"],
        output_variables=["restaurant_names", "menu_suggestions"]
    )

    response = chain_sequence({"cuisine": cuisine})

    # Extract only the necessary responses
    restaurant_names = response["restaurant_names"]
    menu_suggestions = response["menu_suggestions"]

    return restaurant_names, menu_suggestions


if cuisine:
    restaurant_names, menu_suggestions = generate_name(cuisine)
    st.header(restaurant_names.strip())
    menu_items = [item.strip() for item in menu_suggestions.split(",")]
    st.write("***** MENU ITEM *****")
    st.write(menu_items)

# TO RUN THE CODE:
# streamlit run streamlit_Langchain.py