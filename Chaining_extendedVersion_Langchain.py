from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv() # loading the environment variables

openai_api_key= os.getenv('OPEN_API_KEY') # getting the API key from the .env file

model = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)


prompt_template = ChatPromptTemplate.from_messages([("system", "You are a comedian who tells jokes about {topic}."),
                                                    ("human", "Tell me {joke_count} jokes.")]) # messages are used as tuples


uppercase_output = RunnableLambda(lambda x: x.upper()) # capital letters
count_words = RunnableLambda(lambda x: f"word count: {len(x.split())}\n{x}")

# defining the chain
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# invoking the chain
result = chain.invoke({"topic": "humans", "joke_count": 3})
print(result)

# Running multiple chains in parallel

prompt_template = ChatPromptTemplate.from_messages([("system", "You are an expert product reviewer."),
                                                    ("human", "List main features of the product {product_name}.")])

def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages([("system", "You are an expert product reviewer."),
                                                      ("human", "Given these features: {features}, list the pros of these features.")])

    return pros_template.format(features=features)
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages([("system", "You are an expert product reviewer."),
                                                      ("human", "Given these features: {features}, list the cons of these features.")])

    return cons_template.format(features=features)

def combine_pros_cons(pros, cons):
    return f"Pros: {pros}\n, Cons: {cons}"

pros_branch_chain = RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser() # first parallel branch
cons_branch_chain = RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser() # second parallel branch

chain_all = (prompt_template
                  | model
                  | StrOutputParser()
                  | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
                  | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
                  )
result = chain_all.invoke({"product_name": "Macbook Air"})
print(result)