from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from dotenv import load_dotenv
import os

load_dotenv() # loading the environment variables

openai_api_key= os.getenv('OPEN_API_KEY') # getting the API key from the .env file

model = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)

positive_feedback_prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant."),
                                                             ("human", "Generate a thank you note for this positive feedback: {feedback}")])

negative_feedback_prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant."),
                                                             ("human", "Generate a response addressing this negative feedback: {feedback}")])

escalate_feedback_prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant."),
                                                             ("human", "Generate a message to escalate this feedback to a human agent: {feedback}")])

neutral_feedback_prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant."),
                                                            ("human", "Generate a request for more details for this neutral feedback: {feedback}")])

categorizing_feedback_prompt = ChatPromptTemplate.from_messages([("system", "You are helpful assistant."),
                                                                      ("human", "classify the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}")])

branches = RunnableBranch(
    (lambda x: "positive" in x, positive_feedback_prompt | model | StrOutputParser()), # if there is a positive in the feedback
    (lambda x: "negative" in x, negative_feedback_prompt | model | StrOutputParser()), # if there is a negative in the feedback
    (lambda x: "neutral" in x, neutral_feedback_prompt | model | StrOutputParser()),
    escalate_feedback_prompt | model | StrOutputParser() # this is the default branch in this case
)

classification_chain = categorizing_feedback_prompt | model | StrOutputParser()

chain = classification_chain | branches # first classifying what kind of feedback is given and then based on that generating the response

review = "The product is excellent. I really liked using it and it was very convinent to use."

result = chain.invoke({"feedback": review})
print(result)