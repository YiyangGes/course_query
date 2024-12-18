import os
import json
import pandas as pd
import sqlite3
from langchain_ollama import ChatOllama
from langchain_community.llms import DeepInfra
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

conn = sqlite3.connect("courses.db")

# Local LLM setup
# local_model = ChatOllama(model="qwen2.5:7b")
# local_model = ChatOllama(model = "phi3.5:3.8b-mini-instruct-q8_0")

# API setup (optional)
# os.environ["DEEPINFRA_API_TOKEN"] = str(os.environ.get("DEEPINFRA_API_KEY"))
os.environ["OPENAI_API_KEY"] = str(os.environ.get("OPENAI_API_KEY"))
# deepinfra = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-70B-Instruct")

# api_model = None ### 
# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.deepinfra.DeepInfra.html

# Choose which model to use
# llm = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-70B-Instruct")  # Switch to api_model if using Replicate
# llm = local_model  # Switch to api_model if using Replicate
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize database connection
db = SQLDatabase.from_uri("sqlite:///courses.db", sample_rows_in_table_info=0)

# Helper functions
def get_schema(_):
    """Retrieve database schema"""
    return db.get_table_info()

def run_query(query):
    """Execute SQL query"""
    return db.run(query)

def parse(string):
    """To parse the responce of llm, get the pure SQL out from ```sql <SQL>```"""
    
    if "```" in string:
        string = string.split("```")
        string = string[1].replace("sql","")
        
    return string


# Prepare informations potentially needed for conversation
# Load the data from the database into a pandas DataFrame
df = pd.read_sql("SELECT * FROM courses", conn)

# Select 3 keys from the dictionary: 'Instructor', 'Department', 'program'
column_unique_values = {col: df[col].unique().tolist() for col in df.columns}
selected_keys = ['Instructor', 'Department', 'Program']
selected_values = {key: column_unique_values[key] for key in selected_keys}

# turn dictionary into str
column_unique_values_str = json.dumps(selected_values)

# print(column_unique_values_str)

# SQL generation
# SQL generation prompt
sql_prompt_template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""

system_prompt = """Your are an helpful assistence that helps answer questions about course on Spring Semeter of 2025. \
                If the question is unrelated to courses, please respond with "Sorry, I can only answer questions related to courses for the Spring Semester of 2025."
                All needed information is stored in course.db
                Here is a dictionary storing something values of the database. 
                The key of dictionary is Column name, and the value is a list of values of the database under the corresponding Column name
                
                Dictionary : \n
                """+column_unique_values_str[1:-2]

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
    ("human", sql_prompt_template),
])

# Build query generation chain
sql_generator = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
    | parse
)

# Natural language response generation, interpret sql
response_template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

nl_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given an input question and SQL response, convert it to a natural language answer. No pre-amble."),
    ("human", response_template),
])


def print_func(x):
    print(x['query'])
    return x

# printfunc = RunnableLambda(print_func) # object

# Complete chain with natural language output
complete_chain = (
    RunnablePassthrough.assign(query=sql_generator)
    | RunnableLambda(print_func)
    | RunnablePassthrough.assign( # dictionary
        schema=get_schema,
        response=lambda x: db.run(x["query"]), # db.run exsecute database, x is the input
    )
    | nl_prompt
    | llm
)

# question = "How many courses does Mr. Micheal Zhao teaches"
# question = "What are avaliale Departments?"
question = "Which courses are available in the Biomedical sciemce department for Spring 2025?"
nl_response = complete_chain.invoke({"question": question})
print(nl_response)
