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
from validate_safty import SafeQueryExecutor
from validate_safty import InputValidator, QueryValidator
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
import gradio as gr

conn = sqlite3.connect("courses.db")

# Local LLM setup
os.environ["OPENAI_API_KEY"] = str(os.environ.get("OPENAI_API_KEY"))

llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize database connection
db = SQLDatabase.from_uri("sqlite:///courses.db", sample_rows_in_table_info=0)

def list_to_string(lst):
    if not lst:
        return ""
    elif len(lst) == 1:
        return lst[0]
    else:
        return ", ".join(lst[:-1]) + " and " + lst[-1]

# Provided code from the class
input_validator = InputValidator() 
# .validate_question()
query_validator = QueryValidator(db)
# .validate_sql()


def check_valid_query(input_v):
    if not query_validator.validate_sql(input_v):
        print("SQL Faild")
        return "SELECT * FROM courses WHERE 1 = 0;"

    else:
        return input_v
                
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
selected_keys = ['Instructor', 'Department', 'Program', "Course Title"]
selected_values = {key: column_unique_values[key] for key in selected_keys}

# turn dictionary into str
column_unique_values_str = json.dumps(selected_values)


def print_func(x):
    print(x['query'])
    return x


# SQL generation
# SQL generation prompt
sql_prompt_template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}
Extract the keywords in user's questions from the information provided (schema and dictionary)
Question: {question}
SQL Query:"""

#  Includes specific rules for interpreting department-program relationships and handling course title mismatches.
fine_tune = """\n
                **Please Note!! Follow every rules under**
                1. In this dictionary, the Program value (Bachelor of Science), and (Master of Science) 
                belongs to Department Applied Math & Statistics\n
                Program value Data Science (Bachelor of Science), and Data Science (Master of Science)
                belongs to Department Data Science (DS)

                **2**. For Department 'Computer Science (CS), Computer Networks and Cybersecurity (CNCS)' 'General Education (GE)' and 'Biomedical Science (BMS)'
                - Program is equal to Department, because there is no program under
                - When the question contains program of department above, use department name as program name, ignore the department name in the question

                **3**. When human asks about a course title that is **not in provided dictionary**, 
                - First specify that there is no such information and what you can do is to guess one.
                - THen, use your best knowlege to guess a related one from the provided dicitionary with key course title

                4. Put history in consideration when there is more than one human messages.
                """

system_prompt = """Your are an helpful assistence that helps answer questions about course on Spring Semeter of 2025. \
                If the question is unrelated to courses, please respond with "Sorry, I can only answer questions related to courses for the Spring Semester of 2025."
                All needed information is stored in course.db
                Here is a dictionary storing something values of the database. 
                The key of dictionary is Column name, and the value is a list of values of the database under the corresponding Column name
                
                Dictionary (for summplementary information): \n
                """+column_unique_values_str[1:-2] + fine_tune

# Natural language response generation, interpret sql
response_template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

nl_system = """
\n When there is no information about the specific query, Answer in polite way to apologize that you can answer due to lack of information.
"""

nl_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given an input question and SQL response, convert it to a natural language answer. No pre-amble."+nl_system),
    ("human", response_template),
])


# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# # Updated prompt with memory
memory_prompt_sql = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
    MessagesPlaceholder(variable_name="history"),
    ("human", sql_prompt_template),
])

# Memory-enabled query chain
def save_context(input_output):
    """Save conversation context"""
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]

sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
        history=lambda x: memory.load_memory_variables(x)["history"],
    )
    | memory_prompt_sql
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
    | parse
    | check_valid_query
)

sql_memory_chain = RunnablePassthrough.assign(output=sql_chain) | save_context

# Final chain with memory
complete_chain = (
    RunnablePassthrough.assign(
        query=sql_chain,  history=lambda x: memory.load_memory_variables(x)["history"],)
    | RunnableLambda(print_func)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | nl_prompt
    | llm
    | StrOutputParser()
)

complete_memory_chain = RunnablePassthrough.assign(output=complete_chain) | save_context


def display_conversation():
    """Display the current conversation history"""
    history = memory.load_memory_variables({})["history"]
    print("\nConversation History:")
    print("="*50)
    for msg in history:
        role = "Human" if msg.type == "human" else "Assistant"
        print(f"{role}: {msg.content}\n")


# create gradio interface
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>Hello, I am <strong>Course Query Assistant</strong>!</h1>")  # Centered title
    gr.HTML("<h2 style='text-align: center;'>Start to chat with me for any informations about the courses provided in Spring 2025!</h2>")  # Centered title        

    def respond(message, chat_history):
        if not input_validator.validate_question(message):
            print("Question Failed")
            respond = "Sorry, I can't answer this question because: "+ list_to_string(input_validator.error_messages)
        else:
            respond = complete_memory_chain.invoke({"question": message})

        # bot_message = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
        # respond = complete_memory_chain.invoke({"question": check_valid_question(message)})
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": respond})
        return "", chat_history


    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Ask me informations about courses in Spring 2025")
    button = gr.Button(value="Submit")
    # clear = gr.ClearButton([msg, chatbot])
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    button.click(respond, [msg, chatbot], [msg, chatbot])

demo.launch()


# response1 = complete_memory_chain.invoke({"question": "Which courses are available for the Bachelor of Science in Data Science?"})
# response2 = complete_memory_chain.invoke({"question": "Who is teaching the first course on the list?"})
# response3 = complete_memory_chain.invoke({"question": "Which courses are available in the Data Science department for Spring 2025?”"})
# response4 = complete_memory_chain.invoke({"question": "What are the details of the course introduction to statistic?"})
# response5 = complete_memory_chain.invoke({"question": "How many courses are offered under the Computer Science (Bachelor of Science) program?"})
# response6 = complete_memory_chain.invoke({"question": "Show me the courses taught by Dr. Qu in Spring 2025"})
# response7 = complete_memory_chain.invoke({"question": "Make the table disappear."})

# display_conversation()


# # create gradio interface
# with gr.Blocks() as demo:
#     gr.HTML("<h1 style='text-align: center;'>Hello, I am <strong>Course Query Assistant</strong>!</h1>")  # Centered title
#     gr.HTML("<h2 style='text-align: center;'>Start to chat with me for any informations about the courses provided in Spring 2025!</h2>")  # Centered title        

#     def respond(message, chat_history):
#         if not input_validator.validate_question(message):
#             print("Question Failed")
#             respond = "Sorry, I can't answer this question because: "+ list_to_string(input_validator.error_messages)
#         else:
#             respond = complete_memory_chain.invoke({"question": message})

#         # bot_message = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
#         # respond = complete_memory_chain.invoke({"question": check_valid_question(message)})
#         chat_history.append({"role": "user", "content": message})
#         chat_history.append({"role": "assistant", "content": respond})
#         return "", chat_history


#     chatbot = gr.Chatbot(type="messages")
#     msg = gr.Textbox(placeholder="Ask me informations about courses in Spring 2025")
#     button = gr.Button(value="Submit")
#     # clear = gr.ClearButton([msg, chatbot])
    
#     msg.submit(respond, [msg, chatbot], [msg, chatbot])
#     button.click(respond, [msg, chatbot], [msg, chatbot])

# demo.launch()