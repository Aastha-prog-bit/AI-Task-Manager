from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st


# ---------------- DATABASE ----------------

db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

db.run("""
    CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK(status IN ('pending', 'in_progress', 'completed'))
            DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")


# ---------------- MODEL ----------------

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview"
)


# ---------------- SQL TOOLS ----------------

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model
)

# IMPORTANT: call the method
tools = toolkit.get_tools()


# ---------------- MEMORY ----------------

# memory = InMemorySaver()


# ---------------- SYSTEM PROMPT ----------------

system_prompt = """
You are a task management assistant connected to a SQLite database.

Your job is to manage the user's tasks using the SQL database tools provided to you.

DATABASE TABLE:

tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT,
    created_at TIMESTAMP
)

Allowed status values:
- pending
- in_progress
- completed

RULES:

1. Always use the SQL tools to read or modify tasks.
2. Never invent task information.
3. For SELECT queries, return at most 10 tasks.
4. When listing tasks, use:
   ORDER BY created_at DESC
   LIMIT 10
5. After INSERT, UPDATE, or DELETE, perform a SELECT query to verify that the operation succeeded.
6. For creating a task, use the title and description provided by the user.
7. If the user doesn't specify a status when creating a task, use 'pending'.
8. For updating or deleting a task, identify it using its id or title.
9. When displaying multiple tasks, present them in a Markdown table.
10. After successfully creating, updating, or deleting a task, clearly tell the user what happened.

The available SQL tools should be used directly to perform the database operations.
"""
# ---------------- AGENT ----------------
@st.cache_resource
def get_agent():  
    agent=create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )
    return agent

agent=get_agent()

st.subheader("TaskBot - Manage your tasks")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# ---------------- CHAT LOOP ----------------

prompt=st.chat_input("Ask me to manage your tasks?")
if prompt:

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})
    
    with st.chat_message("ai"):
        with st.spinner("Processing..."):

    
            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                {
                    "configurable": {
                        "thread_id": "1"
                    }
                }
            )

            result = response["messages"][-1].content
            
            
            if isinstance(result, list):
                result = "\n".join(
                    item["text"]
                    for item in result
                    if item.get("type") == "text"
                )

            st.markdown(result)
