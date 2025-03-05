import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Literal
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, HumanMessage
from langchain_community.agent_toolkits import FileManagementToolkit
from pprint import pprint
from tavily import TavilyClient
from langgraph.store.memory import InMemoryStore
load_dotenv()
max_tokens = 2000


# Initialize OpenAI LLM
llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), 
    model="gpt-4o",
    temperature=0.0,
    max_tokens=max_tokens,
    max_retries=2,
    timeout=None
)

@tool 
def llm_tool(
    query: Annotated[str, "The query to search for."]
):
    """A tool to call an LLM model to search for a query""" 
    try:
        result = llm.invoke(query)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return result.content

   
# Research tool
@tool
def research_tool(query: Annotated[str, "The query to search for."]):
    """A tool to call Tavily to perform an online search."""
    
    tavily_tool = TavilySearchResults(max_results=2)

    search_results = tavily_tool.invoke(query)  # Ensure invoke is correct for this tool

    return search_results

# Extractor agent
@tool
def extractor(query: Annotated[str, "The query to search and extract detailed results."]):
    """Performs a search using Tavily and extracts detailed content from the top URLs."""

    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(query, max_results=2)
    urls = []
    # # Extracting content and URL from the response
    for result in response.get("results", []):
        # print("Content:", result.get("content"))
        # print("URL:", result.get("url"))
        urls.append(result.get("url"))
        # print("-" * 50)
    
    response = tavily_client.extract(urls=urls, include_images=True)

    extracted_content = [result["raw_content"] for result in response["results"]]

    # return extracted_content
    return {
            "messages": [HumanMessage(content="Detailed content extracted and saved.", name="extractor")],
            "extracted_data": extracted_content,
            "needs_search": False,
        }
 
# File management tools
file_tools = FileManagementToolkit(
    root_dir=str("./data"),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
read_tool, write_tool, list_tool = file_tools

# Agent state
class AgentState(MessagesState):
    next: str

# Agent roles
members = ["assistant","researcher", "extractor", "answer_drafter", "file_writer"]
options = members + ["FINISH"]

system_prompt = (
    f"""You are a supervisor managing a conversation between the following workers: {members}. Given the user request, respond with the worker to act next. 
    Each worker performs a task and responds with their results and status. When finished, respond with FINISH. Start with the research tool:
    Use The `researcher` worker to gathers online information using Tavily based on the asked query. If the user wants to seek more detail information call the extractor worker. Once
    the worker respond with FINISH. Call the answer drafter worker to organize and structure the information into a clear response. Finally, For file management, use the file writer worker.

    Note: If any follow up question is asked based on previous query always start with the llm worker and see if you can answer it without any tools. Then only call other tools when the LLM tool results are inadequate
    """
)

from typing import Annotated, TypedDict, Literal

class SupervisorState(TypedDict):
    next: Literal[*options]

def supervisor_node(state: AgentState) -> AgentState:
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.with_structured_output(SupervisorState).invoke(messages)
    next_ = response["next"]
    # print(messages)
    return {"next": next_ if next_ != "FINISH" else END}

# llm agent
assistant_agent = create_react_agent(llm, 
                               tools=[llm_tool], 
                               state_modifier="You are a highly-trained research analyst and can provide the user with the information they need. You are tasked with finding the answer to the user's question without using any tools. Answer the user's question to the best of your ability."
)
def assistant_node(state: AgentState) -> AgentState:
    result = assistant_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="assistant")
        ]
    }

# Research agent
research_agent = create_react_agent(llm, tools=[research_tool], state_modifier="You are a researcher. Use Tavily to gather online data and when done return the result with status")
def research_node(state: AgentState) -> AgentState:
    result = research_agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name="researcher")], "urls": result.get("urls", [])}

# Extractor agent
extractor_agent = create_react_agent(llm, tools=[extractor], state_modifier="You are an extractor. Use Tavily's Extract API to extract content from URLs and return the result and status.")
def extractor_node(state: AgentState) -> AgentState:
    result = extractor_agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name="extractor")], "needs_search": False}

# Answer drafter agent
answer_drafter_agent = create_react_agent(llm, tools=[], state_modifier="You are an highly professional answer drafter which Summarize and structure research data and respond with status.")
def answer_drafter_node(state: AgentState) -> AgentState:
    result = answer_drafter_agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name="answer_drafter")], "needs_search": False}

# File writer agent
file_agent = create_react_agent(llm, tools=[write_tool], state_modifier="You are file writer who can manage files  and return the status")
def file_node(state: AgentState) -> AgentState:
    result = file_agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name="file_writer")], "needs_search": False}

# Build graph
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_edge(START, "supervisor")
builder.add_node("assistant", assistant_node)
builder.add_node("researcher", research_node)
builder.add_node("extractor", extractor_node)
builder.add_node("answer_drafter", answer_drafter_node)
builder.add_node("file_writer", file_node)

# Define workflow
for member in members:
    builder.add_edge(member, "supervisor")
builder.add_conditional_edges("supervisor", lambda state: state["next"])


in_memory_store = InMemoryStore()
# Compile graph
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}
graph = builder.compile(checkpointer=memory, store=in_memory_store)
# Draw the graph
try:
    graph.get_graph(xray=True).draw_mermaid_png(output_file_path="graph.png")
except Exception as e:
    print("Not able to save the graph:",e)

def main_loop():
    while True:
        user_input = input("$ ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Have a nice day!")
            break
        for s in graph.stream({"messages": [("user", user_input)]}, config=config):
            pprint(s)
            print("------")
    
            
if __name__ == "__main__":
    main_loop()

