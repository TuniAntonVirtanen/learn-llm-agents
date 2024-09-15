
'''
    Defines tools for the agent to use. 
    The agent's path is practically one node which does simple a calculation.
'''

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from typing import List, TypedDict

# Decorator declared this function as a tool and can now be user with .invoke()
@tool
def add_tool(a: int, b:int) -> int:
    """Adds a and b"""
    return a + b

# Agents state
class AgentState(TypedDict):
    number_a: int
    number_b: int
    number_sum: int

def add_agent(state: AgentState) -> AgentState:
    # Tools have to be invoked rather than simply called because metadata is passed within langchain
    result = add_tool.invoke({"a": state["number_a"], "b": state["number_b"]})
    state["number_sum"] = result
    return state


# Create a graph with the state
workflow = StateGraph(AgentState)

# Nodes             name     function
workflow.add_node("entry", add_agent)

# Edges            from      to
workflow.add_edge("entry", END)

# Set entry point
workflow.set_entry_point("entry")

# Build the graph
graph = workflow.compile()

# Invoke the graph with the state we want to start with
res = graph.invoke({"number_a": 1,
                    "number_b": 1,
                    "number_sum": 0})

print(res)
print(res["number_a"])
print(res["number_b"])
print(res["number_sum"])

