
'''
    Extremely simplified agent flow without LLM models.
    This strips all AI-tools from the intelligent-agent concept, focussing on the agent's behaviour.

    Conditional edges have a function which the agent calls, passing it's state and the function returns the node that the agent moves to next.
'''


from langgraph.graph import END, StateGraph
from typing import List, TypedDict


# Agents state
class AgentState(TypedDict):
    marks: List[int]

def add_x(state: AgentState) -> AgentState:    
    state["marks"] += ["X"]
    print(f"Adding new mark to the list. After update: {state["marks"]}")
    return state

# When agent reached edge with this function, it either goes back to "add" node
# or continues to the END node
def is_done(state):
    if len(state["marks"]) > 5:
        return END
    return "add"


# Create a graph with the state
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("add", add_x)

# Edges
workflow.add_conditional_edges("add", is_done)

# Set entry point
workflow.set_entry_point("add")

# Build the graph
graph = workflow.compile()

print(f"\nStarting first run with [X]")
res = graph.invoke({"marks": ["X"]})
print(res)
print(res["marks"])

print(f"\nStarting second run with [XXXXXXXXXXXXXX]")
res = graph.invoke({"marks": ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]})
print(res)
print(res["marks"])


# The graph will look like this:
#                  +-----------------+
#                  | Start (Entry)   |
#                  +--------+--------+
#                           |
#                           v
#                  +--------+--------+
#                  |  "add" Node     |<---+
#                  |  (add_x)        |    |
#                  +--------+--------+    |
#                           |             |
#                    (if len(list) <= 5)--+
#                        (else)-+
#                               |
#                               v
#                  +------------+-----+
#                  |       END        |
#                  +------------------+


# Print the whole state
