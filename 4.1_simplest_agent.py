
'''
    Extremely simplified agent flow without LLM models.
    This strips all AI-tools from the intelligent-agent concept, focussing on the agent's behaviour.

    Structure:
    - Nodes are stopping points within the graph. When agent enters a node, it calls the function associated with it using the agent's own state
    - Node functions update the agent's state in predetermined manner.
    - Edges are connecting points between two nodes.    

    Flow:
    - Agent is pushed into the graph at entry point.
    - The agent's state is configured within the node.
    - The agent continues in the graph using edges
    - Once the agent reaches END node, it's leave the graph.
'''


from langgraph.graph import END, StateGraph
from typing import List, TypedDict


# Agents state
class AgentState(TypedDict):
    numbers: List[int]

# When agent reaches node with this function, add 1 to it's numbers list
def first_step(state: AgentState) -> AgentState:
    state["numbers"] += [1]
    return state

# When agent reaches node with this function, add 2 to it's numbers list
def second_step(state: AgentState) -> AgentState:
    state["numbers"] += [2]
    return state

# Create a graph with the state
workflow = StateGraph(AgentState)

# Nodes             name     function
workflow.add_node("first", first_step)
workflow.add_node("second", second_step)

# Edges            from      to
workflow.add_edge("first", "second")
workflow.add_edge("second", END)

# Set entry point
workflow.set_entry_point("first")

# Build the graph
graph = workflow.compile()

# Invoke the graph with the state we want to start with
res = graph.invoke({"numbers": [0]})

# The graph will look like this:
#                  +-----------------+
#                  | Start (Entry)   |
#                  +--------+--------+
#                           |
#                           v
#                  +--------+-----------+
#                  | Node: "first"      |
#                  | Func: "first_step" | 
#                  +--------+-----------+
#                           |   Edge: "first" -> "second"
#                           v
#                  +--------+------------+
#                  | Node "second"       |
#                  | Func: "second_step" | 
#                  +--------+------------+
#                           |   Edge: "second" -> END
#                           v
#                  +--------+--------+
#                  |       END        |
#                  +------------------+


# Print the whole state
print(res)
# Print particular fields from the state
print(res["numbers"])