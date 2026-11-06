from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from .nodes import AgentNodes
from .routing import after_context, after_limits
from .state import AgentState


class AgentGraph:
    def __init__(self, nodes: AgentNodes | None = None):
        self.nodes = nodes if nodes is not None else AgentNodes()

    def build(self, checkpointer=None):
        graph = StateGraph(AgentState)
        graph.add_node('begin_turn', self.nodes.begin_turn)
        graph.add_node('limits', self.nodes.check_state)
        graph.add_node('context', self.nodes.prepare_context)
        graph.add_node('model', self.nodes.call_model)
        graph.add_node('tools', self.nodes.call_tools)
        graph.add_edge(START, 'begin_turn')
        graph.add_edge('begin_turn', 'limits')
        graph.add_conditional_edges('limits', after_limits, {'context': 'context', 'tools': 'tools', 'stop': END})
        graph.add_conditional_edges('context', after_context, {'model': 'model', 'stop': END})
        graph.add_conditional_edges('model', tools_condition, {'tools': 'tools', '__end__': END})
        graph.add_edge('tools', 'limits')
        compiled = graph.compile(checkpointer=checkpointer)
        return compiled.with_config(recursion_limit=4 * self.nodes.config.max_steps + 6)
