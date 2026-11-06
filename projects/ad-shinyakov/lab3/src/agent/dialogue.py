import sqlite3
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from .graph import AgentGraph
from .nodes import AgentNodes
from .state import AgentState


DEFAULT_DATABASE = Path(__file__).resolve().parents[2] / 'data/dialogues.sqlite'

class DialogueSession:
    def __init__(self, nodes: AgentNodes | None = None, *,
                 database: str | Path = DEFAULT_DATABASE, thread_id: str | None = None):
        database = Path(database)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database), check_same_thread=False)
        self.graph = AgentGraph(nodes).build(checkpointer=SqliteSaver(self.connection))
        self.thread_id = thread_id if thread_id is not None else str(uuid4())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.connection.close()

    def new_dialogue(self):
        self.thread_id = str(uuid4())

    def ask(self, text: str) -> AgentState:
        if not text.strip():
            raise ValueError('Сообщение не должно быть пустым')
        return self.graph.invoke(
            {'messages': [HumanMessage(text)]},
            {'configurable': {'thread_id': self.thread_id}},
        )
