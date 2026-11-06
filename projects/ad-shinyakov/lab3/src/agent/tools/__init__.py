import json
from contextlib import ExitStack

from langchain_core.tools import StructuredTool

from .config import LAB_ROOT, ToolConfig
from .deployments import DeploymentActions
from .retrieval import DocumentationSearch


class AgentTools:
    def __init__(self, config: ToolConfig | None = None):
        self.config = config or ToolConfig()
        self.stack = ExitStack()
        self.search = DocumentationSearch(self.config)
        self.mcp = None

    def build(self) -> list[StructuredTool]:
        if self.mcp is None:
            raise RuntimeError('Use AgentTools as a context manager')
        descriptions = json.loads((LAB_ROOT / 'prompts/tools.json').read_text())
        actions = DeploymentActions(self.mcp)
        handlers = {'search_docs': self.search.search, 'restart_deployment': actions.restart,
                    'scale_deployment': actions.scale, 'rollout_status': actions.rollout}
        local = [StructuredTool.from_function(function, name=name, description=descriptions[name])
                 for name, function in handlers.items()]
        return [*local, *self.mcp.tools]

    def __enter__(self):
        from .mcp import KubernetesMCP

        self.mcp = self.stack.enter_context(KubernetesMCP(self.config))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.search.close()
        finally:
            self.stack.close()
