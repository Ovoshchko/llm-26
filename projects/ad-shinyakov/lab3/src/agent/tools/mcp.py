from contextlib import ExitStack
from copy import deepcopy
from functools import partial
import json

from anyio.from_thread import start_blocking_portal
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langchain.mcp import MCPAdapter
from langchain_core.tools import StructuredTool

from .config import LAB_ROOT, ToolConfig
from .resource_summary import ResourceSummary


class KubernetesMCP:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.stack = ExitStack()
        self.portal = None
        self.tools = []
        self.remote_tools = {}
        self.descriptions = json.loads((LAB_ROOT / 'prompts/tools.json').read_text())
        self.summary = ResourceSummary()

    def __enter__(self):
        try:
            self.portal = self.stack.enter_context(start_blocking_portal())
            adapter = MCPAdapter(Client(self._transport(), timeout=self.config.request_timeout))
            self.stack.enter_context(self.portal.wrap_async_context_manager(adapter))
            remote = self.portal.call(adapter.list_tools)
            self.remote_tools = {tool.name: tool for tool in remote}
            self.tools = [self._wrap(tool) for tool in remote if tool.name in self.descriptions]
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.close()

    def _transport(self):
        return StdioTransport(command=str(LAB_ROOT / 'data/bin/kubernetes-mcp-server'),
                              args=['--config', str(LAB_ROOT / 'infra/kubernetes-mcp.toml')],
                              cwd=str(LAB_ROOT), log_file=LAB_ROOT / 'data/mcp.log')

    def _wrap(self, tool):
        def invoke(**arguments):
            return self.call(tool.name, **arguments)

        schema = deepcopy(tool.args_schema)
        for field in schema.get('properties', {}).values():
            field.pop('description', None)
        return StructuredTool(name=tool.name, description=self.descriptions[tool.name], args_schema=schema,
                              func=invoke, metadata={'mcp_server': 'kubernetes'})

    def call(self, tool_name, **arguments):
        tool = self.remote_tools[tool_name]
        content, _ = self.portal.call(partial(tool.coroutine, **arguments))
        text = '\n'.join(block['text'] for block in content if block['type'] == 'text')
        result = self.summary.format(tool_name, text)
        if tool_name in self.descriptions and len(result) > 8000:
            raise ValueError('MCP result exceeds context budget; request fewer resources or log lines')
        return result
