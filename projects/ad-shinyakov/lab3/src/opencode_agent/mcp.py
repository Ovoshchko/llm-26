import contextlib
import json
import sys
from functools import partial

import anyio
from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult
from pydantic import PrivateAttr

from agent.tools import AgentTools
from benchmark.recording import ToolRecording
from .tracing import TurnContext, tool_observation


class SharedTool(Tool):
    _handler = PrivateAttr()

    def __init__(self, handler):
        schema = handler.args_schema
        parameters = schema if isinstance(schema, dict) else schema.model_json_schema()
        super().__init__(name=handler.name, description=handler.description, parameters=parameters)
        self._handler = handler

    async def run(self, arguments):
        context = TurnContext.current()
        recording = ToolRecording(context.get('evaluation_state'))
        with tool_observation(self.name, arguments, context) as span, contextlib.redirect_stdout(sys.stderr):
            action = partial(self._handler.invoke, arguments)
            result = await anyio.to_thread.run_sync(partial(recording.execute, self.name, arguments, action))
            if span:
                span.update(output=result)
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return ToolResult(content=text)


class ToolServer:
    def run(self):
        with AgentTools() as registry:
            server = FastMCP('agent-lab', mask_error_details=False)
            for handler in registry.build():
                server.add_tool(SharedTool(handler))
            server.run(transport='stdio', show_banner=False)


if __name__ == '__main__':
    ToolServer().run()
