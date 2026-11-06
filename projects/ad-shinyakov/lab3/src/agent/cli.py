import argparse
import os

from httpx import HTTPError
from langchain_core.messages import AIMessage
from ollama import ResponseError

from .config import AgentConfig
from .dialogue import DialogueSession
from .graph import AgentGraph
from .nodes import AgentNodes
from .tools import AgentTools


STOP_MESSAGES = {
    'step_limit': 'Достигнут предел обращений к модели.',
    'timeout': 'Превышено время выполнения.',
    'context_limit': 'Текущий ход не помещается в контекст. Начните новый диалог: /new.',
    'invalid_history': 'Нарушена структура истории. Начните новый диалог: /new.',
    'memory_error': 'Не удалось проверить сжатую память. История сохранена; повторите запрос.',
}


def run_chat(session: DialogueSession):
    print('Диалог с GLM. /new — новый диалог, /exit — выход.')
    print(f'ID диалога: {session.thread_id}')
    while True:
        try:
            text = input('Вы> ').strip()
            if text == '/exit':
                return
            if text == '/new':
                session.new_dialogue()
                print(f'Начат новый диалог: {session.thread_id}')
                continue
            if not text:
                continue
            state = session.ask(text)
            reason = state.get('stop_reason')
            if reason:
                print(f'Остановка: {STOP_MESSAGES.get(reason, reason)}')
                continue
            last = state['messages'][-1]
            if isinstance(last, AIMessage) and not last.tool_calls:
                print(f'Агент> {last.text}')
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except (HTTPError, ResponseError, OSError, ValueError) as error:
            print(f'Ошибка: {error}')


def main():
    parser = argparse.ArgumentParser(description='Диалоговый агент Kubernetes')
    parser.add_argument('command', choices=('chat', 'graph'), nargs='?', default='chat')
    parser.add_argument('--model', default=os.environ.get('LLM_MODEL', AgentConfig.model))
    parser.add_argument('--thread-id', help='ID сохранённого диалога для продолжения')
    args = parser.parse_args()
    if args.command == 'graph':
        print(AgentGraph(AgentNodes(AgentConfig(model=args.model))).build().get_graph().draw_mermaid())
        return
    with AgentTools() as registry:
        nodes = AgentNodes(AgentConfig(model=args.model), tools=registry.build())
        with DialogueSession(nodes, thread_id=args.thread_id) as session:
            run_chat(session)
