import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent import AgentConfig, AgentNodes, ContextManager, DialogueSession, OllamaModel
from agent.cli import run_chat


class Counter:
    def count(self, messages, tools=()):
        return len(messages)


class DialogueTests(unittest.TestCase):
    def session(self, tools=(), **options):
        config = AgentConfig()
        self.model = OllamaModel(config, tools)
        self.model.client = Mock()
        context = ContextManager(config, token_counter=Counter(), tools=self.model.tools)
        database = options.pop('database', ':memory:')
        session = DialogueSession(AgentNodes(config, context_manager=context, model=self.model, tools=tools),
                                  database=database, **options)
        self.addCleanup(session.close)
        return session

    def test_follow_up_sees_previous_user_and_model_messages(self):
        session = self.session()
        self.model.client.invoke.side_effect = [AIMessage('Запомнил checkout-blue.'), AIMessage('checkout-blue')]
        identifier = session.thread_id
        first = session.ask('Наш Deployment checkout-blue')
        second = session.ask('Какой Deployment?')
        model_input = self.model.client.invoke.call_args.args[0]
        self.assertEqual([m.content for m in model_input[1:]],
                         ['Наш Deployment checkout-blue', 'Запомнил checkout-blue.', 'Какой Deployment?'])
        self.assertEqual(len(second['messages']), 4)
        self.assertEqual(first['steps'], 1)
        self.assertEqual(second['steps'], 1)
        self.assertEqual(session.thread_id, identifier)

    def test_new_dialogue_starts_without_previous_history(self):
        session = self.session()
        self.model.client.invoke.return_value = AIMessage('Ответ')
        session.ask('Старый диалог')
        old_id = session.thread_id
        session.new_dialogue()
        result = session.ask('Новый диалог')
        self.assertNotEqual(session.thread_id, old_id)
        self.assertEqual([m.content for m in result['messages']], ['Новый диалог', 'Ответ'])
        self.assertEqual(len(self.model.client.invoke.call_args.args[0]), 2)

    def test_tool_calls_and_results_survive_between_turns(self):
        @tool(description='Прочитать состояние приложения.')
        def inspect(name: str) -> dict:
            return {'name': name, 'ready': True}

        session = self.session([inspect])
        self.model.client.invoke.side_effect = [
            AIMessage('', tool_calls=[{'id': 'a', 'name': 'inspect', 'args': {'name': 'api'}}]),
            AIMessage('api готов.'), AIMessage('Проверил через inspect.')]
        first = session.ask('Проверь api')
        second = session.ask('Как ты проверил?')
        self.assertEqual(first['steps'], 2)
        self.assertEqual(second['steps'], 1)
        self.assertIsInstance(second['messages'][1], AIMessage)
        self.assertEqual(second['messages'][1].tool_calls[0]['id'], 'a')
        self.assertIsInstance(second['messages'][2], ToolMessage)
        self.assertEqual(second['messages'][2].tool_call_id, 'a')
        self.assertIn(second['messages'][2], self.model.client.invoke.call_args.args[0])

    def test_empty_message_does_not_change_history(self):
        session = self.session()
        with self.assertRaises(ValueError):
            session.ask('   ')
        self.model.client.invoke.assert_not_called()

    def test_reopens_saved_dialogue_from_disk(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / 'nested/dialogues.sqlite'
            with self.session(database=database, thread_id='saved-dialogue') as first:
                self.model.client.invoke.return_value = AIMessage('Запомнил checkout-blue.')
                first.ask('Наш Deployment checkout-blue')
            with self.assertRaises(sqlite3.ProgrammingError):
                first.connection.execute('SELECT 1')
            with self.session(database=database, thread_id='saved-dialogue') as second:
                self.model.client.invoke.return_value = AIMessage('checkout-blue')
                result = second.ask('Какой Deployment?')
                model_input = self.model.client.invoke.call_args.args[0]
                self.assertEqual([m.content for m in model_input[1:]],
                                 ['Наш Deployment checkout-blue', 'Запомнил checkout-blue.', 'Какой Deployment?'])
                self.assertEqual(len(result['messages']), 4)
                self.assertEqual(result['steps'], 1)

    def test_new_dialogue_does_not_delete_saved_history(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / 'dialogues.sqlite'
            with self.session(database=database, thread_id='original') as session:
                self.model.client.invoke.return_value = AIMessage('Первый ответ')
                session.ask('Первый вопрос')
                session.new_dialogue()
                self.model.client.invoke.return_value = AIMessage('Другой ответ')
                result = session.ask('Другой вопрос')
                self.assertEqual(len(result['messages']), 2)
            with self.session(database=database, thread_id='original') as restored:
                self.model.client.invoke.return_value = AIMessage('Продолжение')
                result = restored.ask('Продолжим')
                self.assertEqual([m.content for m in result['messages']],
                                 ['Первый вопрос', 'Первый ответ', 'Продолжим', 'Продолжение'])

    def test_cli_handles_commands_and_does_not_print_old_answer_on_stop(self):
        session = Mock()
        session.ask.return_value = {'messages': [AIMessage('Старый ответ')], 'stop_reason': 'context_limit'}
        with patch('builtins.input', side_effect=['', 'Новый вопрос', '/new', '/exit']), patch('builtins.print') as output:
            run_chat(session)
        session.ask.assert_called_once_with('Новый вопрос')
        session.new_dialogue.assert_called_once_with()
        lines = [call.args[0] for call in output.call_args_list if call.args]
        self.assertTrue(any('Остановка:' in line for line in lines))
        self.assertFalse(any('Старый ответ' in line for line in lines))

    def test_cli_prints_answer_and_handles_unavailable_server(self):
        session = Mock()
        session.ask.side_effect = [ConnectionError('Ollama недоступна'),
                                   {'messages': [HumanMessage('Привет'), AIMessage('Здравствуйте')], 'stop_reason': None}]
        with patch('builtins.input', side_effect=['Вопрос', 'Привет', '/exit']), patch('builtins.print') as output:
            run_chat(session)
        lines = [call.args[0] for call in output.call_args_list if call.args]
        self.assertIn('Агент> Здравствуйте', lines)
        self.assertTrue(any('Ollama недоступна' in line for line in lines))


if __name__ == '__main__':
    unittest.main()
