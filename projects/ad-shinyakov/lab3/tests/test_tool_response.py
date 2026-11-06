import unittest

from jsonschema import ValidationError, validate

from agent.tool_response import ToolResponseFormat


class ToolResponseTests(unittest.TestCase):
    def test_schema_requires_a_registered_tool_and_its_arguments_or_an_answer(self):
        schema = ToolResponseFormat([{'type': 'function', 'function': {
            'name': 'resources_get', 'parameters': {'type': 'object', 'properties': {
                'name': {'type': 'string'}}, 'required': ['name']}}}]).schema()
        validate({'name': 'resources_get', 'arguments': {'name': 'api'}}, schema)
        validate({'answer': 'Готовы две реплики.'}, schema)
        for invalid in [{'name': 'resources_get', 'arguments': {}},
                        {'name': 'unknown', 'arguments': {'name': 'api'}},
                        {'answer': ''}, {'answer': 'Ответ', 'name': 'resources_get'}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                validate(invalid, schema)


if __name__ == '__main__':
    unittest.main()
