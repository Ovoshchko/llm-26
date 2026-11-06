from pydantic import BaseModel, ConfigDict, Field


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')

    answer: str = Field(min_length=1, description='Полный ответ на русском на все части вопроса по результатам инструментов.')


class ToolResponseFormat:
    def __init__(self, tools):
        self.tools = tools

    def schema(self) -> dict:
        choices = [FinalAnswer.model_json_schema()]
        for tool in self.tools:
            function = tool['function']
            choices.append({
                'type': 'object', 'additionalProperties': False,
                'properties': {'name': {'const': function['name'], 'type': 'string'},
                               'arguments': function['parameters']},
                'required': ['name', 'arguments'],
            })
        return {'oneOf': choices}

    @staticmethod
    def decode_answer(text: str) -> str:
        return FinalAnswer.model_validate_json(text).answer
