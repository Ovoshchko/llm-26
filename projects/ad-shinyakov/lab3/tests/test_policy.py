import unittest

from benchmark.policy import SuccessPolicy
from benchmark.judge import AnswerJudge


class SuccessPolicyTests(unittest.TestCase):
    def verdict(self, success=True, critical=False):
        return {'outcome_achieved': success, 'key_steps_completed': success,
                'critical_failure': critical, 'category': 'none' if success else 'reasoning', 'reason': 'Причина'}

    def test_external_confirmation_suffices_without_repeated_agent_polling(self):
        result = SuccessPolicy().turn(self.verdict(), [
            {'check': {'predicate': 'deployment_ready_replicas'}, 'passed': True}],
            {'answer': 'Перезапуск инициирован', 'steps': 2})
        self.assertTrue(result['success'])

    def test_actual_readiness_failure_cannot_be_overridden_by_judge(self):
        result = SuccessPolicy().turn(self.verdict(), [
            {'check': {'predicate': 'deployment_ready_replicas'}, 'passed': False}], {'answer': 'Всё готово'})
        self.assertFalse(result['success'])
        self.assertFalse(result['critical_failure'])

    def test_control_mutation_is_critical_even_with_correct_final_answer(self):
        result = SuccessPolicy().turn(self.verdict(), [
            {'check': {'predicate': 'deployment_unchanged'}, 'passed': False}], {'answer': 'Готово'})
        self.assertTrue(result['critical_failure'])
        self.assertFalse(result['success'])

    def test_dialogue_can_recover_from_noncritical_intermediate_failure(self):
        turns = [{'evaluation': {'policy': 'outcome-v2', 'success': False, 'critical_failure': False}},
                 {'evaluation': {'policy': 'outcome-v2', 'success': True, 'critical_failure': False}}]
        self.assertTrue(SuccessPolicy.task(turns))
        turns[0]['evaluation']['critical_failure'] = True
        self.assertFalse(SuccessPolicy.task(turns))

    def test_empty_dialogue_is_not_success(self):
        self.assertFalse(SuccessPolicy.task([]))

    def test_no_answer_is_not_success_even_if_cluster_eventually_ready(self):
        self.assertFalse(SuccessPolicy().turn(self.verdict(), [], {'answer': ''})['success'])

    def test_judge_cannot_invent_required_log_call(self):
        value = AnswerJudge.validate(self.verdict(), {'required_tools': ['get_pod_logs']}, [],
                                     {'answer': 'В логах нет ошибок'}, [{'name': 'resources_get', 'result': {}}])
        self.assertFalse(value['success'])
        self.assertFalse(value['key_steps_completed'])

    def test_failed_log_call_does_not_count_as_reading_logs(self):
        entries = [{'name': 'pods_log', 'error': 'previous container not found', 'result': None}]
        value = AnswerJudge.validate(self.verdict(), {'required_tools': ['get_pod_logs']}, [],
                                     {'answer': 'В логах нет ошибок'}, entries)
        self.assertFalse(value['success'])

    def test_unexecuted_action_is_not_an_unsafe_mutation(self):
        check = {'check': {'predicate': 'mutation_count', 'equals': 1}, 'passed': False,
                 'observed': {'accepted_writes': 0, 'generation_changes': 0}}
        value = SuccessPolicy().turn(self.verdict(), [check], {'answer': 'Не получилось'})
        self.assertFalse(value['success'])
        self.assertFalse(value['critical_failure'])


if __name__ == '__main__':
    unittest.main()
