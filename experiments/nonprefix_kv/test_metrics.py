"""Regression checks for evaluation semantics, rather than model implementation."""
import unittest
from analyze import loss_upper
from run_engine import parse_answer, plan_order


class EvaluationTests(unittest.TestCase):
    def test_answer_parser_rejects_ambiguous_outputs(self):
        self.assertEqual(parse_answer(' "1234"\n'),'1234')
        self.assertIsNone(parse_answer('1234 or 5678'))
        self.assertIsNone(parse_answer('The amount is 1234.'))
        self.assertEqual(parse_answer('B','choice'),'B')
        self.assertIsNone(parse_answer('B or C','choice'))

    def test_history_changes_order_without_changing_requests(self):
        rows=[dict(session_id='s',request_id=str(i),step=i,input_hash=f'h{i}') for i in range(10)]
        first=list(plan_order(rows,'history',17))[0][1]
        second=list(plan_order(rows,'history',29))[0][1]
        self.assertNotEqual([r['request_id'] for r in first],[r['request_id'] for r in second])
        self.assertEqual(sorted(r['input_hash'] for r in first),sorted(r['input_hash'] for r in second))
        self.assertEqual([r['step'] for r in rows],list(range(10)))

    def test_confidence_interval_respects_session_dependence(self):
        pairs=[]
        for session in range(20):
            for _ in range(10):
                pairs.append(({'session_id':str(session),'correct':True},
                              {'correct':session!=0}))
        # One entirely bad session: uncertainty must exceed mean 5% loss.
        self.assertGreater(loss_upper(pairs),.05)
        self.assertEqual(loss_upper([({'session_id':str(i),'correct':True},{'correct':True})
                                    for i in range(20)]),0)
        self.assertIsNone(loss_upper([({'session_id':'only','correct':True},{'correct':False})]))


if __name__=='__main__':unittest.main()
