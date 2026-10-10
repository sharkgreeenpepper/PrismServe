import unittest
from repair_selectors import select_repair


class SelectorTests(unittest.TestCase):
    def test_equal_reusable_budget_with_mandatory_gaps(self):
        scores=[float(i) for i in range(100)]
        reusable=[i%5!=0 for i in range(100)]
        for policy in ('topk','window','document'):
            plan=select_repair(scores,reusable,.10,policy,[(0,30),(30,60),(60,100)])
            self.assertEqual(len(plan.repair_positions),8)
            self.assertEqual(len(plan.mandatory_positions),20)
            self.assertFalse(set(plan.repair_positions)&set(plan.mandatory_positions))
            self.assertFalse(plan.fallback_full)

    def test_window_preserves_contiguity_and_topk_can_scatter(self):
        scores=[100,0,0,99,0,0,98,0,0,97]
        top=select_repair(scores,[True]*10,.3,'topk')
        win=select_repair(scores,[True]*10,.3,'window')
        self.assertEqual(top.repair_positions,(0,3,6))
        self.assertEqual(win.repair_positions,(0,1,2))

    def test_document_budget_remainder_is_a_contiguous_window(self):
        plan=select_repair([9,9,9,1,1,1,1,1],[True]*8,.5,'document',[(0,3),(3,8)])
        self.assertEqual(plan.repair_positions,(0,1,2,3))

    def test_causal_path_requires_coverage_or_full_recompute(self):
        covered=select_repair([1]*20,[True]*20,.2,'edit_window',edit_start=5,dependency_end=8)
        self.assertEqual(covered.repair_positions,(5,6,7,8))
        long=select_repair([1]*20,[True]*20,.2,'edit_window',edit_start=5,dependency_end=12)
        self.assertTrue(long.fallback_full)
        unknown=select_repair([1]*20,[True]*20,.2,'edit_window',edit_start=5)
        self.assertTrue(unknown.fallback_full)

    def test_no_cache_and_bad_scores_are_not_fake_hits(self):
        empty=select_repair([1]*3,[False]*3,.2,'window')
        self.assertEqual(empty.repair_positions,())
        self.assertEqual(empty.mandatory_positions,(0,1,2))
        bad=select_repair([float('nan'),1],[True]*2,.5,'topk')
        self.assertTrue(bad.fallback_full)


if __name__=='__main__':unittest.main()
