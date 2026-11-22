#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from expert_search import (
    Slot,
    State,
    all_integer_candidates,
    apply_action,
    oracle_search,
    search,
)


SCORE_COPY = 2


def score(state, cost=SCORE_COPY):
    return state.max_time + cost * state.copies


def assert_conserved(test, initial, state):
    before = sum(sum(row) for row in initial)
    test.assertEqual(state.task_count, before)
    test.assertTrue(all(n >= 0 for row in state.counts for n in row))
    test.assertEqual(len({a.dst_slot for a in state.actions}), len(state.actions))
    for action in state.actions:
        test.assertGreater(action.q, 0)
        test.assertNotEqual(state.slots[action.dst_slot].owner, action.src_rank)
        test.assertEqual(state.slots[action.dst_slot].layer, state.layer)


class ExpertSearchTests(unittest.TestCase):
    def test_intersection_candidates_fix_q4_while_legacy_misses(self):
        counts = ((12,), (4,))
        slots = (Slot(1),)
        layer = 49
        legacy = search(
            counts,
            slots,
            target_layer=layer,
            max_actions=1,
            exhaustive=True,
            quota_mode="legacy",
        )
        fixed, stats = search(
            counts,
            slots,
            target_layer=layer,
            max_actions=1,
            beam_width=64,
            return_stats=True,
        )
        complete = search(
            counts,
            slots,
            target_layer=layer,
            max_actions=1,
            exhaustive=True,
            quota_mode="all",
        )
        oracle = oracle_search(
            counts, slots, target_layer=layer, max_actions=1, cost_per_copy=2
        )
        self.assertEqual(legacy.actions, ())
        self.assertEqual(fixed.actions[0].q, 4)
        self.assertEqual(complete.actions[0].q, 4)
        self.assertEqual(oracle.actions[0].q, 4)
        self.assertAlmostEqual(score(fixed), 10.5)
        self.assertAlmostEqual(score(complete), 10.5)
        self.assertAlmostEqual(score(oracle), 10.5)
        self.assertGreaterEqual(stats["deduplicated"], 0)
        assert_conserved(self, counts, fixed)

    def test_multi_action_pair_and_oracle_agree_on_score(self):
        counts = ((40,), (39,), (1,), (1,))
        slots = (Slot(2), Slot(3))
        complete = search(
            counts,
            slots,
            target_layer=49,
            max_actions=2,
            exhaustive=True,
            quota_mode="all",
        )
        oracle = oracle_search(
            counts, slots, target_layer=49, max_actions=2, cost_per_copy=2
        )
        self.assertEqual(len(complete.actions), 2)
        self.assertEqual(len(oracle.actions), 2)
        self.assertAlmostEqual(score(complete), score(oracle))
        self.assertLess(score(complete), 40.25)
        assert_conserved(self, counts, complete)
        assert_conserved(self, counts, oracle)

    def test_beam_width_can_prune_multi_action_pair(self):
        counts = ((40,), (39,), (1,), (1,))
        slots = (Slot(2), Slot(3))
        complete_limited = search(
            counts,
            slots,
            target_layer=49,
            max_actions=2,
            exhaustive=True,
            quota_mode="legacy",
        )
        beam_one = search(
            counts,
            slots,
            target_layer=49,
            max_actions=2,
            beam_width=1,
            quota_mode="legacy",
        )
        self.assertEqual(beam_one.actions, ())
        self.assertEqual(len(complete_limited.actions), 2)
        self.assertLess(score(complete_limited), score(beam_one))

    def test_wrong_expert_requires_copy(self):
        counts = ((12,), (4,))
        slot = Slot(1, expert=9, layer=49, ready=True)
        result = search(
            counts,
            (slot,),
            target_layer=49,
            max_actions=1,
            exhaustive=True,
            quota_mode="all",
        )
        self.assertEqual(result.actions[0].q, 4)
        self.assertTrue(result.actions[0].copy)
        self.assertEqual(result.slots[0].expert, 0)
        self.assertEqual(result.copies, 1)

    def test_busy_slot_cannot_be_used(self):
        counts = ((8,), (9,), (1,))
        slot = Slot(2, expert=0, layer=49, ready=True, busy=True)
        result = search(
            counts,
            (slot,),
            target_layer=49,
            max_actions=1,
            exhaustive=True,
            quota_mode="all",
        )
        self.assertEqual(result.actions, ())
        self.assertEqual(result.slot_counts, (0,))

    def test_partial_owner_set_has_no_dummy_slot(self):
        counts = ((40,), (39,), (1,), (1,))
        slots = (Slot(2), Slot(3))
        state = State(49, counts, slots, (0, 0))
        generated = all_integer_candidates(state)
        self.assertTrue(generated)
        self.assertEqual({action.dst_slot for action in generated}, {0, 1})
        self.assertEqual({a.dst_slot for a in generated if slots[a.dst_slot].owner == 1}, set())

    def test_over_allocation_and_zero_q_rejected(self):
        counts = ((5,), (1,))
        slot = Slot(1)
        state = State(49, counts, (slot,), (0,))
        self.assertIsNone(
            apply_action(state, __import__("expert_search").Action(0, 0, 0, 6, True))
        )
        self.assertIsNone(
            apply_action(state, __import__("expert_search").Action(0, 0, 0, 0, True))
        )
        self.assertEqual(state.task_count, 6)

    def test_zero_net_saving_tie_does_not_copy(self):
        counts = ((4,), (0,))
        slots = (Slot(1),)
        baseline = State(49, counts, slots, (0,), active_expert_cost=0.0)
        result = search(
            counts,
            slots,
            target_layer=49,
            max_actions=1,
            exhaustive=True,
            quota_mode="all",
            cost_per_copy=2,
            active_expert_cost=0.0,
        )
        oracle = oracle_search(
            counts,
            slots,
            target_layer=49,
            max_actions=1,
            cost_per_copy=2,
            active_expert_cost=0.0,
        )
        self.assertAlmostEqual(score(baseline), 4.0)
        self.assertEqual(result.actions, ())
        self.assertEqual(oracle.actions, ())
        self.assertEqual(result.copies, 0)
        self.assertEqual(oracle.copies, 0)

    def test_equivalent_states_are_deduplicated(self):
        counts = ((40,), (39,), (1,), (1,))
        slots = (Slot(2), Slot(3))
        _, stats = search(
            counts,
            slots,
            target_layer=49,
            max_actions=2,
            exhaustive=True,
            quota_mode="all",
            return_stats=True,
        )
        self.assertGreater(stats["deduplicated"], 0)
        self.assertGreaterEqual(stats["generated"], stats["deduplicated"])

    def test_zero_profit_action_is_bypassed(self):
        counts = ((1,), (1,))
        result = search(
            ((1,), (1,)),
            (Slot(1),),
            target_layer=49,
            max_actions=1,
            exhaustive=True,
            quota_mode="all",
        )
        oracle = oracle_search(
            counts, (Slot(1),), target_layer=49, max_actions=1, cost_per_copy=2
        )
        self.assertEqual(result.actions, ())
        self.assertEqual(oracle.actions, ())

    def test_existing_replica_avoids_copy(self):
        counts = ((40,), (1,), (1,))
        slot = Slot(1, expert=0, layer=49, ready=True)
        result = search(
            counts,
            (slot,),
            target_layer=49,
            max_actions=1,
            exhaustive=True,
            quota_mode="all",
        )
        self.assertTrue(result.actions)
        self.assertFalse(result.actions[0].copy)
        self.assertEqual(result.copies, 0)
        self.assertLess(score(result), score(State(49, counts, (slot,), (0,))))

    def test_same_expert_different_layer_is_not_reused(self):
        counts = ((12,), (4,))
        slot = Slot(1, expert=0, layer=48, ready=True)
        result = search(
            counts,
            (slot,),
            target_layer=49,
            max_actions=1,
            exhaustive=True,
            quota_mode="all",
        )
        self.assertEqual(result.actions[0].q, 4)
        self.assertTrue(result.actions[0].copy)
        self.assertEqual(result.slots[0].layer, 49)
        # Run the opposite direction too, proving layer participates dynamically.
        slot50 = Slot(1, expert=0, layer=49, ready=True)
        result50 = search(
            counts,
            (slot50,),
            target_layer=50,
            max_actions=1,
            exhaustive=True,
            quota_mode="all",
        )
        self.assertTrue(result50.actions[0].copy)
        self.assertEqual(result50.slots[0].layer, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
