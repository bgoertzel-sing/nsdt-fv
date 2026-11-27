"""T3 (A.5, S0): the hold-out guard raises on each of five injected leaks: proof text, a premise
list, a dream equal to a test statement, a repair pair, and an annotation containing a test proof.
Each leak is injected for every test obligation, with and without the test identifier, and the
guard must not raise on legitimate wake, premise, dream, repair and annotation examples."""
import pytest

from nsdt_fv.data.holdout import HoldoutGuard, LeakError
from nsdt_fv.eval import s0


@pytest.fixture(scope="module")
def t3(stream, tmp_path_factory):
    return s0.run_t3(stream, str(tmp_path_factory.mktemp("t3")))


def test_each_injected_leak_raises(t3, stream):
    lines, _ = t3
    n_test = sum(o["split"] == "test" for o in stream.obligations)
    leaks = [l for l in lines if l["part"] == "leak"]
    assert {l["leak"] for l in leaks} == set(s0.LEAK_TYPES)
    for lt in s0.LEAK_TYPES:
        for variant in ("stealth", "with_id"):
            sub = [l for l in leaks if l["leak"] == lt and l["variant"] == variant]
            assert len(sub) == n_test
            assert all(l["raised"] for l in sub), (lt, variant)


def test_no_false_positives_on_legitimate_data(t3):
    lines, _ = t3
    controls = [l for l in lines if l["part"] == "control"]
    assert len(controls) > 1000
    assert [l for l in controls if l["raised"]] == []


def test_single_leaks_by_hand(stream):
    g = HoldoutGuard(stream.obligations, stream.worlds)
    o = next(x for x in stream.obligations if x["split"] == "test")
    with pytest.raises(LeakError):
        g.check({"kind": "wake", "statement": o["statement"], "input": o["statement"], "target": o["proof"]})
    with pytest.raises(LeakError):
        g.check({"kind": "dream", "statement": o["statement"]})
    with pytest.raises(LeakError) as e:
        g.check({"kind": "wake", "statement": o["statement"], "source": "frontier"})
    assert e.value.rule == "L7"
    assert g.filter_dreams([{"statement": o["statement"]}]) == [] and g.dropped_dreams == 1
    with pytest.raises(LeakError):
        g.check({"kind": "wake", "input": "x"})        # proof-derived data must carry its statement


def test_l2_l4_l6(t3, stream):
    lines, g = t3
    l2 = [l for l in lines if l["part"] == "L2"][0]
    assert l2["edges_removed"] > 0 and l2["removed_all_test_proof_uses"]
    with pytest.raises(LeakError):
        HoldoutGuard.check_eval_order([{"type": "train", "stage": 1}, {"type": "train", "stage": 2},
                                       {"type": "eval", "model_stage": 1}])
    HoldoutGuard.check_eval_order([{"type": "train", "stage": 1}, {"type": "eval", "model_stage": 1},
                                   {"type": "train", "stage": 2}])
    wake = next(x for x in stream.obligations if x["split"] == "wake")
    g6 = HoldoutGuard(stream.obligations, stream.worlds, dev_projects={"toy-lists"})
    ex = {"kind": "wake", "statement": wake["statement"], "source_ids": [wake["id"]]}
    if wake["repo"] != "toy-lists":
        with pytest.raises(LeakError):
            g6.check(ex)
