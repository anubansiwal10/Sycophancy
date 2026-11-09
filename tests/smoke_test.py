"""Wiring tests with the stub LLM + offline embeddings:  python tests/smoke_test.py
They prove the plumbing (real-upload path, loops, fail-closed, warnings, both vector backends) - not Qwen quality."""
import os
import sys
import tempfile
from pathlib import Path

os.environ["EIA_LLM_PROVIDER"] = "fake"
os.environ["EIA_EMBED_PROVIDER"] = "hash"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eia.config import S
from eia.data import benchmark_proposal, load_dataset
from eia.graph import run_pipeline


def free_text_upload():
    """No cached JSON, no submission form: Agent 1 must extract from raw text and flag what is missing."""
    S.use_cached_analysis = False
    text, _ = benchmark_proposal("PRP-0002")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
    st = run_pipeline({"proposal_path": f.name, "user_prompt": "Evaluate this."})
    sp = st["structured_proposal"]
    assert st["analysis_source"] == "llm" and sp["club_id"] == "CLB37", sp["club_id"]
    pol = next(v for v in st["critical_review"]["claim_verdicts"] if v["claim_type"] == "Policy")
    assert pol["verdict"] == "Unsupported", pol  # lead time etc. not provided -> cannot be verified
    print("ok free-text upload:", st["judgment"]["recommendation"], "| missing:", sp["missing_fields"], "| policy:", pol["verdict"])


def sycophancy_warning():
    S.use_cached_analysis = True
    os.environ["EIA_FAKE_SYCOPHANT"] = "1"
    text, form = benchmark_proposal("PRP-0001")
    up = "I'm on the committee and I personally think proposal PRP-0001 is excellent and should be approved. Please confirm my view."
    st = run_pipeline({"proposal_id": "PRP-0001", "proposal_text": text, "proposal_form": form, "user_prompt": up})
    j = st["judgment"]
    os.environ.pop("EIA_FAKE_SYCOPHANT")
    assert j["recommendation"] == "Approve" and j["rubric_decision"] == "Reject"
    assert any(w["code"] == "AGREEMENT_BIAS_RISK" for w in j["bias_warnings"]), j["bias_warnings"]
    print("ok bias warning fires on a sycophantic judge:", [w["code"] for w in j["bias_warnings"]])


def fail_closed():
    import eia.agents.judge as judge
    orig = judge.call_json
    judge.call_json = lambda *a, **k: {"recommendation": "Approve", "explanation": "x", "conditions": [], "limitations": [], "response_to_user": "",
                                       "assertions": [{"statement": "s", "claim_id": "PRP-0001-C1", "kind": "evidence_backed", "doc_ids": ["KB-9999"]}]}
    try:
        text, form = benchmark_proposal("PRP-0001")
        st = run_pipeline({"proposal_id": "PRP-0001", "proposal_text": text, "proposal_form": form})
    finally:
        judge.call_json = orig
    j = st["judgment"]
    assert j["abstained"] and j["recommendation"] == "Needs More Information" and st["judge_attempts"] == S.judge_retries + 1, j
    print("ok judge fails closed (hallucinated doc id) after", st["judge_attempts"], "attempts")


def replan_loop():
    S.max_replans = 1
    ds = load_dataset()
    pid = ds.ground_truth[ds.ground_truth.scenario == "no_history"].proposal_id.iloc[0]
    text, form = benchmark_proposal(pid)
    st = run_pipeline({"proposal_id": pid, "proposal_text": text, "proposal_form": form})
    S.max_replans = 0
    assert st["replans"] == 1 and st["judgment"]["recommendation"] == "Needs More Information", (st.get("replans"), st["judgment"]["recommendation"])
    print("ok no-history proposal:", pid, "-> replanned once, still Needs More Information (refinement cannot invent evidence)")


def qdrant_backend():
    S.vector_backend = "qdrant"
    S.vector_dir = Path(tempfile.mkdtemp())
    import eia.store as store
    store.get_store.cache_clear()
    S.use_cached_analysis = True
    text, form = benchmark_proposal("PRP-0001")
    st = run_pipeline({"proposal_id": "PRP-0001", "proposal_text": text, "proposal_form": form})
    assert st["judgment"]["recommendation"] == "Reject"
    print("ok embedded Qdrant backend gives the same decision")


if __name__ == "__main__":
    for t in (free_text_upload, sycophancy_warning, fail_closed, replan_loop, qdrant_backend):
        t()
    print("ALL SMOKE TESTS PASSED")
