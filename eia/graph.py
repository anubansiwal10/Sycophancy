"""LangGraph wiring of the four agents (flat graph so every LLD box is a visible, timed node).

START -> [a1 cached | a1 Qwen extraction] -> a2 retrieval (refinement loop) -> a3 critical review -> a4 judge -> END
                                                     ^______________ optional re-plan loop ______________|
"""
from __future__ import annotations

import time
import uuid

from langgraph.graph import END, START, StateGraph

from .agents import critical_reviewer as a3
from .agents import evidence_retriever as a2
from .agents import judge as a4
from .agents import proposal_analyzer as a1
from .state import GraphState


def build_graph(entry: str = "start"):
    """entry='start' runs all agents; entry='judge' re-runs only Agent 4 on stored upstream outputs
    (used by evaluate.py: Agents 1-3 are stance-blind, so one upstream run serves all three prompt variants);
    entry='analyze' runs Agent 1 only and stops (the dashboard then lets a human confirm / edit the extraction);
    entry='retrieval' starts at Agent 2 from a given state['structured_proposal'] (Agents 2-4 run)."""
    after_a1 = END if entry == "analyze" else "a2_retrieval_planner"
    g = StateGraph(GraphState)
    nodes = {
        # agent 1
        "a1_load_cached": a1.load_cached, "a1_input_validator": a1.input_validator, "a1_document_parser": a1.document_parser,
        "a1_prompt_builder": a1.prompt_builder, "a1_llm_claim_extractor": a1.llm_claim_extractor,
        "a1_schema_validator": a1.schema_validator, "a1_requirements_generator": a1.requirements_generator,
        # agent 2
        "a2_retrieval_planner": a2.retrieval_planner, "a2_semantic_search": a2.semantic_search, "a2_ranker": a2.ranker,
        "a2_sufficiency_check": a2.sufficiency_check, "a2_query_refinement": a2.query_refinement,
        "a2_evidence_packager": a2.evidence_packager, "a2_replan": a2.replan,
        # agent 3
        "a3_claim_evidence_mapper": a3.claim_evidence_mapper, "a3_gap_contradiction_detector": a3.gap_contradiction_detector,
        "a3_assumption_framing_challenger": a3.assumption_framing_challenger, "a3_review_formatter": a3.review_formatter,
        # agent 4
        "a4_evidence_aggregator": a4.evidence_aggregator, "a4_judgment_prompt_builder": a4.judgment_prompt_builder,
        "a4_recommendation_generator": a4.recommendation_generator, "a4_output_validator": a4.output_validator,
        "a4_confidence_engine": a4.confidence_engine, "a4_bias_warning": a4.bias_warning,
    }
    for name, fn in nodes.items():
        g.add_node(name, fn)

    if entry == "judge":
        g.add_edge(START, "a4_evidence_aggregator")
    elif entry == "retrieval":
        g.add_edge(START, "a2_retrieval_planner")
    else:
        g.add_conditional_edges(START, a1.route_start, {"a1_load_cached": "a1_load_cached", "a1_input_validator": "a1_input_validator"})
    # --- agent 1
    g.add_edge("a1_load_cached", after_a1)
    g.add_conditional_edges("a1_input_validator", a1.after_validator, {"a1_document_parser": "a1_document_parser", "end": END})
    g.add_edge("a1_document_parser", "a1_prompt_builder")
    g.add_edge("a1_prompt_builder", "a1_llm_claim_extractor")
    g.add_edge("a1_llm_claim_extractor", "a1_schema_validator")
    g.add_conditional_edges("a1_schema_validator", a1.after_schema, {"a1_requirements_generator": "a1_requirements_generator", "a1_prompt_builder": "a1_prompt_builder", "end": END})
    g.add_edge("a1_requirements_generator", after_a1)
    # --- agent 2
    g.add_edge("a2_retrieval_planner", "a2_semantic_search")
    g.add_edge("a2_semantic_search", "a2_ranker")
    g.add_edge("a2_ranker", "a2_sufficiency_check")
    g.add_conditional_edges("a2_sufficiency_check", a2.after_sufficiency, {"a2_query_refinement": "a2_query_refinement", "a2_evidence_packager": "a2_evidence_packager"})
    g.add_edge("a2_query_refinement", "a2_retrieval_planner")
    g.add_edge("a2_evidence_packager", "a3_claim_evidence_mapper")
    g.add_edge("a2_replan", "a2_retrieval_planner")
    # --- agent 3
    g.add_conditional_edges("a3_claim_evidence_mapper", a3.after_mapper, {"a3_gap_contradiction_detector": "a3_gap_contradiction_detector", "a3_review_formatter": "a3_review_formatter"})
    g.add_edge("a3_gap_contradiction_detector", "a3_assumption_framing_challenger")
    g.add_edge("a3_assumption_framing_challenger", "a3_review_formatter")
    g.add_edge("a3_review_formatter", "a4_evidence_aggregator")
    # --- agent 4
    g.add_edge("a4_evidence_aggregator", "a4_judgment_prompt_builder")
    g.add_edge("a4_judgment_prompt_builder", "a4_recommendation_generator")
    g.add_edge("a4_recommendation_generator", "a4_output_validator")
    g.add_conditional_edges("a4_output_validator", a4.after_validator, {"a4_confidence_engine": "a4_confidence_engine", "a4_judgment_prompt_builder": "a4_judgment_prompt_builder"})
    g.add_edge("a4_confidence_engine", "a4_bias_warning")
    g.add_conditional_edges("a4_bias_warning", a4.after_bias, {"a2_replan": "a2_replan", "end": END})
    return g.compile()


_GRAPHS: dict = {}


def get_graph(entry: str = "start"):
    if entry not in _GRAPHS:
        _GRAPHS[entry] = build_graph(entry)
    return _GRAPHS[entry]


def run_pipeline(inputs: dict, entry: str = "start", on_step=None) -> dict:
    """Run the graph. If on_step is given it is called with the full state after every node (used by the dashboard for live progress)."""
    t = time.perf_counter()
    init, cfg = {"trace": [], "errors": [], **inputs}, {"recursion_limit": 60}
    if on_step is None:
        state = get_graph(entry).invoke(init, cfg)
    else:
        state = init
        for state in get_graph(entry).stream(init, cfg, stream_mode="values"):
            on_step(state)
    state["total_seconds"] = round(time.perf_counter() - t, 2)
    state.setdefault("run_id", uuid.uuid4().hex[:8])
    return state


def agent_timings(trace: list) -> dict:
    """Aggregate node timings into per-agent totals (seconds, LLM calls) for the latency table."""
    out: dict = {}
    for t in trace:
        a = t["node"][:2]
        d = out.setdefault(a, {"seconds": 0.0, "llm_seconds": 0.0, "llm_calls": 0})
        d["seconds"] = round(d["seconds"] + t["seconds"], 2)
        d["llm_seconds"] = round(d["llm_seconds"] + t["llm_seconds"], 2)
        d["llm_calls"] += t["llm_calls"]
    return out
