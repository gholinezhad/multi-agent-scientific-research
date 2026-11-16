from research_assistant.verification.routing import decide_next_action

def test_passing_report_exits():
    assert decide_next_action(passed=True, has_research_queries=False, verification_research_rounds=0, auto_revision_count=0) == "publish"

def test_research_budget_separate_from_initial_rounds():
    assert decide_next_action(passed=False, has_research_queries=True, verification_research_rounds=0, auto_revision_count=0) == "research"

def test_research_budget_exhausted_revises():
    assert decide_next_action(passed=False, has_research_queries=True, verification_research_rounds=1, auto_revision_count=0) == "revise"

def test_both_budgets_exhausted_exits():
    assert decide_next_action(passed=False, has_research_queries=True, verification_research_rounds=1, auto_revision_count=2) == "publish"
