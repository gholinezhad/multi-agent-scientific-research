"""Pure, testable next-step policy for the verification feedback loop."""
from typing import Literal
from research_assistant.config import DEFAULT_CONFIG, ResearchConfig

NextAction = Literal["publish", "research", "revise"]

def decide_next_action(*, passed: bool, has_research_queries: bool,
                       verification_research_rounds: int, auto_revision_count: int,
                       config: ResearchConfig = DEFAULT_CONFIG) -> NextAction:
    if passed:
        return "publish"
    if has_research_queries and verification_research_rounds < config.max_verification_research_rounds:
        return "research"
    if auto_revision_count < config.max_auto_revisions:
        return "revise"
    return "publish"
