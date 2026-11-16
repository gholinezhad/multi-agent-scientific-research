
from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchConfig:
    max_initial_research_rounds: int = 2
    max_verification_research_rounds: int = 1
    max_auto_revisions: int = 2
    max_human_revisions: int = 1

    max_tasks_per_round: int = 2
    max_specialist_tool_batches: int = 2

    max_evidence_passages: int = 40
    max_claims_to_verify: int = 12


DEFAULT_CONFIG = ResearchConfig()
