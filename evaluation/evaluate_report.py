
import re


def extract_urls(text: str) -> set[str]:
    return {
        url.rstrip(".,;:")
        for url in re.findall(
            r"https?://[^\s<>\]\)]+",
            text,
        )
    }


def evaluate_report(state: dict):
    report = state.get("report", "")

    known_urls = {
        paper["url"]
        for paper in state.get("papers", [])
        if paper.get("url")
    }

    cited_urls = extract_urls(report)

    valid_urls = cited_urls & known_urls
    invalid_urls = cited_urls - known_urls

    citation_validity = (
        len(valid_urls) / len(cited_urls)
        if cited_urls
        else 0.0
    )

    verification = state.get("verification", {})

    checks = verification.get("checks", [])

    supported = sum(
        item.get("verdict") == "supported"
        for item in checks
    )

    return {
        "citation_validity": citation_validity,
        "invalid_citations": sorted(invalid_urls),
        "checked_claim_support_rate": (
            supported / len(checks)
            if checks else 0.0
        ),
        "claims_checked": len(checks),
        "research_rounds": state.get("rounds", 0),
        "verification_attempts": state.get(
            "verification_attempts", 0
        ),
        "report_status": state.get("report_status"),
    }
