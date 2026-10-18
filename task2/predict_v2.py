#!/usr/bin/env python3
"""Version 2 of the agent: v1 (predict.py) plus deterministic rule checks.

Everything else (PDF reading, retrieval, aspect analysis, time budgets,
fallbacks, event logging) is reused from predict.py unchanged, so v1 stays
submittable as is. v2 changes two things:

1. build_case_evidence() additionally runs rule_checks.run_rule_checks() and
   adds its findings as `rule_checks` to the evidence package (logged as a
   `tool_call` plus one `observation` per non-trivial finding).
2. The decision prompt is the "material" prompt (reject only for a
   documented, material violation) extended with how to weigh rule_checks.

Usage is identical to predict.py:
    /app/predict_v2.py --input $inputDataset --output $outputDir
"""
from pathlib import Path
from typing import Any

import predict
from case_tools import case_type_for
from event_logging import log_event
from rule_checks import run_rule_checks

RULE_CHECK_INSTRUCTIONS = (
    "Der Abschnitt rule_checks enthält exakt berechnete Prüfergebnisse (Summen, Beträge, Anzahl "
    "der Angebote, Schwellenwerte), die zuverlässiger sind als eigenes Nachrechnen. Ein Ergebnis mit "
    "severity 'verstoss' ist ein wesentlicher Mangel und führt zur Ablehnung, außer ein Dokument "
    "löst ihn ausdrücklich auf (z. B. eine dokumentierte Begründung für fehlende Angebote oder "
    "nachgereichte Angebote); benenne ihn dann in der Begründung. Ergebnisse mit 'hinweis' musst du "
    "selbst bewerten, 'ok' bestätigt einen Punkt. Fehlende rule_checks bedeuten nicht, dass alles "
    "in Ordnung ist: prüfe die übrigen Punkte wie gewohnt.\n"
)

_build_case_evidence_v1 = predict.build_case_evidence


def build_case_evidence(input_directory: Path, case_id: str) -> dict[str, Any]:
    evidence = _build_case_evidence_v1(input_directory, case_id)
    case_type = case_type_for(case_id)
    try:
        findings = run_rule_checks(case_type, evidence["documents"])
    except Exception as error:
        # The checks are an aid; never let them cost the case its decision.
        log_event("tool_call", tool="rule_checks", input={"case_type": case_type},
                  output=None, status="error", error=str(error))
        findings = []
    else:
        log_event(
            "tool_call",
            tool="rule_checks",
            input={"case_type": case_type, "documents": sorted(evidence["documents"])},
            output={
                "verstoss": [f["check"] for f in findings if f["severity"] == "verstoss"],
                "findings": len(findings),
            },
        )
        for finding in findings:
            if finding["severity"] != "ok":
                log_event("observation", tool="rule_checks", input={"check": finding["check"]}, output=finding)
    evidence["rule_checks"] = findings
    return evidence


def system_prompt_for(case_type: str) -> str:
    prompt = predict.material_system_prompt_for(case_type)
    marker = "Antworte ausschließlich mit einem JSON-Objekt"
    head, tail = prompt.split(marker, 1)
    return head + RULE_CHECK_INSTRUCTIONS + marker + tail


predict.build_case_evidence = build_case_evidence
predict.system_prompt_for = system_prompt_for
predict.PROMPT_VARIANT = "material+rule_checks"


if __name__ == "__main__":
    predict.main()
