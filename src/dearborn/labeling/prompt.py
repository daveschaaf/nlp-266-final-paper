"""Versioned chat prompt for prepared-remark classification."""

from __future__ import annotations

import json

from .dataset import LabeledLabelingExample
from .contracts import LabelingInput


PROMPT_VERSION = "prepared-remark-labeling-v1"

SYSTEM_INSTRUCTION = """You classify one statement from the prepared-remarks portion of a public company earnings call.

The statement is shown with its closest matching statement from the same company's preceding earnings call and their lexical similarity score. Use the comparison to determine whether the current statement is reusable low-value language, call administration, a substantive update expressed in repeated language, or substantively new information.

Choose exactly one of these labels:

- boilerplate
- administrative
- repeated_substantive
- novel_substantive

Definitions:

- boilerplate: Generic reusable disclosure with little company- or period-specific information, including legal, risk, forward-looking-statement, and non-GAAP disclosure language. Repetition alone is not sufficient when the statement contains a meaningful fact.
- administrative: Call management or presentation scaffolding, including introductions, transitions, instructions, navigation, acknowledgments, and rhetorical staging without an independently useful business fact.
- repeated_substantive: A meaningful company-specific fact that resembles or continues the prior-call statement, including an updated or repeated value, date, result, forecast, milestone, investment, or project status.
- novel_substantive: Meaningful company-specific information materially different from the displayed prior-call match, including a new event, product, transaction, result, customer development, or business claim.

First decide whether the current statement contains an independently useful business fact. If it does not, distinguish administrative language from generic disclosure boilerplate. If it is substantive, determine whether the prior-call match expresses the same material fact or recurring claim. Never classify a statement as boilerplate solely because its similarity is high or its form is reused.

Return one JSON object containing only:

{
  "label": "one allowed label",
  "rationale": "one concise sentence explaining the decisive evidence"
}

Do not return Markdown, multiple labels, probabilities, or additional keys. Do not infer facts absent from the two statements. The similarity score is supporting evidence, not a decision threshold."""

DEMONSTRATION_RATIONALES = {
    15: "This is a reusable forward-looking-statement disclaimer without a company- or period-specific business fact.",
    28: "This generic explanation of non-GAAP measures is reusable disclosure rather than a substantive company update.",
    66: "This is generic forward-looking-statement language with no period-specific business information.",
    7: "This is a rhetorical transition that introduces later discussion but contains no business fact itself.",
    64: "This introduces call participants and does not communicate operating or financial information.",
    96: "This directs listeners to presentation materials rather than stating the outlook itself.",
    38: "The company repeats the same investment outlook while updating the expected R&D expense range.",
    61: "The statement continues the same distribution-network fact and reports its ongoing expansion status.",
    92: "Although repeated exactly, the sentence states a material company-specific gross-margin outlook rather than boilerplate.",
    4: "The current statement introduces a specific new Dynamics 365 product capability absent from the prior match.",
    22: "The current statement reports a different completed acquisition and its specific business capability.",
    53: "The current statement supplies a new Copilot adoption milestone not stated by the prior revenue comparison.",
}


def format_labeling_input(example: LabeledLabelingExample | LabelingInput) -> str:
    """Render only the three input fields permitted by the prompt contract."""
    inputs = example if isinstance(example, LabelingInput) else example.input

    return "\n".join(
        (
            "CURRENT STATEMENT:",
            inputs.current_statement,
            "",
            "PREVIOUS-CALL MATCH:",
            inputs.previous_call_match,
            "",
            "SAME-COMPANY SIMILARITY:",
            f"{inputs.same_company_similarity:.4f}",
        )
    )


def build_prompt_messages(
    demonstrations: tuple[LabeledLabelingExample, ...],
    target: LabeledLabelingExample | LabelingInput,
) -> list[dict[str, str]]:
    """Build identical few-shot chat messages for either candidate model."""

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for demonstration in demonstrations:
        rationale = DEMONSTRATION_RATIONALES.get(demonstration.audit_id)
        if rationale is None:
            raise ValueError(
                f"No fixed rationale for demonstration {demonstration.audit_id}."
            )
        messages.extend(
            (
                {
                    "role": "user",
                    "content": "Classify this statement.\n\n"
                    + format_labeling_input(demonstration),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "label": demonstration.human_label,
                            "rationale": rationale,
                        },
                        separators=(",", ":"),
                    ),
                },
            )
        )

    messages.append(
        {
            "role": "user",
            "content": "Classify this statement.\n\n" + format_labeling_input(target),
        }
    )
    return messages
