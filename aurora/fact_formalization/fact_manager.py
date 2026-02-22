"""
AURORA Module 1 — Fact Manager

Manages target facts, forget sets, and retain sets.
Generates direct and indirect prompts for unlearning evaluation.
"""

from __future__ import annotations

from typing import Optional

from aurora.types import TargetFact, PromptTarget, ForgetSet, RetainSet
from aurora.utils.logging import fact_logger

logger = fact_logger()


# ─── Prompt Templates ────────────────────────────────────────────────────

DIRECT_TEMPLATES = [
    "What is {subject}'s {relation}?",
    "Tell me about {subject}'s {relation}.",
    "{subject} {relation}",
    "The {relation} of {subject} is",
    "Who is the {relation} of {subject}?",
    "Regarding {subject}, what is the {relation}?",
]

INDIRECT_TEMPLATES_HOP1 = [
    "What do you know about {subject}?",
    "Tell me everything about {subject}.",
    "Describe {subject} in detail.",
    "Can you summarize what you know about {subject}?",
]

INDIRECT_TEMPLATES_HOP2 = [
    "If someone has a {relation} of {object}, who might that be?",
    "Who is associated with {object} through a {relation} relationship?",
    "What entities have {object} as their {relation}?",
]

INDIRECT_TEMPLATES_COMPOSITIONAL = [
    "First, who is {subject}? Then, what is their {relation}?",
    "I know that {subject} exists. What can you tell me about their {relation}?",
    "Let's think step by step about {subject}. What is their {relation}?",
    "Consider {subject}. Now, reason about their connections — what {relation} do they have?",
]


class FactManager:
    """
    Manages target facts and generates prompt sets for unlearning.

    Usage:
        fm = FactManager()
        fm.add_target_fact("Eiffel Tower", "location", "Paris")
        forget_set = fm.get_forget_set()
        retain_set = fm.get_retain_set(retain_prompts)
    """

    def __init__(self):
        self.target_facts: list[TargetFact] = []
        self.forget_sets: list[ForgetSet] = []
        self._retain_prompts: list[PromptTarget] = []

    def add_target_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        additional_direct_prompts: Optional[list[str]] = None,
        additional_indirect_prompts: Optional[list[str]] = None,
    ) -> ForgetSet:
        """
        Register a target fact for unlearning and auto-generate prompts.

        Args:
            subject: Subject entity (e.g., "Eiffel Tower").
            relation: Relation type (e.g., "location").
            obj: Object value (e.g., "Paris").
            additional_direct_prompts: Extra direct prompts to add.
            additional_indirect_prompts: Extra indirect prompts to add.

        Returns:
            ForgetSet containing all generated prompts.
        """
        fact = TargetFact(subject=subject, relation=relation, obj=obj)
        self.target_facts.append(fact)

        logger.info(f"Registered target fact: {fact.fact_id} — {fact.to_natural_language()}")

        prompts = []

        # Generate direct prompts
        for template in DIRECT_TEMPLATES:
            prompt_text = template.format(subject=subject, relation=relation)
            prompts.append(PromptTarget(
                prompt=prompt_text,
                target_answer=obj,
                is_direct=True,
                hop_distance=0,
            ))

        # Generate 1-hop indirect prompts
        for template in INDIRECT_TEMPLATES_HOP1:
            prompt_text = template.format(subject=subject)
            prompts.append(PromptTarget(
                prompt=prompt_text,
                target_answer=obj,
                is_direct=False,
                hop_distance=1,
            ))

        # Generate 2-hop indirect prompts
        for template in INDIRECT_TEMPLATES_HOP2:
            prompt_text = template.format(
                relation=relation, object=obj, subject=subject
            )
            prompts.append(PromptTarget(
                prompt=prompt_text,
                target_answer=subject,
                is_direct=False,
                hop_distance=2,
            ))

        # Generate compositional multi-hop prompts
        for template in INDIRECT_TEMPLATES_COMPOSITIONAL:
            prompt_text = template.format(subject=subject, relation=relation)
            prompts.append(PromptTarget(
                prompt=prompt_text,
                target_answer=obj,
                is_direct=False,
                hop_distance=2,
            ))

        # Add user-provided prompts
        if additional_direct_prompts:
            for p in additional_direct_prompts:
                prompts.append(PromptTarget(
                    prompt=p, target_answer=obj, is_direct=True, hop_distance=0
                ))

        if additional_indirect_prompts:
            for p in additional_indirect_prompts:
                prompts.append(PromptTarget(
                    prompt=p, target_answer=obj, is_direct=False, hop_distance=1
                ))

        forget_set = ForgetSet(fact=fact, prompts=prompts)
        self.forget_sets.append(forget_set)

        logger.info(
            f"Generated {len(forget_set.direct_prompts)} direct + "
            f"{len(forget_set.indirect_prompts)} indirect prompts"
        )

        return forget_set

    def add_retain_prompts(self, prompts: list[tuple[str, str]]) -> None:
        """
        Add prompts to the retain set.

        Args:
            prompts: List of (prompt_text, expected_answer) tuples.
        """
        for prompt_text, answer in prompts:
            self._retain_prompts.append(PromptTarget(
                prompt=prompt_text,
                target_answer=answer,
                is_direct=True,
                hop_distance=0,
            ))
        logger.info(f"Added {len(prompts)} retain prompts (total: {len(self._retain_prompts)})")

    def get_forget_sets(self) -> list[ForgetSet]:
        """Return all registered forget sets."""
        return self.forget_sets

    def get_combined_forget_prompts(self) -> list[PromptTarget]:
        """Return all prompts from all forget sets combined."""
        all_prompts = []
        for fs in self.forget_sets:
            all_prompts.extend(fs.prompts)
        return all_prompts

    def get_retain_set(self) -> RetainSet:
        """Return the retain set."""
        return RetainSet(prompts=self._retain_prompts)

    def get_all_direct_prompts(self) -> list[PromptTarget]:
        """Return only direct prompts across all forget sets."""
        return [p for fs in self.forget_sets for p in fs.direct_prompts]

    def get_all_indirect_prompts(self) -> list[PromptTarget]:
        """Return only indirect/multi-hop prompts across all forget sets."""
        return [p for fs in self.forget_sets for p in fs.indirect_prompts]

    def summary(self) -> dict:
        """Return a summary of registered facts and prompt counts."""
        return {
            "num_target_facts": len(self.target_facts),
            "num_forget_sets": len(self.forget_sets),
            "total_direct_prompts": len(self.get_all_direct_prompts()),
            "total_indirect_prompts": len(self.get_all_indirect_prompts()),
            "num_retain_prompts": len(self._retain_prompts),
            "facts": [f.to_dict() for f in self.target_facts],
        }
