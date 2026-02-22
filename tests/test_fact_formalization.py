"""
Tests for AURORA Module 1 — Fact Formalization
"""

import pytest
from aurora.types import TargetFact, PromptTarget, ForgetSet
from aurora.fact_formalization.fact_manager import FactManager


class TestTargetFact:
    def test_fact_creation(self):
        fact = TargetFact(subject="Eiffel Tower", relation="location", obj="Paris")
        assert fact.subject == "Eiffel Tower"
        assert fact.relation == "location"
        assert fact.obj == "Paris"
        assert len(fact.fact_id) == 12

    def test_fact_id_deterministic(self):
        f1 = TargetFact(subject="A", relation="B", obj="C")
        f2 = TargetFact(subject="A", relation="B", obj="C")
        assert f1.fact_id == f2.fact_id

    def test_fact_to_natural_language(self):
        fact = TargetFact(subject="Sun", relation="is a", obj="star")
        assert fact.to_natural_language() == "Sun is a star"

    def test_fact_to_dict(self):
        fact = TargetFact(subject="X", relation="Y", obj="Z")
        d = fact.to_dict()
        assert "fact_id" in d
        assert d["subject"] == "X"


class TestFactManager:
    def test_add_target_fact(self):
        fm = FactManager()
        fs = fm.add_target_fact("Eiffel Tower", "location", "Paris")
        assert isinstance(fs, ForgetSet)
        assert len(fs.prompts) > 0

    def test_direct_and_indirect_prompts(self):
        fm = FactManager()
        fs = fm.add_target_fact("Sun", "type", "star")
        assert len(fs.direct_prompts) > 0
        assert len(fs.indirect_prompts) > 0

    def test_add_retain_prompts(self):
        fm = FactManager()
        fm.add_retain_prompts([
            ("Capital of France?", "Paris"),
            ("2 + 2 =", "4"),
        ])
        rs = fm.get_retain_set()
        assert len(rs.prompts) == 2

    def test_summary(self):
        fm = FactManager()
        fm.add_target_fact("A", "B", "C")
        s = fm.summary()
        assert s["num_target_facts"] == 1
        assert s["total_direct_prompts"] > 0

    def test_multiple_facts(self):
        fm = FactManager()
        fm.add_target_fact("A", "B", "C")
        fm.add_target_fact("X", "Y", "Z")
        assert len(fm.get_forget_sets()) == 2

    def test_additional_prompts(self):
        fm = FactManager()
        fs = fm.add_target_fact(
            "Eiffel Tower", "location", "Paris",
            additional_direct_prompts=["Where is the Eiffel Tower?"],
            additional_indirect_prompts=["Tell me about Paris landmarks"],
        )
        total = len(fs.prompts)
        assert "Where is the Eiffel Tower?" in [p.prompt for p in fs.prompts]
