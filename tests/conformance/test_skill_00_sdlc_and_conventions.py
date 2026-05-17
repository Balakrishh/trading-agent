"""Conformance test: skill 00 — SDLC, glossary, and code conventions.

Skill 00 is the meta-skill. It doesn't describe a behavioral primitive;
it documents the process: where specs live, how SDLC works, what CI
gates exist. The conformance assertions here are structural — they
verify the SDLC artefacts that skill 00 mandates actually exist and
have the right shape.

Failure modes caught:
- CLAUDE.md or CONTRIBUTING.md gets deleted/renamed without skill 00 update
- PROJECT_MANIFEST.md disappears
- The five SDD CI gates listed in skill 00's process diagram aren't
  actually present in scripts/checks/
- _template.md (the skill template referenced by specify_new_feature.py)
  is missing or no longer in docs/skills/
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSkill00ProcessArtefacts:
    """Skill 00 mandates specific files as part of the SDLC contract.
    Missing any of them breaks the cross-LLM handoff brief."""

    def test_claude_md_exists(self) -> None:
        """CLAUDE.md is the entry-point brief for any AI session."""
        assert (REPO_ROOT / "CLAUDE.md").is_file()

    def test_contributing_md_exists(self) -> None:
        """CONTRIBUTING.md operationalises CLAUDE.md into commands."""
        assert (REPO_ROOT / "CONTRIBUTING.md").is_file()

    def test_project_manifest_exists(self) -> None:
        """PROJECT_MANIFEST.md is the repo-level orientation doc."""
        assert (REPO_ROOT / "PROJECT_MANIFEST.md").is_file()

    def test_skill_template_exists(self) -> None:
        """docs/skills/_template.md is the boilerplate the specify
        CLI uses to scaffold new skills. Required by specify_new_feature.py."""
        assert (REPO_ROOT / "docs" / "skills" / "_template.md").is_file()

    def test_skills_readme_exists(self) -> None:
        """The skill index — every contributor lands here first."""
        assert (REPO_ROOT / "docs" / "skills" / "README.md").is_file()


class TestSkill00CiGates:
    """Skill 00 documents the five SDD CI gates by name. This pins
    them to actual files under scripts/checks/."""

    def test_invariant_check_present(self) -> None:
        assert (REPO_ROOT / "scripts" / "checks" /
                "scan_invariant_check.py").is_file()

    def test_skill_quotes_match_present(self) -> None:
        assert (REPO_ROOT / "scripts" / "checks" /
                "scan_skill_quotes_match.py").is_file()

    def test_skill_freshness_present(self) -> None:
        assert (REPO_ROOT / "scripts" / "checks" /
                "scan_skill_freshness.py").is_file()

    def test_build_traceability_present(self) -> None:
        assert (REPO_ROOT / "scripts" / "checks" /
                "build_traceability.py").is_file()

    def test_specify_new_feature_present(self) -> None:
        """The Tier-2 scaffolding CLI."""
        assert (REPO_ROOT / "scripts" / "specify_new_feature.py").is_file()
