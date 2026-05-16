# Skill conformance tests

This directory holds one test file per skill that asserts the skill's
documented behavior against the live implementation. They're distinct
from the per-module tests under `tests/`:

| Question | Module test | Conformance test |
|---|---|---|
| Does the code work correctly across edge cases? | ✅ | (only for documented edge cases) |
| Does the implementation match what the skill claims? | sometimes | **always** |
| Will the test fail if the skill is renamed or restructured? | no | sometimes (intentional) |
| Will the test fail if a refactor changes internals but not behavior? | sometimes | **no** |

## Naming convention

`test_skill_NN_short_name.py` — the `NN` matches the skill file under
`docs/skills/`. One test file per skill keeps the mapping 1-to-1.

## What each test should assert

For each skill, write a test that:

1. **Imports the documented function/constant verbatim** from the
   location cited in the skill's "Source of truth" header. If the
   import fails, that's a structural drift the conformance check
   catches.

2. **Asserts the documented math/behavior** using concrete values
   the skill mentions. The skill should be the source of test
   data — quote the same numbers the skill quotes.

3. **References the skill section number** in the test docstring so
   when the assertion fails, the reader knows exactly which skill
   section to re-read.

Keep these tests TIGHT — they're meant to flunk on a real spec-vs-code
divergence, not on irrelevant code reshuffling. If a test feels brittle,
it's probably testing implementation rather than spec.

## Integration with SDD checks

These conformance tests complement two CI scripts:

- `scripts/checks/scan_skill_freshness.py` — flags skills whose
  cited source was modified after the footer date
- `scripts/checks/scan_skill_quotes_match.py` — string-matches the
  first code line of each §3 block against the source

The conformance tests *execute* the documented behavior. The two checks
above are static (file-mtime + regex). All three layers together give
fast-feedback drift detection.
