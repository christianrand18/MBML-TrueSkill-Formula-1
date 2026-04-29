# Handoff Log

Running record of what was actually built in each task — reality vs spec.
DeepSeek appends after each task. Claude reads before writing the next CURRENT_TASK.md.

---

<!-- DeepSeek: append your summary below using the template at the bottom of this file -->

---

## Template

```markdown
## T[ID] — [Task name] — [YYYY-MM-DD]

**Status:** PASSED / FAILED

**Files created/modified:**
- `path/to/file.py` — one line description

**Actual output values (spot checks):**
- n_races: ___
- n_drivers: ___
- n_constructors: ___
- [any other key values the next task depends on]

**Deviations from spec:**
- None
- OR: [what changed and why]

**Anything the next task must know:**
- [e.g. field name changed, tensor shape differs from spec, workaround applied]
```
