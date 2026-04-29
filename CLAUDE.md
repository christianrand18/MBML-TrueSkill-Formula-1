# F1 PGM — Project Rulebook

**Both agents read this. Neither agent edits it.**  
Claude (Sonnet) owns planning. DeepSeek owns implementation.

**Key files at a glance:**
- `tasks/todo.md` — master checklist (Claude edits only)
- `CURRENT_TASK.md` — active handoff spec (Claude writes, DeepSeek reads and implements)
- `tasks/handoff_log.md` — running log of what was actually built (DeepSeek appends, Claude reads)
- `tasks/report_notes.md` — design decisions for the report (DeepSeek may append, Claude curates)

---

## Project

DTU MBML course project. Three-tier Bayesian PGM to infer latent F1 driver skill and constructor performance from race finishing orders (2011–2024). Deadline: **2026-05-15**.

Full specification: `SPEC.md`  
Detailed task breakdown: `tasks/plan.md`  
Design decisions and report notes: `tasks/report_notes.md`

---

## Tech Stack

- Python 3.13, managed with **uv** (not pip)
- **Pyro** 1.9+ over PyTorch 2.11+
- Pandas 3+, NumPy, Matplotlib/Seaborn for data and visualisation

### Key commands

```bash
# Run all tests
uv run python -m pytest models/pgm_backend/tests/ -v

# Run a specific test file
uv run python -m pytest models/pgm_backend/tests/test_likelihood.py -v

# Run the full pipeline
uv run python -m models.pgm_backend.run_pgm

# Run a specific module
uv run python -m models.pgm_backend.data_preparation
```

---

## Directory Layout

```
CLAUDE.md            ← this file (rulebook, read-only)
CURRENT_TASK.md      ← active implementation spec (Claude writes, DeepSeek reads)
SPEC.md              ← full model specification
tasks/
  todo.md            ← master task checklist (Claude edits only)
  plan.md            ← detailed per-task implementation spec
  report_notes.md    ← design decisions for the report (append-only, see rules below)
data_preprocessing/
  f1_enriched.csv    ← primary dataset (DO NOT MODIFY)
  f1_model_ready.csv ← pre-processed dataset
models/
  pgm_backend/       ← NEW implementation goes here
    __init__.py
    data_preparation.py
    likelihood.py
    model_baseline.py
    model_extended.py
    model_full.py
    inference.py
    posterior.py
    run_pgm.py
    tests/
      test_likelihood.py
      test_synthetic_recovery.py
  pyro_backend/      ← OLD prototype, reference only, do not extend
outputs/
  pgm_model/         ← all output CSVs and plots go here
```

---

## Architecture Constraints

These are invariants. Any implementation that violates them is wrong.

1. **Plackett-Luce likelihood only.** The old `pyro_backend` uses pairwise probit — do not port this pattern. All new models use `plackett_luce_log_prob` from `likelihood.py`.

2. **Sum-to-zero on constructors via reparameterisation.** Sample `c_raw` with shape `(K-1,)`, derive `c = cat([c_raw, -c_raw.sum()])`. The guide never samples `c` directly.

3. **Mechanical DNFs excluded from Plackett-Luce ranking in Models 1 & 2.** They are included as a Bernoulli reliability term in Model 3 only.

4. **AR(1) via cumsum of innovations — no recursive `pyro.sample` loop.** See `tasks/plan.md` Task 6 for the exact pattern.

5. **NUTS on Model 1 only.** Models 2 and 3 are SVI only. NUTS does not scale to their latent dimension.

6. **No grid position.** It is a blocking variable on the causal path. Do not add it.

7. **Constructor rebranding merges:** Apply `CONSTRUCTOR_REMAP` before building integer indices. Key merges: 211→10 (Racing Point→Force India), 117→10 (Aston Martin→Force India), 214→4 (Alpine→Renault), 213→5 (AlphaTauri→Toro Rosso), 215→5, 51→15 (Alfa Romeo→Sauber).

---

## Roles and Responsibilities

### Claude (Sonnet) — Architect and Verifier
- Reads `tasks/todo.md` and `tasks/plan.md`
- Decides which task to implement next
- Writes `CURRENT_TASK.md` with a precise implementation spec
- After DeepSeek finishes: reviews `git diff`, runs tests, checks acceptance criteria
- Updates `tasks/todo.md` when a task passes all criteria
- Curates any DeepSeek additions to `tasks/report_notes.md`
- Writes the next `CURRENT_TASK.md`

### DeepSeek — Implementer
- Reads `CLAUDE.md` (this file) and `CURRENT_TASK.md`
- Implements exactly what `CURRENT_TASK.md` specifies — no more, no less
- Targets only the files listed in `CURRENT_TASK.md → Target Files`
- After implementing: runs the verification commands listed in `CURRENT_TASK.md`
- Appends to `tasks/report_notes.md` if a non-obvious decision was made (see rule below)
- Reports results back (test output, ELBO curve description, any errors)

---

## Coding Behavior

These apply to all code written in this project.

**Think before coding.** State assumptions explicitly before implementing. If multiple interpretations exist, present them — don't pick silently. If something is unclear, stop and ask. Push back when a simpler approach exists.

**Simplicity first.** Minimum code that solves the problem. No features beyond what was asked, no abstractions for single-use code, no error handling for impossible scenarios. If you write 200 lines and 50 would suffice, rewrite it.

**Surgical changes.** Touch only what the task requires. Don't improve adjacent code, comments, or formatting. Match existing style. If you notice unrelated dead code, mention it — don't delete it. Remove only imports/variables/functions that your own changes made unused.

**Goal-driven execution.** Define success criteria before starting. For multi-step tasks, state a brief plan with a verification step for each item. A task is not complete until verification passes.

---

## DeepSeek Hard Constraints

**Violation of any of these is a workflow failure. Stop and flag instead.**

1. **Do NOT edit `tasks/todo.md`.** Only Claude edits this file.
2. **Do NOT implement any task other than the one in `CURRENT_TASK.md`.** Even if you see an easy adjacent fix.
3. **Do NOT touch files not listed in `CURRENT_TASK.md → Target Files`** unless a file must be created as a dependency explicitly described in the task.
4. **Do NOT change the model architecture.** Implement the spec as written.
5. **Do NOT proceed to the next task.** When done, report results and stop.
6. **Do NOT modify data files** (`*.csv`, `uv.lock`, `pyproject.toml`).
7. **If the verification commands fail**, report the failure with the full traceback. Do not attempt to silently fix adjacent code outside the target files.
8. **Do NOT copy code from `models/pyro_backend/`.** It is an old prototype with incorrect patterns (pairwise probit, grid position as covariate, no sum-to-zero constraint). All new code goes in `models/pgm_backend/` and follows the spec in CLAUDE.md → Architecture Constraints.

## Handoff Log Rule

After every task, **before reporting results**, append a summary to `tasks/handoff_log.md` using the template at the bottom of that file. Include:
- Actual output values the next task depends on (real tensor shapes, real counts)
- Any deviations from the spec and why they were necessary
- Anything the next DeepSeek session must know that isn't in the spec

This is not optional. The next DeepSeek session starts with no memory — the handoff log is its only source of truth about what was actually built.

---

## Report Notes Rule

Append to `tasks/report_notes.md` when you make a decision that is **non-obvious and could matter for the report** — for example:
- A deviation from the spec and why it was necessary
- An unexpected edge case in the data and how you handled it
- A tensor shape or dtype constraint that wasn't obvious from the spec
- A numerical stability fix (e.g. adding eps, clamping)

**Do NOT append** for routine implementation choices, variable naming, or anything already covered in `tasks/report_notes.md`.

Use this format at the bottom of the file:

```markdown
## [Task ID] — [Short decision title]

**Decision:** One sentence.

**Reasoning:** Why this was necessary.

**For the report:** What this means for the methods or discussion section.
```
