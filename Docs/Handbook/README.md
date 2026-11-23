# The TSSFL Engineering Handbook — NEXTGAMIS CLOUD / TSSFL AI

**The standard for how context is written down across the TSSFL software
ecosystem** — NEXTGAMIS CLOUD, NEXTGAMIS ABS, NEXTGAMIS ERP, and whatever comes
next.

This is **CLOUD's own copy**. Every software keeps its own: the Handbook is
copied into each repository, never referenced across checkouts, because a path
into a sibling checkout is exactly the machine-local pointer chapter 01 warns
about.

It is written for someone arriving cold — a new engineer, or an AI assistant
with no memory of the project — who must change working financial software
without breaking it. It assumes you have read nothing and remember nothing.

⛔ **This Handbook is not a style guide.** Every rule in it was paid for. Each
one is followed by the incident that bought it, with the real figures, so you
can judge whether it still applies rather than obeying it on faith.

---

## ⛔ Read this before you touch `nextgamis_cloud.py`

**Pushing `master` is a deploy.** The file is fetched from the GitHub raw URL
by the forum's `[dfb]` SageMathCell and by the TSSFL AI pipeline, with
`cache: 'no-store'`. There is no build, no artifact and no staging step between
your push and the next report a customer runs.

Seven-report parity on CPython before every push, and on the live SageMathCell
for anything touching column lists. See [chapter 06](06_cloud_traps.md).

## The chapters

| | What it holds | Reach for it when |
|---|---|---|
| [01 — Where context lives](01_where_context_lives.md) | The four homes for a fact, and **what this repo is missing** | You have learned something and do not know where to put it |
| [02 — The traps](02_traps.md) | Ecosystem-wide failure patterns, with evidence | Always. Read this one first |
| [03 — Proving it works](03_proving_it_works.md) | What counts as evidence; controls; measuring instead of reasoning | You are about to say "verified", "fixed" or "confirmed" |
| [04 — One rule, two engines](04_one_rule_two_engines.md) | Parity between ABS, CLOUD and ERP | You are editing anything that exists in more than one place |
| [05 — Shipping](05_shipping.md) | Builds, packaging, published links, things that cannot be recalled | You are about to produce or publish an artifact |
| [06 — CLOUD's own traps](06_cloud_traps.md) | The deploy, Pyodide, the two worktrees, the legacy family | You are touching this engine at all |

## The one-paragraph version

Write the rule as a **test** if the code must keep it; as **prose in the
maintained reference** if it explains why the code is shaped the way it is; as
a **numbered implementation** if it is work shipped or owed; and in
**`Docs/context/`** if it is a deployment, a person, or a standing decision.
Inline the evidence — never a link to a screenshot on somebody's laptop.
Before claiming something works, **break it and watch it fail**; a check that
cannot fail proves nothing, and neither does one that cannot succeed. When a
rule exists in two files, assume only one of them has been edited.

## ⚠️ This repo has the least context of the three

It has no maintained reference, no numbered backlog, no test suite and no
`Docs/context/` of its own — those currently live in the **ABS** checkout, at
`Docs/NEXTGAMIS_Cloud_and_TSSFL_AI.md` and `Docs/implementations.tex`. Chapter
01 says what to create here, and why the present arrangement is itself one of
the traps. Until then, read the ABS handover document first and copy what you
learn **into this repository**.

## What this Handbook is not

- **Not the engine's contract.** That is the handover document.
- **Not the backlog.** Engine changes are numbered in ABS's
  `Docs/implementations.tex`.
- **Not onboarding prose.** Nothing here is general advice; if a sentence does
  not carry a consequence, delete it.

## Keeping the copies in step

Chapters 02–05 are ecosystem-wide and are **copied verbatim** from the canonical
set. When one of them gains a trap, copy it to every repository that holds a
Handbook — ABS, ERP and CLOUD. Chapters 01 and 06 are per-repository.
