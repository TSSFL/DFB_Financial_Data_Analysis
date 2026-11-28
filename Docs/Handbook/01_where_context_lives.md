# 01 — Where context lives (NEXTGAMIS CLOUD / TSSFL AI)

A fact you learn while fixing something has exactly four possible homes. Choose
wrong and it rots, or it is lost with a laptop.

| Kind of fact | Goes in | Why there |
|---|---|---|
| A rule the code must keep | **a test** | It re-runs and goes red. Prose rots silently |
| Why the code is shaped this way | **the maintained reference** | It explains what a test cannot: the reasoning, the alternatives rejected, the incident |
| What is numbered, shipped or owed | **the backlog** | So work can be referred to by number across years |
| Deployments, people, standing decisions | **`Docs/context/`** | Nothing else holds them |

---

## ⛔ This repository is the one with the least context, and the most exposure

Be honest about the state of this repo, because it is unusual:

| | |
|---|---|
| A maintained reference of its own | **None.** The handover document lives in the **ABS** repository, at `Docs/NEXTGAMIS_Cloud_and_TSSFL_AI.md` |
| A numbered backlog of its own | **None.** Engine changes are numbered in ABS's `Docs/implementations.tex` |
| A test suite of its own | **None.** Parity is proved by running seven reports by hand |
| `Docs/context/` | **Does not exist yet** |
| A staging step before customers | **None.** Pushing `master` *is* the deploy |

⚠️ **That arrangement is itself a trap.** Every pointer above is a path into a
sibling checkout — `/home/tssfl/nextgamis_abs/…` — which is a machine-local
pointer wearing a respectable suit. It survives only as long as both checkouts
sit on one machine under those exact names.

**What to create here, when there is reason to:**

1. `Docs/context/README.md` — the deployment contract in this repo's own words:
   which raw URL is fetched, by what, how often, and what breaks if the file
   does not parse.
2. A short `CLAUDE.md` naming the engine's contract — the entry-column
   functions, the parity requirement, and the fact that a push is a deploy.
3. A parity script, so "seven reports" is a command rather than a ritual.

Until those exist, **read the ABS handover document first**, and copy anything
you learn *into this repository* rather than back into ABS.

---

## ⚠️ Inline the evidence; never a pointer to it

A path like `/home/tssfl/Downloads/Digital_Financial_Bridge-Comprehensive_Report-21_11_2026.html`
is dead the moment the machine changes. The figures copied **out** of it are
not.

This is why the records carry

> `TOTAL MOBILE FLOAT` + `TOTAL BANK FLOAT` came to **9,489,320** against a
> `TOTAL FLOAT` of **13,225,487**, and the **3,736,167** difference was
> `SELCOM 1` to the shilling

rather than a link to the report. That sentence still works years later, to a
reader who never had the file. The same applies to a SageCell URL: a report
opens at `sagecell.sagemath.org/kernel/<id>/files/<name>.html` and **dies with
the kernel** — there is nothing to link to by design.

## One file, one fact, dated

Each file in `Docs/context/` holds **one** fact, with a `name`, a one-line
`description`, a `type` (`user` / `feedback` / `project` / `reference`) and a
date. Convert relative dates to absolute as you write.

Update the existing file rather than adding a second on the same subject, and
**delete a fact that turns out to be false** — a wrong note is more expensive
than a missing one, because it will be believed.

## When the reference and the code disagree

The code wins, and the reference is **wrong and must be corrected**. A
maintained reference that is 95% true is a trap, because the 5% is
indistinguishable from the rest.
