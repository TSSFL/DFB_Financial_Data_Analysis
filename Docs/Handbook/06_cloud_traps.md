# 06 — CLOUD's own traps

Chapter 02 holds the ecosystem-wide failure patterns and chapter 04 the
ABS↔CLOUD parity rules. These are the ones that belong to *this* repository:
the deploy, the runtime, and the two worktrees.

---

## ⛔ Pushing `master` is a deploy. There is no step between it and a customer

`nextgamis_cloud.py` is fetched from the GitHub **raw URL** with
`cache: 'no-store'` by the forum's `[dfb]` SageMathCell and by the TSSFL AI
"Agency banking (DFB)" pipeline. The engine alone defines the entry columns
(`ABS_ENTRY_HEAD`, `ABS_ENTRY_AMOUNTS`, `ABS_TEXT_FIELDS`,
`abs_entry_columns()`).

So there is no build, no artifact, no release, and **no staging**. A push lands
in the next report anybody runs.

**Before every push:** seven-report parity on CPython, and on the live
SageMathCell for anything touching column lists.

⚠️ A syntax error here is not a failed build — it is a customer opening a
report and getting a traceback.

## ⚠️ The two worktrees must sit on the same commit

`/home/tssfl/DFB_Financial_Data_Analysis` is `master`;
`/home/tssfl/DFB_dev_branch` is `dev`. They are worktrees of one repository,
and the deploy reads `master`.

**Check both before and after a push.** As of 2026-11-23 the dev worktree
carried an uncommitted deletion of `dfb_python_class.py` while master was
clean — harmless to the deploy, but exactly the kind of divergence that gets
committed by accident later.

## ⛔ The runtime is Pyodide in a browser. Anything baked in must be a literal

The same file runs under CPython, under SageMathCell, and under **Pyodide in
the viewer's browser**. In that last one there is:

- **no file system** — nothing may be read from disk at run time,
- **no native wheels** — no fontTools, no compiled extensions,
- **no network** beyond what the page itself fetches.

This is why the Details-column width table is **baked in as a literal** rather
than read from `LiberationSans-Bold.ttf`: ABS can read the font file beside its
binary, and CLOUD has no font file at all. The table was extracted once with
fontTools — a *development* tool — and the extraction is pinned by a test in
ABS that compares the literal against the shipped TTF.

> If a value can only be obtained from a file, it cannot be obtained here.
> Bake it, and test the baking somewhere that *can* read the file.

## ⚠️ The report is self-contained, and that is a deliberate property

Verified: 1.5 MB, inline `<style>`, **no** `src=`/`href=` resources, fonts
named as system families (`'Liberation Sans', Arial, Helvetica`), the only two
`http` references being footer hyperlinks to tssfl.com.

That is what makes all three delivery routes work — save-and-open, host on
tssfl.com, or send as an attachment. **Adding an external resource would break
all three silently**, because it would still render on the developer's machine.

## ⛔ The SageCell address cannot be changed, and no technique can change it

A report opens at `sagecell.sagemath.org/kernel/<id>/files/<name>.html`. Asked
on 2026-10-01 whether the bar could read `tssfl.com` instead: **it cannot.** A
page may rewrite its own URL only within its own origin
(`history.pushState`); browsers forbid a host change precisely because that is
address-bar spoofing, so any technique achieving it would be the attack the
restriction exists to prevent.

A `<title>` was added and **reverted the same day** — it renamed the browser
tab, which was not what was asked, and cost ten lines in two builders for
nothing.

⚠️ Only a different **delivery route** moves the address. Do not re-litigate
the URL.

## ⚠️ The legacy report family looks broken and is not

`calculate_expected_capital()` does not mention `DEBIT` or `DEBIT PAID`.
Reading that as a missing term and "fixing" it was **measured**: untouched
legacy matched the parity layer to the cent (−0.01 on a figure of
−8,431,151); the "fix" moved them to +219,999.99 and −8,801,151, and maximum
divergence went from **10,300.99 to 370,000**.

`update_operating_capital()` already applies the pair to ACTUAL, so adding them
to EXPECTED double-counts.

> **Grepping an expression is not searching a model.** Before changing a
> formula, find every place its inputs are adjusted.

## ⚠️ Date parsing differs from ABS on purpose

`_abs_parse_dates` tries `%m/%d/%Y` first, then dayfirst, because CLOUD ingests
**Google Forms / Sheets** while ABS ingests **operator typing**. Do not "tidy"
the two into agreement. Likewise `_DETAIL_FONT_PX` is 17.28 here and 18.0 in
ABS — different rendering stacks, measured separately.

## ⚠️ There is no audit trail here, and there should not be

CLOUD and TSSFL AI are **read-only reporting engines**: no write path, no
`audit_trail`, no `log_action` (0 matches). When an ABS change touches the
audit trail, the correct CLOUD action is **none** — record that it was checked
and does not apply, so the next reader does not check again.
