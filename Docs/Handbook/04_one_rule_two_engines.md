# 04 — One rule, two engines

Three products share rules, and each pair drifts in its own way.

| Pair | What is shared | How it drifts |
|---|---|---|
| **ABS** `src/engine.py` ↔ **CLOUD** `nextgamis_cloud.py` | The whole reporting engine: column lists, totals, reconciliation, report CSS | Two files, edited separately. The commonest and costliest drift |
| **ABS** ↔ **ERP** (`app/core/auth.py`) | Security constants — lockout, PIN expiry, rate-limit windows | Deliberate parity with deliberate exceptions |
| **CLOUD** ↔ **tssfl.com/ai** and the forum `[dfb]` cell | The same file, fetched raw from GitHub at run time | No drift — but **pushing `master` is a deploy** |

---

## ⛔ A fix in one engine is not finished until the other has been read

Drift has run in **both** directions, which is why neither file is the
authority:

- The Daily Snapshot summary-row fault was fixed in ABS and the Cloud was not
  checked. It had the identical fault in vectorised form, and was reported on a
  real Cloud report **after** the ABS fix shipped.
- The Monthly Commission breakdown gained `TOTAL SUPERAGENT …` in CLOUD and not
  in ABS, so for a few hours the two disagreed about a report they are supposed
  to render identically.
- The alphabetical-months fault was fixed in CLOUD and left in ABS for weeks.

**A renderer fault in one is worth checking in the other before the report is
closed.** `grep` the sibling file for the symbol you just changed, every time.

## ⚠️ Mirrored code is not identical code, and the differences are load-bearing

Do not "tidy" these into agreement:

| Difference | Why |
|---|---|
| `_abs_parse_dates` tries `%m/%d/%Y` first, then dayfirst | CLOUD ingests Google Forms/Sheets; ABS ingests operator typing |
| `_DETAIL_FONT_PX` is 17.28 in CLOUD, 18.0 in ABS | Different rendering stacks, measured separately |
| CLOUD has a **legacy** report family beside the ABS-parity layer | Two algebraically-equivalent reconciliation models. The legacy one adjusts ACTUAL by `+cumsum(DEBIT) − cumsum(DEBIT PAID)`; the parity layer puts them in the flows with raw ACTUAL |
| CLOUD must run under **Pyodide in a browser** | No file system, no fontTools, no native wheels. Anything baked in must be a literal |

### ⛔ The legacy family looks broken and is not

`calculate_expected_capital()` does not mention `DEBIT` or `DEBIT PAID`. Reading
that as a missing term, and "fixing" it, was measured: untouched legacy matched
the parity layer to the cent (−0.01 on a figure of −8,431,151); the "fix" moved
them to +219,999.99 and −8,801,151, and maximum divergence went from
**10,300.99 to 370,000**.

`update_operating_capital()` already applies the pair to ACTUAL, so adding them
to EXPECTED double-counts.

**Grepping an expression is not searching a model.** Before changing a formula,
find every place its inputs are adjusted.

## ⚠️ Pushing CLOUD `master` is a deploy

`nextgamis_cloud.py` is fetched from the GitHub raw URL with `cache: 'no-store'`
by the forum's `[dfb]` SageMathCell and by the TSSFL AI pipeline. The engine
alone defines the entry columns. There is no staging step between the push and
a customer's report.

**Seven-report parity on CPython before every push**, and on the live
SageMathCell for anything touching column lists. Keep the `master` and `dev`
worktrees on the same commit.

## ⚠️ ERP parity is a decision, not an accident

ABS deliberately tracks ERP's security constants: recovery 3/12 h, PIN change
3/12 h, backup/restore 3/12 h, lockout 3 fails → 12 h, PIN expiry 90 days,
warning at 75, board report email 3/24 h per report type.

One divergence is deliberate and recorded: ABS's OTP-dispatch caps are
**global** buckets where ERP keys them **per admin username**, so in ABS one
admin exhausting the allowance blocks the others.

**Record a divergence where the constant lives**, or the next person aligning
them will "fix" it.

## The general rule

> When a rule exists in two files, assume only one of them has been edited.

If the duplication is deliberate, write a test that compares the two copies
directly — over the whole table, not a sample.
