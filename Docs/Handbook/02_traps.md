# 02 — The traps

Eighteen failure patterns, each one recorded because it happened here, most
of them more than once. They are ordered by how much they have cost.

---

## 1. ⛔ One rule written twice, and only one copy edited

The single most expensive pattern in this ecosystem. It has produced wrong
money columns, wrong reports and wrong schemas.

| Incident | Cost |
|---|---|
| `TOTAL CASH INFLOW` / `OUTFLOW` named neither `CREDIT`/`DEBIT PAID` nor `CREDIT PAID`/`DEBIT`, while `EXPECTED OPERATING CAPITAL` was chained on flows that did | On 61 real dates, 3 disagreed with the reconciliation printed beside them, by −220,000, −150,000 and +370,000 |
| The Monthly Commission breakdown named the normal and LIPA totals but not SUPERAGENT | Mobile parts summed to 1,493,174 against a printed total of 1,793,174 |
| Two Selcom family totals duplicated provider grand totals | **Eight sites** across two engines, two of them in the same file |
| Alphabetical months in the Monthly Commission report | Fixed in CLOUD, not in ABS — the same divergence running the other way |

**What to do:** before editing a rule, `grep` the *other* engine and the *other*
file for it. A fix in one engine is not finished until the other has been read.
See [04 — One rule, two engines](04_one_rule_two_engines.md).

⚠️ **Deliberate duplication is allowed, but must be guarded by a test.**
`run_app.py` keeps its own copy of the failure matcher because its guard runs
before `src` is importable — a failure to load *that* module would be the one
bug with no way to report itself. `TheTwoCopiesOfTheMatcherAgree` compares
every keyword in either table, not a sample corpus: a corpus proves the cases
somebody thought of, the tables are what actually drift.

## 2. ⛔ A guard keyed to a value where a structural property was meant

It holds only while the two happen to coincide, then fails silently.

The Daily Snapshot blanked its AVERAGE, MAXIMUM and MINIMUM rows for every
reconciliation column. The renderer suppressed individual counter rows on
multi-submission dates with `trans_date not in _SUMMARY_LABELS` — asking
whether a **date cell** happened to hold a summary label. On every other report
the first column *is* the date, so a summary row held `"COLUMN TOTALS"` there
and the guard caught it. The Daily Snapshot leads with `Reporting Date`, so its
four summary rows held `''` in the date column — four identical values that
`duplicated()` read as one date submitted four times.

In the **initial commit**. Found by reading a report, not by any test. The
guard now identifies the **row** (`fdf[fdf.columns[0]].isin(_SUMMARY_LABELS)`).

**Ask:** is this condition true because of what the thing *is*, or because of
what it happens to *contain* today?

## 3. ⛔ A check that cannot fail proves nothing — and one that cannot succeed proves nothing either

| Check | Why it was worthless |
|---|---|
| `strings run_app.bin \| grep 1.0.0` on a `--onefile` binary | The payload is compressed; **nothing** is visible. A zero result is meaningless — the control string was absent too |
| `grep -a $'⚙' run_app.exe` | **False positive.** Matched three bytes inside an x86 `call` instruction, `e8 e2 9a 99 05`. A 119 MB binary contains any 3-byte sequence by chance |
| `strings -a run_app.exe \| grep ⚙` | **False negative on everything.** `strings` emits runs of printable ASCII, so a multi-byte character terminates the run and can never appear |
| `ldd lib/*.so` without `LD_LIBRARY_PATH` pointing at the bundle | Resolves against the host. Cannot fail here, cannot succeed on a customer machine |
| Four of five excepthook tests matching source strings | Commenting a guard out leaves `# sys.excepthook = …`, which still *contains* the substring. They passed against code with the guards removed |
| An ERP test loading a module copy from a dead scratchpad path | `skipTest`-ed on every run, for months |

**Always run the control first.** If the positive case does not light up, the
negative case tells you nothing.

## 4. ⚠️ Reasoning from what a thing is *named* rather than what the system does with it

`max-width` sounds like "how wide this may become". On a table cell it is a
**downward clamp only**. The Details columns were widened by raising
`max-width` 600px → 1200px (650 → 1300 in CLOUD); nothing moved, in either
engine, because the auto table layout already wanted only ~392px. Measured
afterwards in a browser:

| Declaration | Rendered |
|---|---|
| `width: 700px` | ignored outright — 392px |
| `max-width: 200px` | 229px — clamps downwards |
| `max-width: 5000px` | 392px |
| `min-width: 800px` | **829px** |

The reference text written that day asserted the floor was "the half not asked
for". It was the only half that did anything. **One measurement, before
shipping, would have caught it.**

Same shape: the Nuitka flag list in the reference carried
`--include-package=dropbox`, which the build does not pass — naming the package
explicitly pulls in every module `dropbox/base.py` imports and the kernel kills
the compiler.

## 5. ⚠️ Three lists, and only two of them generated

`get_dynamic_columns()` declared **twelve** free-text Details columns; the entry
form and the edit hub each named a hand-typed **five**. Six columns were created
in the database, carried on every row and printed in every report — and
**nothing in the application could put a value in any of them.**

`git log -S` puts both the twelve and the five in the **initial commit**. It
survived because a blank column reads as *"nothing entered"*, not as *"nothing
can be entered"*. It took an operator trying to type into one.

**A column reaching the schema and the reports does not mean a screen can write
to it.** Anything added to a dynamic column group needs the question asked
explicitly: *which screen fills this in?*

## 6. ⛔ A check a human is trusted to repeat is a check that will be skipped on the day it matters

The safety scan of the distribution passed at 13:30. Packaging happened at
14:50 without re-running it, and **nine customer licences went into the
tarball**. Caught in the `tar` listing; nothing was uploaded.

There was no committed packaging step at all, so every repack was done by hand
and the scan was optional again each time.

The repair was `scripts/package-release.sh`, which refuses to archive a folder
it has not *just* scanned and then **re-reads the archive it actually wrote**.
And then the same pattern bit the repair twice:

- The hashing step was left **outside** that run, as a separate `sums`
  subcommand. A rebuilt archive landed beside the previous run's
  `SHA256SUMS.txt`, 3,978 bytes apart, so `sha256sum -c` — exactly what both
  published guides tell the customer to run — answered **FAILED on a sound
  download**.
- The Windows path searched for a directory *named* `abs_dist_windows`. CI
  uploads a directory's *contents*, so the artifact root **is** the dist. That
  path **died on the first real artifact it was ever pointed at**, because only
  the Linux path and the two refusals had ever been exercised.

**Automate the check into the same run as the thing it checks** — and then run
every branch of the automation at least once.

## 7. ⛔ Silent success is the dangerous failure

A failure that prints is cheap. A failure that reports success is what reaches
customers.

| What reported success | What was actually true |
|---|---|
| `export_plaintext_db()` | Called `iterdump()` on a SQLCipher connection, which has no such method; the `AttributeError` went into a bare `except: return None`, read as "nothing to export". **Every remote backup for months was restorable only on the machine that made it** — the one route documented as surviving a lost machine |
| Restoring a backup | Merged the audit trail into itself: *"32 entries merged, 0 already present"* — doubled, reported as success |
| A restore writing files | Never opened what it wrote. A foreign archive restored "successfully" and surfaced one restart later as corruption, with the original gone |
| `_dropbox_client()` | Constructs lazily, so a revoked token returns a client that looks fine and fails later inside `except Exception: return False`. **A revoked token, no internet, and an empty folder all render identically** |
| A packaged Windows build | Packaged and uploaded cleanly with six transitive GTK DLLs missing; `dlopen` failed at the customer and **every PDF silently fell back to HTML** |

**Make the failing path loud, and prefer blocking to proceeding** where false
confidence is the worse outcome: a failed portable export now blocks the
upload.

## 8. ⚠️ The optimistic path is tested; the failure path ships untested

CI built the Windows package for months and **never once ran it**.
`--check-licence` returns before `app_main`, so it never touches pandas or
WeasyPrint — a package with every extension module missing would have passed
every check. `run_app.py --selftest` now forces the whole import chain, and CI
runs it **before** the upload, because a check after the upload proves nothing
about the artifact anyone downloads.

## 9. ⚠️ An estimate stated as a measurement

- **The glibc floor is a property of the build host, not of the code.** Do not
  trust that the container ran; read it back off the binary with `objdump -T`.
- **Character width is not character count.** At 18px, prose measures 8.0px per
  character, digits 10.0, capitals 12.4 — and real free-text entries here are
  digit-heavy. A flat 10px guess was replaced by the advance-width table of the
  bundled font, which agrees with the browser to 1px over a 150-character
  string.
- **Glyph width cannot be assumed.** `✔` and `✖` are East Asian Width
  *Neutral*, so a conforming terminal gives them one cell — and Windows
  Terminal paints them over the closing `]`. Both consoles were probed by hand
  before the glyph set was chosen.

## 10. ⚠️ A warning that cannot be substantiated is worse than none

`_is_foreign_archive()` answers "no" whenever it cannot tell. The blocked-startup
picker is the only screen a machine that will not start ever shows, and hiding
the one archive that could recover it — because the licence happened to be
unreadable — would turn a recoverable machine into a support call.

**Certainty is required to exclude, never to include.** The loss warning before
a restore is silent when either side is unreadable, for the same reason: a
warning the operator learns to type past is worse than no warning at all.

## 11. ⚠️ Fixing the screen is not fixing the application

Guarding the import block that failed left two worse paths open, both found a
day later by audit rather than by incident: `--check-licence` imports
`cryptography` and so tracebacked **in the middle of an installation**, and the
Dropbox upload's import sat above its `try` in a daemon thread, printing a
traceback **over whatever the operator was reading**.

`sys.excepthook` and `threading.excepthook` are the net under both — and under
code added later that forgets to guard itself.

## 12. ⚠️ Someone else's library prints to your screen

WeasyPrint's `_dlopen()` prints **eight lines to stdout** — including two links
to its own installation docs — and only *then* raises. So an agency operator
generating a PDF was told to read the WeasyPrint documentation. No `except`
takes that back.

Two layers are needed, and each catches a different writer: **file-descriptor
redirection** for anything written from C (SQLCipher's failed-key log goes
straight to fd 2, uncatchable by Python), and **replacing the stream objects**
for an ordinary `print`. The first version did neither properly and leaked only
when stdout was a pipe — passing every check made by hand, failing in a
customer's log.

## 13. ⚠️ Machine-local state, and reports of state

- Context in a home-directory notes folder is lost with the laptop
  ([01](01_where_context_lives.md)).
- **A stale downloaded artifact has been mistaken for a fresh one.** Check the
  artifact is the build you think it is.
- **A second staging folder with the same filenames.** `Release_packages/` held
  the October build under names identical to the live ones — different hashes,
  different sizes, gitignored so it never appeared in `git status`. Uploading
  from the wrong folder would have shipped a seven-week-old binary under the
  same version with no error anywhere. Deleted 2026-11-22; **`releases/` is the
  only staging path.**
- **A report that an action happened is not the action.** A push reported as
  done had not reached the remote — twice in one session; `origin/main` was
  still three commits behind. Verify the end state (`git log origin/main..HEAD`),
  do not trust the report, including your own.
- **A backgrounded wrapper is not the job.** `nohup … &` nested inside a
  background task made the harness track the wrapper, which exited after 20
  seconds and reported "completed" while the build ran for another hour.

## 14. ⛔ "Nothing" has more than one spelling, and they are not interchangeable

A field can be **empty**, hold **whitespace**, be **NULL/NaN**, or genuinely be
**zero**. Collapsing them is how a record starts lying.

The audit trail recorded `0.00` as the previous value of a note nobody had
written. The normaliser got it right — `return 0.0 if numeric else ""` — and
the display line one step later threw the distinction away, because the `else`
of

```python
f"{float(n_old):,.2f}" if is_numeric else (str(n_old) if n_old != "" else "0.00")
```

is the **non-numeric** branch. So a blank `EXPENDITURES Details` was written
into a financial audit trail as `0.00`, in a column that elsewhere holds money.

**The shape is: the correct distinction is computed, then discarded one line
later by a default belonging to the other type.** Look for it wherever a
formatter follows a parser.

Two consequences worth copying:

- **Whitespace is its own case.** The normaliser strips, so a field *cleared to
  spaces* also arrives as `""` — a different event from one never filled in.
  Both are now recorded, as `(empty)` and `(whitespace)`.
- ⚠️ **Decide where the distinction is stored, not just displayed.** These are
  written into the record rather than applied at render time, because **HTML
  collapses whitespace** — a raw `"   "` reaches the reader as one space, so a
  render-time fix would be destroyed at exactly the moment someone reads the
  document.

## 15. ⚠️ A rule that selects by *name* misses content travelling under another name

The report widening tier matches `"Details" in col_name or col_name ==
"INCIDENTS"`. The audit trail's `Old Value` and `New Value` carry **whatever
field was edited** — the free-text ones included — and match neither. So the
columns most in need of room were the only ones with no rule at all: a
151-character value wrapped to **5 lines** in 312px while `S/N` held one digit
in 152px.

**Ask what a column *carries*, not what it is called.** The same question
caught six Details columns that existed in the schema and the reports but on no
screen that could fill them in ([trap 5](02_traps.md)).

⚠️ And when you fix it, check the fix can actually take effect: the first
attempt set `width: 1%`, which **cannot beat a `min-width`** — every cell
carried `min-width: 130px` from the base rule, so nothing moved. Same family as
[trap 4](02_traps.md): a property that sounds decisive and is not.

## 16. ⛔ Some things cannot be recalled

A GitHub release asset is served at `releases/latest/download/<FILENAME>`.
`latest` resolves the *release*, but **the filename is matched exactly** — so
bumping the version 404s every link ever published: the site, both guides, the
forum post, and the **flyer PDF already on customers' phones**, which can never
be corrected.

So assets are replaced **in place** under the same name, and the price is paid
in the download counter, which is per asset and **resets on replacement**. The
figures are written down by hand before every repack, because nothing can read
them back afterwards.

See [05 — Shipping](05_shipping.md).

## 17. ⛔ A filter added to a reader redefines what the reader reads

One ERP installation serves three separate bodies — a workers' voluntary
association, a church, and TSSFL — and `view_audit_log()` gained a module scope
so that each body's Admin stopped reading the other two's entries. Underneath
it, three readers each took a **fixed** window and filtered afterwards: the
last 50 lines, the last 200 KB, the whole file.

So *"the last 50 entries"* quietly became *"whichever of the last 50 entries
are mine"*. A body with one entry in the file's last fifty lines would open its
own audit log and be shown **one line** — its history present in the file,
absent from the screen, and nothing on the screen saying so. The window now
**grows** (50 → 200 → 800 lines) until it holds N visible lines or reaches the
file cap.

> `N` was defined on the unfiltered sequence, and the filter moved the sequence
> out from under it. Ask this of every tail, paginator, "recent" list and
> `LIMIT` that later acquires a `WHERE`.

## 18. ⚠️ Hiding a row and admitting it exists are two different disclosures

The same scoped view withholds two kinds of line and deliberately treats them
differently:

| Line | Hidden | Counted |
|---|---|---|
| No module tag — logins, backups, user changes, and everything written before tagging existed | yes | **yes** — *"3 earlier entries are not shown"* |
| Another body's entry | yes | **no** |

Untagged history is the installation's own, and a reader who meets a gap with
no explanation reads it as a broken log. But a count of **another body's**
lines would tell the church's administrator how many entries the workers'
association wrote this week — which is precisely what the scope exists to
withhold.

> Volume is information. Decide separately whether a reader may see a row and
> whether they may learn that it exists.
