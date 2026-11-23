# 03 — Proving it works

Read this before writing the words *verified*, *fixed*, *confirmed* or *works*.

---

## ⛔ Break it and watch it fail

A guard nobody has seen refuse is a guard nobody knows works. Every check worth
having has been run against **deliberately broken code** and seen to go red.

| Guard | How it was proved |
|---|---|
| The packaging safety scan | A `.lic` file was planted in the tree; a config with a company name was shipped. Both refused, exit 1 |
| The Details-width rule | The helper was made to ignore its data — four of eight cases went red; the two-line rule was restored — two went red |
| The baked font table | One advance width was changed from 1139 to 1140; the font comparison failed naming the character |
| `RestoringOnTheSameMachine…` | Run against the old code, it fails with exactly the message the incident reported |
| The snapshot blanking rule | A **control test** was added precisely because a "fix" that disabled the rule entirely would pass the first test and break every multi-submission report |

If you cannot make a test fail, you have not written a test. You have written a
sentence that happens to execute.

## ⚠️ Measure; do not reason from the name

The `max-width` incident ([trap 4](02_traps.md)) cost a release cycle and a
wrong paragraph in the maintained reference, and a single browser measurement
settled it in under a minute. Where something renders, runs, or links, **ask
the thing itself**:

```bash
objdump -T run_app.bin | grep -oP 'GLIBC_\K[0-9.]+' | sort -V | tail -1   # the floor
LD_LIBRARY_PATH=lib ldd lib/libfoo.so                                      # the bundle
run_app --selftest                                                         # the import chain
sha256sum -c SHA256SUMS.txt --ignore-missing                               # the download
```

## ⚠️ Run the control first

A search that returns zero proves nothing until you have seen the same search
return non-zero for something you know is there.

```bash
grep -c "Finalizing data"   /tmp/exe.txt     # CONTROL -- must be non-zero
grep -c "sqlcipher_export"  /tmp/exe.txt     # the real question
```

Under `--onefile` the control fails, which is the answer: **this method cannot
work here.** Find another one. For the glyph question the methods that actually
settle it are the source (`grep -rn` plus a test that runs before the build), a
contiguous ASCII run across the place the glyph would sit, and a photograph of
the running application.

## What counts as evidence, in order

1. **A test that fails against the old code.** The strongest, because it
   re-runs forever.
2. **A measurement of the real artifact** — the binary, the rendered page, the
   downloaded file, the live data.
3. **A before/after on real data, with a control.** The inflow/outflow fix was
   confirmed on a customer file of 61 dates: 3 disagreed before, 0 after, and a
   control proved the old engine disagreed with itself on exactly those 3.
4. **A photograph of the running application.** Sometimes the only thing that
   can settle a rendering question.
5. **Reasoning from the source.** Necessary, never sufficient.

## ⚠️ Prove the *identity* of what you tested

- Check the artifact is the build you think it is. A stale download has been
  mistaken for a fresh one.
- Check the tree was clean when it built, or you have tested something that
  exists nowhere.
- Check the remote actually moved (`git log origin/main..HEAD`) before saying
  it is pushed.

## ⚠️ Pin the rule, not the arithmetic

A test asserting `800px` for a 150-character value would break the next time
the body font moved, and would be measuring arithmetic rather than the rule.
Pin instead: an empty column keeps the base; an entry inside the cap gets room
for one line; nothing exceeds twice the base.

Likewise, a test that pinned the literal string `"6": ("All Commissions"` broke
on a renumbering while the behaviour it guarded was untouched. **A test tied to
a spelling rather than an outcome will cost you more than it saves.**

## ⚠️ Read the parse tree, not the source text

Four of five tests guarding `sys.excepthook` passed against code with the
guards commented out, because `# sys.excepthook = …` still contains the
substring. They now assert an actual `Assign` node.

## ⚠️ Make the code testable rather than testing its text

When a rule lives in a nested closure, the temptation is to assert on the
source. Lift it to module level instead and call it. `audit_value()` was moved
out of `handle_edit()` for exactly this, and the test then exercises eight real
inputs — empty, whitespace, `None`, `NaN`, text, and two genuine numbers —
instead of grepping for a string.

⚠️ **Check the lift landed where you think.** The first attempt left the
function nested inside a neighbouring one at the same indentation, where every
call site would have raised `NameError`. The parse tree said so
(`[n.name for n in ast.parse(src).body]`); reading the diff did not.

## Report faithfully

- If tests fail, say so and show the output.
- If a step was skipped, say which and why.
- If something is done and verified, say it plainly — no hedging.
- If you reported something wrong earlier, correct it in one sentence and move
  on. This project's own history contains a reversed explanation corrected
  hours later **by evidence** (the Dropbox token), and the correction is more
  valuable than the original note.
