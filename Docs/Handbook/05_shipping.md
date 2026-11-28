# 05 — Shipping

Everything here is about artifacts that reach someone else's machine, where a
mistake cannot be edited afterwards.

---

## ⛔ The version does not change, and neither do the filenames

A GitHub release asset is served at `releases/latest/download/<FILENAME>`.
`latest` resolves the *release* — but **the filename is matched exactly**.
Bumping the version returns a bare **404** on every link ever published: the
site, both guides, the forum post, and the flyer PDF already on customers'
phones. GitHub does not fall forward to the newer file.

So: assets are **replaced in place** under the same name.

The price is the **download counter**, which is per asset and resets when that
asset is replaced. It is the only readership data there is, so the figures are
written into `Docs/context/project_release_download_counts.md` **before** every
repack. Nothing can read them back afterwards.

`scripts/package-release.sh` reads the version from `src/version.py` and never
changes it, so the filenames stay put by construction.

**The indirection that makes a future bump survivable** is `site/download/`:
`windows.html`, `linux.html`, `checksums.html`, each a `<meta http-equiv=
"refresh">` to the current asset, served at `/download/windows` and friends. A
future bump edits one line in each instead of every document ever sent. No
JavaScript, so it works where scripting is blocked.

⚠️ The flyer deliberately still links straight to GitHub and **names the
version on the button** (`v1.0.0 · 64-bit · 85 MB`), because repointing it
would protect only copies sent *after* the change while the ones already
circulating kept the direct links — two forms of the same sheet disagreeing.
**Anything published from now on uses `/download/…`.**

## ⛔ Package and verify in one run

See [trap 6](02_traps.md). The packaging script must:

1. **Scan** the tree — no `.lic`, no keys, no credentials, no databases, no
   developer tools, and a shipped config that is blank with remote backup off.
2. **Refuse to archive** a folder it has not just scanned.
3. **Re-read the archive it actually wrote** — the step that caught nine
   licences.
4. **Write the folder's `SHA256SUMS.txt` in the same step as the archive**, and
   drop the combined file, which has just gone stale.

A new archive beside a previous run's checksum reports a **sound download as
FAILED** — worse than no checksum at all, because a mismatch is the one signal
that means *tampered or truncated*.

## ⚠️ Ask the artifact, not the build log

| Check | Why it is not left to a person |
|---|---|
| glibc floor via `objdump -T` | A property of the build host, not the code. A build on a modern machine demands 2.34 and will not start on Ubuntu 20.04 |
| `lib/*.so` resolved **within `lib/`** | Resolving against the build host cannot fail there and cannot succeed on a customer machine |
| `--selftest` | `--check-licence` returns before `app_main`; a package with no extension modules passes everything else |
| **CRLF restored** on `install.bat` / `uninstall.bat` | Copied by CI, never compiled; the runner converts on checkout. Repacking on Linux is the one route by which `cmd.exe` gets a file it mishandles |
| DLL closure recomputed | A hand-written list cannot see a *transitive* dependency. Six were missing from a shipped build; every PDF silently fell back to HTML |
| `fonts.conf` in all three locations | On Windows fontconfig finds its config relative to **its own DLL**, and `FONTCONFIG_FILE` cannot reach it across two C runtimes |

**The DLL count moves with upstream — 34 in September 2026, 48 now.** A number
that changes is the point; a fixed one would mean the list had been written
down again.

## ⚠️ Identify the input by what it *is*

CI uploads a directory's **contents**, so the artifact root *is* the
distribution, and `gh run download` names the folder after the artifact. Find
the dist by locating `run_app.dist/run_app.exe`, not by looking for a directory
with a particular name.

## ⚠️ Artifacts expire

CI artifacts are deleted after **14 days**, deliberately: a Windows artifact is
~90 MB against 0.5 GB of included storage, and twelve of them once reached
1.02 GB and a quota warning. A build is downloaded and installed within hours,
so two weeks is generous — but **fetch the package before it ages out**, or
re-run CI at ~1h45m of billed minutes.

## The two release procedures

There are exactly two, and they are not interchangeable. **`releases/` is the
only staging path** — a second folder of packages with the same filenames is a
loaded gun (one existed here for seven weeks, seven days' divergence apart, and
was deleted on 2026-11-22).

### A — Replacing the assets of an existing version (the usual case)

The filenames do not change, so every published link survives. This is what you
do for a bug fix, a rebuild, or anything that is not a new product version.

```bash
# 1. Build from a committed, clean tree
bash docker/build-linux.sh                       # Linux, in the focal container
gh workflow run <id> --ref main                  # Windows, ~1h45m
gh run download <run-id> --dir /tmp/win          # when it finishes

# 2. Record the download counts BEFORE anything is replaced -- they reset
gh api repos/<owner>/<releases-repo>/releases \
   -q '.[] | .tag_name as $t | .assets[] | [$t,.name,(.download_count|tostring)] | @tsv'
#    write them into Docs/context/project_release_download_counts.md

# 3. Package -- each run scans, archives, re-reads and writes its own checksum
bash scripts/package-release.sh linux
bash scripts/package-release.sh windows /tmp/win/<artifact-name>
bash scripts/package-release.sh sums              # the combined file

# 4. Replace in place: same tag, same filenames
gh release upload v1.0.0 \
   releases/linux/NEXTGAMIS_ABS_1.0.0_Linux_x64.tar.gz \
   releases/windows/NEXTGAMIS_ABS_1.0.0_Windows_x64.zip \
   releases/SHA256SUMS.txt \
   --repo <owner>/<releases-repo> --clobber

# 5. Verify as a customer: download from the PUBLIC url, not a local copy
curl -sSL -O https://github.com/<owner>/<releases-repo>/releases/latest/download/<file>
sha256sum -c SHA256SUMS.txt --ignore-missing
```

⚠️ **`--clobber` is what replaces rather than fails.** Without it the upload is
rejected because the asset name already exists.

⚠️ **Check `releases/latest` still resolves to the tag you uploaded to** —
`gh api repos/<owner>/<repo>/releases/latest -q .tag_name`. Every published
link goes through `latest`, so a draft or pre-release flag silently breaks all
of them.

⚠️ **The source repo and the releases repo are different repositories here.**
`gh release list` run in the source checkout shows nothing, which reads exactly
like "the release was deleted". Pass `--repo`.

### B — Publishing a genuinely new version

Only when the product version changes. **It breaks every direct link ever
published**, including PDFs already on customers' phones, so the indirection
pages must be updated in the same sitting.

```bash
# 1. Bump src/version.py -- the packaging script reads the version from there
#    and never changes it, so the filenames follow automatically
# 2. Build, package and verify exactly as in A
# 3. Create the release on a new tag
gh release create v1.1.0 --repo <owner>/<releases-repo> \
   --title "NEXTGAMIS ABS 1.1.0" --notes-file <notes> \
   releases/linux/*.tar.gz releases/windows/*.zip releases/SHA256SUMS.txt
# 4. Repoint the indirection pages, which is what keeps older documents alive
#    Docs/release/site/download/{linux,windows,checksums}.html -- one line each
bash Docs/release/publish.sh            # shows what would change
bash Docs/release/publish.sh --push     # copies, commits, pushes
# 5. Start a fresh counts table for the new version
```

⛔ **Do not bump a version to avoid the counter reset.** The counter is cheap;
a dead link in a printed flyer cannot be recalled. See
[trap 14](02_traps.md).

## ⚠️ An archive is not byte-reproducible, but its contents are

Two packs of the same tree can differ, because **the directory's own mtime is
recorded in the tar header** — touching anything in the folder, even moving a
file out and back, changes it. The gzip header timestamp is zeroed, so that is
not the cause.

Measured on 2026-11-22: moving a licence out of the dist and back produced
`1ec2a301…` where the published asset was `a69f75fa…`, while
`tar tzvf | awk '{print $1,$3,$6}'` over both was **identical across all 43
entries** — same names, same modes, same sizes.

**So a differing hash does not mean a differing build.** Compare the listings
before concluding anything. And after any such detour, **restore the local copy
from the published artifact** rather than re-uploading a needlessly different
file: the published bytes are what the customer's `SHA256SUMS.txt` names.

## ⛔ Nothing but the distribution goes in the distribution folder

`tar czf "$OUT" "$SRC"` archives the folder whole. A package left in
`abs_dist_linux/` — to install from, say — would be **nested inside the next
release**, roughly doubling it, with nothing reporting an error. `scan_tree()`
refuses any `.tar`, `.tar.gz`, `.tgz`, `.zip` or `.7z` found in there.

Extract a downloaded package *as* the distribution folder; keep the archive in
`releases/`.

## Before you publish

- [ ] Tests green, and the new ones seen to fail against the old code
- [ ] Built from a **committed, clean** tree; note the commit
- [ ] Packaged through the script, every gate passed
- [ ] `sha256sum -c` answers `OK` on the file you are about to upload
- [ ] Download counts recorded **before** replacing anything
- [ ] Filenames unchanged; `releases/latest` still resolves to the release
- [ ] Downloaded back from the public URL and verified, as a customer would
- [ ] Manuals and guides regenerated if operator-visible behaviour moved —
      edit `generate_manual.py`, never its output

## ⛔ What never ships

`private_key.pem`, `public_key.pem`, any `.lic`, the licence registry,
`src/credentials.py` with real values, `config/gas_credentials.json`, any
`*.db`, `.last_seen`, `.config_sig`, `keygen.py`, `generate_license.py`,
`license_mailer.py`.

The scan enforces this; the scan is not a substitute for not putting them there.
