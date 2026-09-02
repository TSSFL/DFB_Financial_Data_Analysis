# NEXTGAMIS Cloud — Agency Banking Reporting

Financial reporting for multi-provider agency banking (*UWAKALA*) in East Africa,
as a single Python class that runs anywhere — a notebook, a SageCell session, or a
plain script — without an install step.

Transaction data is collected through Google Forms, KoboToolbox, or a mobile app,
lands in Google Sheets, Dropbox, or a local file, and comes back as formatted HTML
or PDF reports: reconciliation, commission, daily snapshots, and a one-page daily
summary.

The class is the cloud counterpart of **NEXTGAMIS ABS**, the offline-first agency
banking terminal. Reports 1–7 here are column-for-column identical to the ones ABS
produces on a terminal, so a branch running the terminal and a branch submitting
through a web form reconcile against the same figures.

---

## Quick start

The class is served directly from this repository — there is nothing to install
and nothing to clone.

**SageCell**

```python
load("https://raw.githubusercontent.com/TSSFL/DFB_Financial_Data_Analysis/master/nextgamis_cloud.py")

report = FinancialReport(data_source='dropbox',
                         file_path='<your Dropbox direct-download link>',
                         file_name='data.csv')
report.abs_report(1)                       # Comprehensive Report, all time
report.quick_report()                      # today's one-page summary
```

**Notebook or script**

```python
import importlib.util, urllib.request

url = ("https://raw.githubusercontent.com/TSSFL/"
       "DFB_Financial_Data_Analysis/master/nextgamis_cloud.py")
urllib.request.urlretrieve(url, "nextgamis_cloud.py")

spec = importlib.util.spec_from_file_location("ngc", "nextgamis_cloud.py")
ngc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ngc)
FinancialReport = ngc.FinancialReport
```

Reports are written to the working directory. Each call returns the filename it
wrote, or `None` when the requested period holds no data.

> **Note on `load()` in SageCell:** re-running `load()` rebinds the class name but
> leaves any instance you already created on the old class. After updating the
> module, rebuild the instance as well.

---

## Data sources

Pick one at construction time. The frame is read once and reused by every report.

```python
# Dropbox — use a direct-download link (dl=1)
FinancialReport(data_source='dropbox', file_path=url, file_name='data.csv')

# Local file
FinancialReport(data_source='local_drive', file_path='data.csv')

# Google Sheets — service account credentials
FinancialReport(data_source='google_drive',
                spreadsheet_id='<sheet id>',
                service_account_file='<path or URL to the JSON key>',
                range_name='<optional A1 range>')

# KoboToolbox — choose the survey by name or uid
FinancialReport('kobo', token='<api token>', asset='My Survey Name')
FinancialReport('kobo', token='<api token>', asset='aN5e7mf4CY6uKq2NrckxtY')
FinancialReport('kobo', token='<api token>', asset_index=0)   # by position
```

`url=` defaults to `https://kf.kobotoolbox.org/api/v2`; pass it for a different
server. `kobo_debug=True` turns on the extractor's request trace.

Every submission is fetched, following the server's pagination, and the row count
is checked against the count KoboToolbox reports for the asset — a short download
raises rather than reporting on part of a survey. Prefer `asset=` over
`asset_index=`: the index is a position in the server's own listing, so it is not
guaranteed to name the same survey twice.

#### Naming an ABS collection form

Kobo submissions arrive keyed by the form's XML question names, not by the labels
the reports expect, so the names are mapped on the way in. Write the questions in
snake_case and the mapping is automatic:

| Kobo question | Becomes |
|---|---|
| `group_txn/date_of_transaction` | `Date of Transaction` |
| `group_float/m_pesa` | `M-PESA` |
| `halo_pesa_superagent_2_comm` | `HALO-PESA SUPERAGENT 2 COMM` |
| `selcom_1_details` | `SELCOM 1 Details` |
| `_submission_time` | `Timestamp` (UTC → EAT) |
| `_submitted_by` | `Name of Submitter` |
| `_id`, `_uuid`, `meta/…`, `formhub/…` | dropped |

Group paths are stripped, and separators and case are ignored when matching, so
`m_pesa` recovers the roster's `M-PESA` rather than becoming `M PESA`.

A question the roster does not know is uppercased and **reported**, because its
original casing cannot be recovered from a Kobo name — `mobile_bundles_and_shares`
cannot say that `and` was lowercase. Settle those with `KOBO_FIELD_MAP`:

```python
FinancialReport.KOBO_FIELD_MAP = {
    'group_txn/txn_date': 'Date of Transaction',
    'group_float/mb_shares_details': 'MOBILE BUNDLES and SHARES Details',
}
```

An entry there wins over every other rule. If `Date of Transaction` is missing
after mapping, the load says so by name — without it every report is empty for
what otherwise looks like a healthy download.

`prepare=True` builds the legacy prepared frame eagerly at construction. By default
it is built on first use, so a run that only touches reports 1–7 never pays for it.

---

## Reports

### NEXTGAMIS ABS parity reports

Reports 1–5 accept any of three periods. Reports 6 and 7 cover a single date.

| # | Report | Call |
|---|--------|------|
| 1 | Comprehensive — every column, full reconciliation | `abs_report(1)` |
| 2 | Clean Full — comprehensive with the per-provider subtotals dropped, group totals kept | `abs_report(2)` |
| 3 | Monthly Commission — commission summed by calendar month, earliest first | `abs_report(3)` |
| 4 | Mini Totals Dashboard — float, hard cash, and the operating-capital position | `abs_report(4)` |
| 5 | Compact Accounts — every account column, no totals | `abs_report(5)` |
| 6 | Daily Snapshot — every submission for one date, plus the COMBINED row | `daily_snapshot_report(date=…)` |
| 7 | Quick Report — one page: `# · Description · Amount (TZS) · Details` | `quick_report(date=…)` |

**Periods** — reports 1–5:

```python
report.abs_report(1)                                            # all time
report.abs_report(3, period='year', year=2025)                  # one year
report.abs_report(4, period='range',
                  start='01/11/2024', end='30/11/2024')         # a date range
```

**Single-date reports** — reports 6 and 7 default to the most recent date that
holds data. An explicitly requested date is never substituted: you get that day or
nothing.

```python
report.daily_snapshot_report(date='01/11/2024')
report.quick_report(date='01/11/2024', fmt='pdf')   # A4 portrait via WeasyPrint
```

The Quick Report lays out three zones: non-zero numeric rows, non-empty free text,
and the balance rows, which are always shown even at zero. `fmt='pdf'` falls back
to HTML if WeasyPrint is unavailable.

Every report takes an optional `company_name=` — it heads the document and prefixes
the filename — and an optional `output_file=` to override the generated name.

### Legacy reports

The original reporting methods are carried unchanged. `report_type` is `'full'` or
`'brief'`.

```python
report.comm_report()
report.specific_month_for_all_years_report(1, 'full')
report.specific_month_of_year_report(4, 2024, 'brief')
report.weekdays_report('brief')
report.weekdays_of_target_month_and_year_report(5, 2024, 'full')
report.date_range_report('01/05/2024', '31/05/2024', 'full')
report.graphs('04/05/2024', 'full')          # 'full' | 'top' | 'low'; date may be None
```

---

## Output

Filenames follow a fixed shape, dashes between fields and underscores inside them:

```
<Company>-<Report>[_<Scope>]-DD_MM_YYYY-HH_MM_SS.html
```

```
DIGITAL_FINANCIAL_BRIDGE-Comprehensive_Report_All_Time-02_09_2026-11_00_15.html
DIGITAL_FINANCIAL_BRIDGE-Quick_Report-01_09_2026-11_00_15.html
```

The date field is the day the report **covers** when it covers exactly one day, and
the day it was **generated** otherwise. The clock is always the generation time, and
the document header carries the full stamp either way.

All generated timestamps are East Africa Time (UTC+3), fixed in the class rather
than read from the host clock, so a report generated on a UTC server carries the
same reading as one generated in Dar es Salaam.

---

## Column naming

Forms in the field emit provider columns in several shapes. `normalise_columns()`
rewrites them all to ABS naming before anything is computed, and the mapping is
idempotent — columns already in ABS form pass through untouched.

```
AIRTEL MONEY 1 SUPERAGENT COMM   ->  AIRTEL MONEY SUPERAGENT 1 COMM
AKIBA BANK                       ->  ACB BANK 1
SELCOM COMM                      ->  SELCOM 1 COMM
Timestamp                        ->  Date of Submission
```

The provider roster covers SELCOM, 10 mobile money providers, 45 banks, and 4
remittance agencies. A column outside the roster still has its totals computed; the
run reports it, so an unrostered provider is visible rather than silent.

Set `FinancialReport.ABS_VERBOSE = False` to silence progress output.

---

## Requirements

`pandas`, `numpy`, and `regex` are needed to import the module.

Everything else is imported by the method that needs it, not at module scope, so a
run that never reaches those methods never pays for them:

| Package | Needed by |
|---------|-----------|
| `pretty_html_table` | legacy reports, graphs |
| `matplotlib`, `seaborn` | `graphs()` |
| `weasyprint` | `quick_report(fmt='pdf')` |
| `gspread` | the `google_drive` data source |
| `koboextractor` | the `kobo` data source |

HTML reports work on a machine where WeasyPrint is not installed at all.

---

## Repository layout

| File | Purpose |
|------|---------|
| `nextgamis_cloud.py` | The class. Legacy reports plus the ABS parity layer. |
| `dfb_python_class.py` | The earlier standalone class, kept for callers still pinned to it. |

New work should target `nextgamis_cloud.py`.

---

## Related

**NEXTGAMIS ABS** — the offline-first agency banking terminal these reports match:
compiled cross-platform binary, encrypted local storage, offline licensing, and a
forensic audit trail. Reports 1–7 above are its reports, computed in the cloud.
ABS is proprietary and licensed separately per installation.

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE.md).

| | |
|---|---|
| **Free** | Research, teaching, study, personal and hobby projects; charities, educational institutions, public research bodies, and government institutions — regardless of funding source |
| **Not permitted** | Any commercial use |

For a commercial license, contact [sales@tssfl.co](mailto:sales@tssfl.co).

The module carries the required notice in its header, so the terms travel with the
file when it is loaded straight from a URL. NEXTGAMIS™ is a trademark and
servicemark of TSSFL / TSSFL Technology Stack; the license grants no rights in the
name, logo, or branding.

Copyright © 2024–2026 TSSFL / TSSFL Technology Stack. All rights reserved.

---

<sub>TSSFL Technology Stack — [www.tssfl.com](https://www.tssfl.com)</sub>
