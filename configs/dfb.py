# configs/dfb.py — Digital Financial Bridge
#
# Public by necessity: SageCell fetches this anonymously from GitHub raw, so it
# must contain NO secrets. The Dropbox file id and share key come from the post.
#
# Define CO (header name) and run_reports(report). Everything else is optional.
#
# Report numbers, all accepting period='all'|'year'|'range':
#   1 Comprehensive Extended   2 Full Clean          3 Monthly Commission
#   4 Mini Totals Dashboard    5 Compact Accounts
# Single-date: daily_snapshot_report(date), quick_report(date, fmt='html'|'pdf')
# Dates are DD/MM/YYYY.

CO = "DIGITAL FINANCIAL BRIDGE"


def run_reports(report):
    report.abs_report(1, period="all", company_name=CO)
    report.abs_report(3, period="all", company_name=CO)
    report.quick_report(fmt="html", company_name=CO)
