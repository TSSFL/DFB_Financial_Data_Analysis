Financial Data Analysis Python Class for Agency Banking in East Africa

This Python class is designed to support and monitor agency banking, commonly known as UWAKALA, with a focus on East Africa. The class allows for easy access to data through callable endpoints with credentials provided during instance creation. Data can be accessed from various sources including local drives, Google Drive, Dropbox, or KoboToolbox. Data submission can be done using a mobile app, Google Forms, or other convenient methods.

Usage

Create an Instance

Depending on where the data is sourced from, create an instance using one of the following options:

Google Drive:

google_drive_report = FinancialReport(data_source='google_drive', spreadsheet_key=" ", json_file=" ")

Local Drive:

local_drive_report = FinancialReport(data_source='local_drive', file_path=' ')

Dropbox:

dropbox_report = FinancialReport(data_source='dropbox', file_path=' ', file_name='XXXX_Data.csv')

KoboToolbox:

kobo_report = FinancialReport("kobo", token=" ", url=" ", asset_index=0)

Create Reports

Generate reports using the following methods

Full or Brief Major Reports:

report._full_report('brief') # Options: 'full', 'brief'

Specific Month for All Years Report:

report.specific_month_for_all_years_report(1, 'full') # Options: month - 1,2,3,... , 'full', 'brief'

Specific Month of Year Report:

report.specific_month_of_year_report(4, 2024, 'brief') #Provide target month and year

Weekdays Report:

report.weekdays_report('brief')# 'full', 'brief'

Weekdays of Target Month and Year Report:

report.weekdays_of_target_month_and_year_report(5, 2024, 'full') #month, 'full', 'brief'; Provide the target month and year

Graphs and Charts:

report.graphs('04/05/2024', 'full') #Options: date, top, low, full, default date: None
