#NEXTGAMIS Cloud - Agency Banking Reporting
#TSSFL Technology Stack - www.tssfl.com
#
#Single class, loadable directly from GitHub:
#   load("https://raw.githubusercontent.com/TSSFL/DFB_Financial_Data_Analysis/master/nextgamis_cloud.py")
#
#Carries the original report methods unchanged, plus the NEXTGAMIS ABS
#parity layer: reports 1-7 over All Time / a Year / a Date Range, ABS column
#naming with backward compatibility for the legacy form headers, ABS
#reconciliation maths, and ABS typography.

import urllib.request
import numpy as np
import pandas as pd

import regex as re

#gspread, seaborn, matplotlib, pretty_html_table, weasyprint and koboextractor are
#imported inside the handful of methods that use them, not here. Together they cost
#0.9s and 143MB at import time, and not one of them sits on the path the ABS parity
#reports take - so a SageCell run was paying for them on the way in, every time, for
#code most runs never reach. weasyprint in particular is only needed when
#quick_report is asked for a PDF; importing it there means the module now loads and
#every HTML report still works on a machine where weasyprint is not installed at all.
#(The unused `from weasyprint import CSS` is gone entirely.)

from datetime import date, datetime, timezone, timedelta

import os
import time
import warnings
warnings.filterwarnings('ignore')

class FinancialReport:
    def __init__(self, data_source, spreadsheet_id =None, service_account_file=None, range_name=None, file_path=None, file_name=None, token=None, url=None, asset_index=None, prepare=False):
        self.data_source = data_source
        self._df = None          #backs the lazy `df` property below
        if self.data_source == 'google_drive':
            self.spreadsheet_id = spreadsheet_id #Using spreadsheet id instead of key
            self.service_account_file = service_account_file
            self.range_name = range_name  #Add range_name
            self.data = self._get_data_from_google_drive()
        elif self.data_source == 'local_drive':
            self.file_path = file_path
            self.file_name = file_name
            self.data = self._get_data_from_local_drive()
        elif self.data_source == 'dropbox':
            self.file_url = file_path
            self.file_name = file_name
            self.data = self._get_data_from_dropbox()
        elif self.data_source == 'kobo':
            from koboextractor import KoboExtractor
            self.kobo = KoboExtractor(token, url, debug=True)
            self.asset_index = asset_index
            self.asset_uid = None
            self.df = None
            self.df_copy = None
            self.df = self._get_data_from_kobo()
    
        #Untouched source frame, used by the NEXTGAMIS ABS parity reports
        #Every source gets the same date normalisation. It used to live inside
        #_get_data_from_dropbox alone, so a local CSV, a Google Sheet or a Kobo asset
        #arrived carrying raw JavaScript date strings and every ABS report then said
        #"No transaction data" - same data, same code, silent empty output, decided
        #only by where the file came from.
        source = self.df if self.data_source == 'kobo' else getattr(self, 'data', None)
        if source is not None:
            self._normalise_source_dates(source)

        #Untouched source frame, used by the NEXTGAMIS ABS parity reports
        self.raw_df = source.copy()

        #The legacy frame is built on first use, not here - see the `df` property.
        #prepare=True forces it now, for a caller that would rather pay the cost up
        #front than on first access.
        if prepare:
            self.df

    @property
    def df(self):
        """The legacy prepared frame, built on first use.

        This used to be built eagerly in __init__, and prepare defaulted to True,
        so every instantiation paid ~2.6s for it. Anyone running only the ABS
        parity reports paid that in full for nothing: those read self.raw_df and
        never touch this frame. It was the single largest fixed cost in a run -
        larger than generating all seven reports.

        The cost itself is real work (consolidate_transactions groups by date and
        aggregates row by row, ~15M pandas calls on 546 rows), so it is deferred
        rather than removed: whoever actually needs the legacy frame still pays,
        once, on first access.
        """
        if self._df is None:
            self._df = self._full_report(render=False)
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    #The form emits JavaScript Date strings - 'Fri Nov 01 2024 00:00:00 GMT+0300
    #(East Africa Time)'. Everything downstream expects MM/DD/YYYY.
    _JS_DATE_TAIL = r'GMT[+-]\d{4}.*'
    _JS_DATE_FMT = '%a %b %d %Y %H:%M:%S'
    _DATE_COLS = (('Timestamp', '%m/%d/%Y %H:%M:%S'),
                  ('Date of Transaction', '%m/%d/%Y'))

    @classmethod
    def _normalise_source_dates(cls, df):
        """Put the date columns into the format the rest of the class expects.

        Called once from __init__ for every data source. It is deliberately
        forgiving, because it now runs on input that may never have carried the
        JavaScript form at all:

        - a column that holds no 'GMT' marker is left untouched, so a CSV that was
          already normalised passes straight through and the function is safe to
          run twice;
        - a row that fails to parse keeps its original text rather than becoming
          NaT. The old dropbox-only version parsed without errors='coerce', so one
          malformed row raised and the whole load died; a row that cannot be read
          is now simply left for the period filter to drop.
        """
        for col, out_fmt in cls._DATE_COLS:
            if col not in df.columns:
                continue
            text = df[col].astype(str)
            if not text.str.contains('GMT', na=False).any():
                continue
            stripped = text.str.replace(cls._JS_DATE_TAIL, '', regex=True).str.strip()
            parsed = pd.to_datetime(stripped, format=cls._JS_DATE_FMT, errors='coerce')
            df[col] = parsed.dt.strftime(out_fmt).fillna(text)
        return df

    def _get_data_from_google_drive(self):
        import gspread
        urllib.request.urlretrieve(self.service_account_file, "agency_banking.json")
        #Define the scope
        scope = ['https://www.googleapis.com/auth/spreadsheets'] 
            
        urllib.request.urlretrieve(self.service_account_file, "agency_banking.json")
        #gc = gspread.service_account(filename="agency_banking.json")
        gc = gspread.service_account("agency_banking.json", scopes=scope)
        sh = gc.open_by_key(self.spreadsheet_id)
        worksheet = sh.sheet1
        data = worksheet.get_all_records()
        filtered_data = [row for row in data if any(row.values())]
        return filtered_data

    def _get_data_from_local_drive(self):
        data = pd.read_csv(self.file_path)
        return data
        
    def _get_data_from_dropbox(self):
        self._p("Fetching data from Dropbox ...")
        url = self.file_url
        urllib.request.urlretrieve(url, self.file_name)
        df = pd.read_csv(self.file_name)
        #Date normalisation used to happen here, and only here. It is now applied to
        #every source from __init__ - see _normalise_source_dates.
        self._p("Data loaded: {} rows, {} columns".format(len(df), len(df.columns)), done=True)
        return df
        
    def _get_data_from_kobo(self):
        assets = self.kobo.list_assets()
        self.asset_uid = assets['results'][self.asset_index]['uid']
        asset = self.kobo.get_asset(self.asset_uid)
        choice_lists = self.kobo.get_choices(asset)
        questions = self.kobo.get_questions(asset=asset, unpack_multiples=True)

        new_data = self.kobo.get_data(self.asset_uid)
        new_results = self.kobo.sort_results_by_time(new_data['results'])
        self.df = pd.DataFrame(new_results) #Avoid self.df is referencing outside this methhod
        self.df_copy = self.df.copy() #Create a copy of the original dataframe
        return self.df
      
    def date_time(self, df):
        df['Date of Submission'] = pd.to_datetime(df['Date of Submission']).dt.strftime('%d/%m/%Y %H:%M:%S')
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction']).dt.strftime('%d/%m/%Y')
        
        return df
        
    def update_operating_capital(self, df):
        """
        Updates the 'ACTUAL OPERATING CAPITAL' column based on 'DEBIT',
        handling duplicate dates.
        Args:
            df: The Pandas DataFrame with a 'Date of Transaction' column.
        Returns:
            The updated DataFrame.
        """
        #Sort by 'Date of Transaction' to ensure correct processing of duplicates
        df.sort_values(by='Date of Transaction', inplace=True)

        #Group by 'Date of Transaction'
        grouped = df.groupby('Date of Transaction')

        #Modify DEBIT and DEBIT PAID within each group
        for date, group in grouped:
            last_index = group.index[-1]  #Get the index of the last row in the group
            debit_sum = group['DEBIT'].sum()
            debit_paid_sum = group['DEBIT PAID'].sum()

            df.loc[group.index[:-1], 'DEBIT'] = 0  #Set DEBIT to 0 for all but the last row
            df.loc[last_index, 'DEBIT'] = debit_sum  #Set DEBIT sum in the last row

            df.loc[group.index[:-1], 'DEBIT PAID'] = 0  #Set DEBIT PAID to 0 for all but the last row
            df.loc[last_index, 'DEBIT PAID'] = debit_paid_sum #set DEBIT PAID sum in the last row
            
        #Now calculate cumulative sum and update ACTUAL OPERATING CAPITAL (after handling duplicates)
        df['DEBIT_CUMSUM'] = df['DEBIT'].cumsum()
        df['DEBIT_PAID_CUMSUM'] = df['DEBIT PAID'].cumsum()
        df['ACTUAL OPERATING CAPITAL'] = df['ACTUAL OPERATING CAPITAL'] + df['DEBIT_CUMSUM'] - df['DEBIT_PAID_CUMSUM']
        df.drop(['DEBIT_CUMSUM'], axis=1, inplace=True)
        df.drop(['DEBIT_PAID_CUMSUM'], axis=1, inplace=True)
        return df
        
    """
    #Simple method does not take multiple submissions in a single date
    def update_operating_capital(self, df):
        #Calculate cumulative sum of DEBIT
        df['DEBIT_CUMSUM'] = df['DEBIT'].cumsum()
        #Add the cumulative DEBIT to the initial ACTUAL OPERATING CAPITAL
        df['ACTUAL OPERATING CAPITAL'] = df['ACTUAL OPERATING CAPITAL'] + df['DEBIT_CUMSUM']
        df.drop(['DEBIT_CUMSUM'], axis=1, inplace=True) #Remove helper columns
        return df
    """
    def consolidate_transactions_insert_new_summed_row(self, df):
        """
        Consolidates rows in a Pandas DataFrame with the same 'Date of Transaction' by summing numerical columns 
        and handling string columns as specified.
        Args:
            df: The input DataFrame.
        Returns:
            A new DataFrame with the consolidated rows added.
        """
        #Group by 'Date of Transaction'
        grouped = df.groupby('Date of Transaction')

        new_rows = []
        for date, group in grouped:
            if len(group) > 1:  #Only process groups with more than one row
                new_row = {}

                for col in df.columns:
                    if 'Details' in col or 'INCIDENTS' in col or 'DAY NAME' in col:
                        #Concatenate non-empty strings with ';'
                        strings = [s for s in group[col].astype(str).fillna('') if s.strip()]  #added strip() to remove spaces and make it efficient; fillna keeps missing values out of the join
                        new_row[col] = '; '.join(strings) if strings else None  #Handle empty list case
                    elif col == 'Name of Submitter':
                    
                        #Extract first names and concatenate with ';'
                        names = [str(s).split()[0] for s in group[col] if str(s).strip()] # Check for empty or only space strings
                        new_row[col] = '; '.join(names) if names else None  #Handle empty list case
            
                        
                    elif col == 'Timestamp':
                        #Get date and most recent time 
                        dates = pd.to_datetime(group['Timestamp'], errors='coerce')  #handle invalid Timestamps
                        if not dates.empty: #Added to check for NaT values due to invalid timestamp formats
                            most_recent_datetime = dates.max()
                            new_row[col] = most_recent_datetime.strftime('%m/%d/%Y %H:%M:%S') if not pd.isnull(most_recent_datetime) else None  #added isnull check
                            print(new_row[col])
                        
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        #Sum numeric columns
                        new_row[col] = group[col].sum()
                    else:
                        #Take the first value for other columns
                        new_row[col] = group[col].iloc[0]

                new_rows.append(new_row)

        #Create a new DataFrame from the consolidated rows
        df_new = pd.DataFrame(new_rows)
        
        #Update the index of the new DataFrame to match the original DataFrame
        
        #Concatenate the original DataFrame and the new DataFrame
        df_result = pd.concat([df, df_new], ignore_index=True)

        #Sort the resulting DataFrame by 'Date of Transaction' if needed
        df_result = df_result.sort_values(by='Date of Transaction', na_position='first').reset_index(drop=True)
        
        #Custom Sorting: Place rows with semicolon-separated names last within each Date of Transaction
        #A stable sort on (date, flag) does this in one pass, and unlike groupby().apply() it keeps
        #every column and does not silently drop rows with a missing 'Date of Transaction'
        df_result['semicolon_flag'] = df_result['Name of Submitter'].apply(lambda x: ';' in str(x))
        df_result = df_result.sort_values(
            by=['Date of Transaction', 'semicolon_flag'],
            ascending=True, na_position='first', kind='stable',
        ).drop(columns=['semicolon_flag'])

        return df_result
    
    #Columns whose values are joined rather than summed or taken from one row.
    _CT_TEXT_KEYS = ('Details', 'INCIDENTS', 'DAY NAME')

    def consolidate_transactions(self, df): #Consolidates string values
        """One row per transaction date: numbers summed, free text joined with '; '.

        Vectorised. This was a groupby().apply() over a function that walked every
        column of every group and called Series.sum() on each one - roughly 15
        million pandas calls to fold 546 rows, and five of the six seconds the
        legacy frame took to build. The same work is now done one column at a time
        across all groups at once, which is the shape pandas is fast at.

        The original's quirks are preserved deliberately, because the legacy
        reports are built on top of them:

        - A missing 'Name of Submitter' joins as the literal string 'nan'. The
          original mapped str over the split result, so NaN became 'nan' before
          the dropna() behind it could ever see it.
        - 'Timestamp' takes the group's LAST row in file order, not its latest
          time. Every other unhandled column takes the group's FIRST row.
        - Blank and missing text is dropped from a join, so a group with nothing
          to say joins to '' rather than to '; ; '.

        One behaviour is not preserved, because it could not be: where a group's
        last Timestamp is missing, the original called strftime on NaT, which
        raises ValueError. This returns NaN for that row instead. No input the
        original survived can reach that path, so nothing that worked changes.
        """
        #Convert 'Date of Transaction' and 'Timestamp' to datetime objects
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        key = 'Date of Transaction'
        grouped = df.groupby(key, sort=True)
        keys = df[key]

        #First and last row of each group by position, not by value: groupby.first()
        #and .last() skip missing values, where the original's .iloc[0] / .iloc[-1]
        #did not. cumcount is used rather than head(1).index so a duplicated frame
        #index cannot pull in extra rows. Rows whose date is missing get NaN from
        #cumcount and fall out of both masks, matching groupby's own dropna.
        first_mask = grouped.cumcount() == 0
        last_mask = grouped.cumcount(ascending=False) == 0

        #Start from each group's first row, then overwrite the columns that
        #aggregate. Sorted by date, as grouped.apply() returned them.
        out = df[first_mask].set_index(key).sort_index()

        text_cols = [c for c in df.columns
                     if any(k in c for k in self._CT_TEXT_KEYS)]
        num_cols = [c for c in df.columns
                    if c != key and c != 'Timestamp' and c != 'Name of Submitter'
                    and c not in text_cols and pd.api.types.is_numeric_dtype(df[c])]

        if num_cols:
            out[num_cols] = grouped[num_cols].sum()

        for col in text_cols:
            s = df[col].astype(str).fillna('')
            keep = s.str.strip() != ''
            joined = s[keep].groupby(keys[keep], sort=True).agg('; '.join)
            #A group with no text at all is absent from `joined`, and joins to ''.
            out[col] = joined.reindex(out.index).fillna('')

        if 'Name of Submitter' in df.columns:
            firsts = df['Name of Submitter'].str.split().str[0].map(str)
            out['Name of Submitter'] = (firsts.groupby(keys, sort=True)
                                        .agg('; '.join).reindex(out.index))

        if 'Timestamp' in df.columns:
            last = df[last_mask].set_index(key)['Timestamp'].sort_index()
            out['Timestamp'] = last.reindex(out.index).dt.strftime('%m/%d/%Y %H:%M:%S')

        #Bring the date back out of the index and restore the original column order
        consolidated_df = out.reset_index()
        consolidated_df = consolidated_df.reindex(columns=df.columns).reset_index(drop=True)
        return consolidated_df
        
    def calculate_expected_capital(self, df):
        """
        Calculates the EXPECTED OPERATING CAPITAL based on previous day's 
        ACTUAL OPERATING CAPITAL and other financial data.

        Args:
            df: Pandas DataFrame with 'Name of Submitter', 'Date of Transaction',
                'ACTUAL OPERATING_CAPITAL', 'TOTAL COMMISSION', 'CAPITAL INFUSION',
                'TRANSFER FEES', 'SALARIES', and 'EXPENDITURES' columns.

        Returns:
            Pandas DataFrame with updated 'EXPECTED_OPERATING CAPITAL' column.
        """

        #Ensure the DataFrame is sorted by submitter and date
        df = df.sort_values(['Name of Submitter', 'Date of Transaction'])

        #Convert 'Date of Transaction' to datetime objects if not already
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'])

        df['EXPECTED OPERATING CAPITAL'] =  df['ACTUAL OPERATING CAPITAL']  #Initialize the column

        for submitter in df['Name of Submitter'].unique():
            submitter_df = df[df['Name of Submitter'] == submitter].copy()  #Crucial: use .copy() to avoid SettingWithCopyWarning

            for i in range(1, len(submitter_df)):  #Iterate from the second row onwards for each submitter
                prev_actual = submitter_df['ACTUAL OPERATING CAPITAL'].iloc[i-1]
                commission = submitter_df['TOTAL COMMISSION'].iloc[i]
                infusion = submitter_df['CAPITAL INFUSION'].iloc[i]
                fees = submitter_df['TRANSFER FEES'].iloc[i]
                salaries = submitter_df['SALARIES'].iloc[i]
                expenditures = submitter_df['EXPENDITURES'].iloc[i]
                credit = submitter_df['CREDIT'].iloc[i]
                credit_paid = submitter_df['CREDIT PAID'].iloc[i]
                
                expected = prev_actual + commission + infusion + credit - fees - salaries - expenditures - credit_paid

                submitter_df.loc[submitter_df.index[i], 'EXPECTED OPERATING CAPITAL'] = expected #Use .loc with boolean indexing

            df.loc[submitter_df.index, 'EXPECTED OPERATING CAPITAL'] = submitter_df['EXPECTED OPERATING CAPITAL']  #Update original df
            #df = df.sort_values('Date of Transaction') #Delete this and the below line to maintain order by names
            #df = df.reset_index(drop=True) #Maintain date order from low to highest
        return df
    
    def clean_numeric_columns(self, df):
        """
        Cleans numeric columns in a DataFrame:
        - Converts series of zeros (e.g., "00", "000") to "0.00".
        - Removes leading zeros from numbers (e.g., "0123" -> "123.00").
        - Replaces NaNs, empty strings, and "0" with float 0.00.
        - Leaves excluded columns untouched.

        Parameters:
            df (pd.DataFrame): The input DataFrame to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        #Keywords to exclude
        keywords_exclude = ['Details', 'INCIDENTS', 'Transaction', 'Submission', 'Submitter', 'Timestamp', 'DAY NAME']
    
        #Select relevant columns (exclude those with keywords in their names)
        relevant_cols = [col for col in df.columns if not any(keyword in col for keyword in keywords_exclude)]
    
        #Process each relevant column
        for col in relevant_cols:
            try:
                #Convert all values in the column to strings for cleaning
                df[col] = df[col].astype(str)
                #Handle cases of series of 0s (e.g., "00", "000") by replacing them with "0.00"
                df[col] = df[col].replace(r'^0+$', '0.00', regex=True)
                #Remove leading zeros from numbers (e.g., "0123" -> "123", "0045678" -> "45678") and ensure float format
                df[col] = df[col].replace(r'^0*(\d+)$', r'\1.00', regex=True)
                #Replace NaNs, empty strings, and "0" with float 0.00
                df[col] = df[col].replace(['', ' ', '0', np.nan], '0.00')
                #Convert the column back to numeric (float)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.00)
                #Format all floats to two decimal places (e.g., "2000.00", "23.00")
                df[col] = df[col].apply(lambda x: float(f"{x:.2f}"))
            except Exception as e:
                print(f"Error processing column '{col}': {e}")
                
        #Return the cleaned DataFrame
        return df
    def calculations(self, df): 
        df = df.copy()
        df = self.clean_numeric_columns(self.df)        
        #Sum columns
        keywords = ['COMM', 'INFUSION', 'TRANSFER', 'SALARIES', 'EXPENDITURES', 'INFLOW', 'OUTFLOW', 'EXCESS', 'LOSS', 'CREDIT', 'DEBIT']
        exclusions = ['Details', 'INCIDENTS', 'Transaction', 'Submitter', 'Timestamp', 'DAY NAME']
        pattern = f"^.*(?:{'|'.join(keywords)}).*$(?!.*(?:{'|'.join(exclusions)})).*$"  #Advanced regex
        df.loc["COLUMN TOTALS"] = df.filter(regex=pattern).sum(numeric_only=True, axis=0)
        
        def calculate_averages(df):
            """Calculates averages of specific columns in a Pandas DataFrame, excluding the last row.
            Args:
                df: The Pandas DataFrame.
            Returns:
                A Pandas Series containing the calculated averages, or None if no matching columns are found.
            """
            keywords = []  #If you have any keywords to INCLUDE, put them here
            exclusions = ["DETAILS", "INCIDENTS", "TRANSACTION", "SUBMITTER", "TIMESTAMP", "DAY NAME"]

            #Exclude the last row from calculations
            df_calc = df.iloc[:-1, :]
            #Filter columns based on keywords and exclude keywords
            filtered_columns = []
            for col in df_calc.columns:
                if (not keywords or any(keyword in col.upper() for keyword in keywords)) and not any(exclude_keyword in col.upper() for exclude_keyword in exclusions):  #Case-insensitive matching
                    filtered_columns.append(col)

            if not filtered_columns:
                return None  #No matching columns found

            averages = {}
            for col in filtered_columns:
                if 'COMM' in col.upper() or 'COMMISSION' in col.upper() or 'CREDIT' in col.upper() or 'DEBIT' in col.upper(): #Special handling for COMM/COMMISSION, CREDIT, DEBIT
                    values = df_calc[col][df_calc[col] > 0]  # Consider only values > 0
                    if not values.empty:  # Check if there are any values > 0
                        averages[col] = values.mean()
                else:
                    #Handle potential non-numeric values gracefully, skipping them in the average
                    numeric_values = pd.to_numeric(df_calc[col], errors='coerce')
                    if not numeric_values.isnull().all(): # Check if all values are NaN after conversion.
                        averages[col] = numeric_values.mean()

            return pd.Series(averages, name="AVERAGE AMOUNT")

        #Example Usage (assuming you have a DataFrame 'df'):

        #Apply the function and add the results as a new row
        result_series = calculate_averages(df)

        if result_series is not None:
            df.loc["AVERAGE AMOUNT"] = result_series
        #df.loc["AVERAGE AMOUNT"] = df.iloc[:-1].filter(regex="^(?!.*(?:Details|INCIDENTS|Transaction|Submitter|Timestamp)).*$").mean(numeric_only=True, axis=0)
        #Column maximums
        df.loc["COLUMN MAXIMAMUS"] = df.iloc[:-2].filter(regex="^(?!.*(?:Details|INCIDENTS|Transaction|Submitter|Timestamp)).*$").max(numeric_only=True, axis=0)
        #Column minimums
        df.loc["COLUMN MINIMUMS"] = df.iloc[:-3].filter(regex="^(?!.*(?:Details|INCIDENTS|Transaction|Submitter|Timestamp)).*$").min(numeric_only=True, axis=0)
        
        return df
        
    def format_data(self, x):
        if pd.notnull(x):
            if isinstance(x, float):
                try:
                    return "{:,.2f}".format(float(x))
                except ValueError:
                    return x
            else:
                return x
        else:
            return ""
        
    def generate_html_table(self, df, output_file):
        from pretty_html_table import build_table
        #Create custom CSS styles
        css_styles = """
        <style scoped>
        .dataframe-div {
            max-height: 750px; /* Limit height for scrolling */
            overflow: auto; /* Enable scroll if content is too large */
            position: relative;
            }
 
        .dataframe thead th {
             position: -webkit-sticky; /* Sticky header for webkit browsers */
             position: sticky; /* Sticky header */
             top: 0;
             background: green; /*#4CAF50*/
             color: darkblue; /*white*/
             z-index: 1;
             }
 
        .dataframe thead th:first-child {
            left: 0;
            z-index: 1;
            }
 
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
            }
 
        .dataframe tbody tr th {
            position: -webkit-sticky; /* for Safari */
            position: sticky;
            left: 0;
            background: blue; /*white*/
            color: green;
            vertical-align: top;
            z-index: 1;
            }
 
        /*Additional styles*/
 
        .dataframe table {
            width: 100%; /* Make table take full available width */
            table-layout: fixed; /* Required for column width control */
            border-collapse: collapse; /* Ensure proper table borders */
            /*font-size: 2.0rem; Base font size. Adjust as needed. */
            }
    
         /*.dataframe th,*/
         
         .dataframe td {
            /*overflow: hidden;   Hide text that overflows the cell */
            /*text-overflow: ellipsis;  Add ellipsis for overflow */
            white-space: nowrap; /*Prevent text wrapping */
            /*max-width: 100ch; Ensure the cell width does not exceed 100 characters */
            line-height: 2.0rem;
            padding: 8px;
            border: 1px solid #ddd;
            overflow-x: auto; /* Allow horizontal scrolling within cells if needed */
            box-sizing: border-box; /* Ensures padding and border are included in cell width */
            }
 
        /*.dataframe th {
            background-color: #f2f2f2;
            }*/
            
         /* Media query for smaller screens */
         @media screen and (max-width: 767px) {
         .dataframe table {
             width: 100%; /* Ensures table takes full width */
             border-collapse: collapse; /* Improves visual clarity */
             font-size: 2.5rem !important; /*Increased font size for mobile */
             }
         .dataframe th,
         .dataframe td {
          /* Consider further adjustments for smaller screens, e.g., smaller padding */
              padding: 2px; /* Adds padding for better spacing */
              border: 1px solid #ccc; /* Adds subtle borders for clarity */
              }
          }
         """
        css_styles += "</style>"
 
        df.columns = pd.MultiIndex.from_product([[(f"Automated Daily UWAKALA Business Financial Reports Generated at TSSFL Technology Stack - www.tssfl.com on  {pd.Timestamp.now(tz='Africa/Nairobi').strftime('%d-%m-%Y %H:%M:%S')} Estern Africa Time")], df.columns])
 
        #Build the HTML table using pretty_html_table
        df_html = build_table(
        df,
        'green_light',
        font_size='large',
        font_family='Open Sans, sans-serif',
        text_align='left', width = 'auto',
        index=True,
        even_color='darkblue',
        even_bg_color='#c3d9ff',
        )
 
        #Combine CSS styles with the HTML table
        final_html = css_styles + '<div class="dataframe-div">' + df_html + "</div>"
 
        #Write the final HTML to a file
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(final_html)
     
        return final_html
    
    def mini_report(self, df):
        mini_df = df[['Date of Transaction'] + list(df.loc[:, 'ACTUAL OPERATING CAPITAL':'DEBIT PAID'].columns)]
        return mini_df
            
    #This method is for brief report - select a df subset
    def subset_df(self, df):
        #Subset a dataframe
        subset_df = df[[col for col in df.columns if any(kw in col for kw in ["Timestamp", "Submitter", "TOTAL", "AGENCY", "BUNDLES", "ACTUAL", "EXPECTED", "INFUSION", "TRANSFER", "SALARIES", "EXPENDITURES", "HARD", "Transaction", "INCIDENTS"])]]
        
        return subset_df
        
    #Subset a df for Float Summary
    def summary_df(self, df):
        #Remove columns that starts with TOTAL and those contained the keywords shown
        summary_df = df.loc[:, ~df.columns.str.startswith('TOTAL') & ~df.columns.str.contains('COMM|LIPA|INFUSION|TRANSFER|SALARIES|EXPENDITURES|HARD|ACTUAL|EXPECTED|EXCESS|LOSS|CREDIT|DEBIT|Details|INCIDENTS', case=False)]
        cols = summary_df.columns.drop(['Date of Submission', 'Name of Submitter', 'Date of Transaction'])
        sorted_cols = sorted(cols)
        summary_df = summary_df[['Date of Submission', 'Name of Submitter', 'Date of Transaction'] + sorted_cols]
        return summary_df
    
    def rearrange_columns(self, df):
        """Rearranges DataFrame columns alphabetically, keeping specific columns in place."""
        fixed_cols = [col for col in df.columns if col.startswith("TOTAL") or 
                 "INFLOW" in col or "OUTFLOW" in col]
        movable_cols = sorted([col for col in df.columns if col not in fixed_cols])

        #More efficient way to interleave the lists
        reordered_cols = []
        fixed_iter = iter(fixed_cols)
        movable_iter = iter(movable_cols)

        for col in df.columns:
            if col in fixed_cols:
                reordered_cols.append(next(fixed_iter))
            else:
                reordered_cols.append(next(movable_iter))

        return df[reordered_cols]
             
    def _full_report(self, report_type = "default_report_type", render=True):
         if self.data_source == 'google_drive':
            self.df = pd.DataFrame(self.data).copy()
         elif self.data_source == 'local_drive' or self.data_source == 'dropbox':
            self.df = self.data.copy()
         else: #Kobo
              self.df = self.df.copy()
         
         #Calculations for total lipa charges, total transfer fees, total mobile commission, total bank commission, etc.
         if self.df is None:
            print("DataFrame is not available. Please call process_data() first.")
            return

         #Convert to string types, handling 0, 0.0, and NaN
         for col in self.df.columns:
             if any(keyword in col for keyword in ["Details", "INCIDENTS"]):
                 self.df[col] = self.df[col].astype(str).fillna('').replace(r'^(0\.0|0|nan)$', '', regex=True)
         #Convert 'Date of Transaction' column to datetime
         self.df['Date of Transaction'] = pd.to_datetime(self.df['Date of Transaction'], format='%m/%d/%Y')
         self.df['Grouped Date'] = self.df['Date of Transaction'].dt.date
         self.df['Grouped Date'] = self.df.groupby('Grouped Date')['Grouped Date'].transform('first')
         self.df['Date of Transaction'] = self.df['Grouped Date']
         self.df = self.df.sort_values('Date of Transaction')
         self.df = self.df.reset_index(drop=True) #Maintain date order from low to highest
         self.df = self.df.drop('Grouped Date', axis=1)
         
         #New patch
         #Remove spaces before, after, and reduce multiple spaces to a single space in column names
         self.df.columns = self.df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

         #Base keywords
         base_key = ['AIRTEL MONEY ', 'AIRTEL LIPA ', 'VODA LIPA ', 'HALO LIPA ', 'TIGO-PESA ', 'TIGO LIPA ', 'M-PESA ', 'HALO-PESA ', 'AZAM PESA ', 'CRDB BANK ', 'NMB BANK ', 'NBC BANK ', 'EQUITY BANK ', 'SELCOM ', 'AZANIA BANK ']
 
         #Initialize the grouped column list
         grouped_column_list = []
 
         #Define patterns for grouping
         patterns = ["", "SUPERAGENT", "COMM", "SUPERAGENT COMM"]
 
         #Extract column names
         columns = self.df.columns.tolist()
 
         #Group columns based on patterns
         for key in base_key:
             for pattern in patterns:
                 group = [
                     col for col in columns
                     if col.startswith(key) and pattern in col and
                     col.count(' ') == (key.count(' ') + (1 if pattern else 0) + (1 if 'COMM' in pattern and 'SUPERAGENT' in pattern else 0))
                 ]
                 if group and tuple(group) not in grouped_column_list:
                     grouped_column_list.append(tuple(group))
 
         #Check if any groups were formed
         if grouped_column_list:
             #Add columns that do not match any base key - check if those columns exist first
             other_columns = [col for col in columns if not any(col.startswith(k) for k in base_key)]
 
             #Flatten the grouped list and append non-matching columns
             ordered_columns = [col for group in grouped_column_list for col in group] + other_columns
 
             #Reorder DataFrame columns
             self.df = self.df[ordered_columns]
 
         for group in grouped_column_list:
         #Check if the group has more than one column
             if len(group) > 1:
                 #Extract base name (e.g., "AIRTEL MONEY", "TIGO PESA")
                 base_name = re.sub(r'\d+', '', group[0]).strip()  #Using regex to remove numbers
                 base_name = re.sub(r'\s+', ' ', base_name) #Remove multiple spaces
                 #Or use one liner instead of the two above:
                 #base_name = re.sub(r'\s*(?:\d+\s*)+', ' ', group[0]).strip()
                 total_col_name = f"{base_name} TOTAL"
                 #Ensure all columns in the group are numeric
                 self.df[list(group)] = self.df[list(group)].apply(pd.to_numeric, errors='coerce')
                 #Calculate sum and insert new column
                 self.df.insert(self.df.columns.get_loc(group[-1]) + 1, total_col_name, self.df[list(group)].sum(axis=1))
        
         self.df = self.clean_numeric_columns(self.df)
         #NORMAL MOBILE FLOAT TOTAL
         #Keywords to exclude
         exclude_keywords = ["BANK", "COMM", "SUPERAGENT", "LIPA", "TOTAL", "SELCOM", 
                    "AGENCY", "INFUSION","TRANSFER", "SALARIES", "EXPENDITURES", "HARD", "Timestamp", "Submitter", "Details", "INCIDENTS", "Transaction", 'CREDIT', 'DEBIT']
                    
         #Columns to sum for "NORMAL MOBILE FLOAT TOTAL"
         normal_mobile_columns = [col for col in self.df.columns if not any(keyword in col for keyword in exclude_keywords)]  
         if normal_mobile_columns:
                self.df['TOTAL NORMAL MOBILE FLOAT'] = self.df[normal_mobile_columns].sum(axis=1) 
         else: 
             pass  #absent provider block is normal - each agent carries a different mix
          
         #SUPER AGENT MOBILE FLOAT TOTAL
         #Keywords to exclude
         exclude_keywords = ["BANK", "COMM", "LIPA", "TOTAL", "SELCOM"]
         #Keyword to include
         include_keyword = "SUPERAGENT"
 
         #Columns to sum for "TOTAL SUPERAGENT MOBILE FLOAT"
         superagent_mobile_columns = [col for col in self.df.columns if
                              include_keyword in col and
                              not any(keyword in col for keyword in exclude_keywords)]
 
         #Calculate the sum of the selected columns
         if superagent_mobile_columns:  # Check if any columns match the criteria
             self.df['TOTAL SUPERAGENT MOBILE FLOAT'] = self.df[superagent_mobile_columns].sum(axis=1)
         else:
             pass  #absent provider block is normal - each agent carries a different mix
             
         #LIPA MOBILE FLOAT TOTAL
         #Keywords to exclude
         exclude_keywords = ["COMM", "TOTAL"]
         include_keyword = "LIPA"
 
         #Columns to sum for "LIPA MOBILE FLOAT TOTAL"
         lipa_mobile_columns = [
             col for col in self.df.columns
             if include_keyword in col and not any(keyword in col for keyword in exclude_keywords)
         ]   
 
         #Calculate the sum of the selected columns
         if lipa_mobile_columns:  # Check if any columns match the criteria
             self.df['TOTAL LIPA MOBILE FLOAT'] = self.df[lipa_mobile_columns].sum(axis=1)
         else:
             pass  #absent provider block is normal - each agent carries a different mix
       
         #TOTAL SELCOM FLOAT
         selcom_cols = [col for col in self.df.columns if "SELCOM" in col and all(kw not in col for kw in ["COMM", "TOTAL"])]
         if selcom_cols:
             self.df['SELCOM FLOAT TOTAL'] = self.df[selcom_cols].sum(axis=1)  
         else:
             pass  #absent provider block is normal - each agent carries a different mix
 
         #NORMAL BANK FLOAT TOTAL
         bank_cols = [col for col in self.df.columns if "BANK" in col and all(kw not in col for kw in ["SUPERAGENT", "TOTAL", "COMM"])]
         if bank_cols:
             self.df['TOTAL NORMAL BANK FLOAT'] = self.df[bank_cols ].sum(axis=1)   
         else:
             pass  #absent provider block is normal - each agent carries a different mix
 
         #SUPERAGENT BANK FLOAT TOTAL
         sup_bank_cols = [col for col in self.df.columns if all(kw in col for kw in ["BANK", "SUPERAGENT"]) and all(kw not in col for kw in ["TOTAL", "COMM"])]
         if sup_bank_cols:
             self.df['TOTAL SUPERAGENT BANK FLOAT'] = self.df[sup_bank_cols].sum(axis=1) 
         else:
             pass  #absent provider block is normal - each agent carries a different mix

         #NORMAL MOBILE COMMISSION TOTAL - includes MOBILE BUNDLES COMM and SHARES
         mobile_comm_cols = [col for col in self.df.columns if "COMM" in col and all(kw not in col for kw in ["BANK", "SUPERAGENT", "LIPA", "TOTAL", "SELCOM", "AGENCY", "Details"])]
         if mobile_comm_cols:
             self.df['TOTAL NORMAL MOBILE COMMISSION'] = self.df[mobile_comm_cols].sum(numeric_only=True, axis=1)
         else:
             pass  #absent provider block is normal - each agent carries a different mix
 
         #SUPERAGENT MOBILE COMMISSION TOTAL
         sup_mobile_comm_cols = [col for col in self.df.columns if all(kw in col for kw in ["COMM", "SUPERAGENT"]) and all(kw not in col for kw in ["BANK", "LIPA", "TOTAL", "SELCOM"])]
         if sup_mobile_comm_cols:
             self.df['TOTAL SUPERAGENT MOBILE COMMISSION'] = self.df[sup_mobile_comm_cols ].sum(axis=1)   
         else:
             pass  #absent provider block is normal - each agent carries a different mix

         #LIPA MOBILE COMMISSION TOTAL
         lipa_comm_cols = [col for col in self.df.columns if all(kw in col for kw in ["LIPA", "COMM"]) and all(kw not in col for kw in ["TOTAL"])]
         if lipa_comm_cols:
             self.df['TOTAL LIPA MOBILE COMMISSION'] = self.df[lipa_comm_cols].sum(axis=1)   
         else:
             pass  #absent provider block is normal - each agent carries a different mix

         #TOTAL SELCOM COMMISSION
         selcom_comm_cols = [col for col in self.df.columns if all(kw in col for kw in ["SELCOM", "COMM"]) and all(kw not in col for kw in ["TOTAL"])]
         if selcom_comm_cols:
             self.df['TOTAL SELCOM COMMISSION'] = self.df[selcom_comm_cols].sum(axis=1)   
         else:
             pass  #absent provider block is normal - each agent carries a different mix

         #NORMAL BANK COMMISSION TOTAL
         bank_comm_cols = [col for col in self.df.columns if all(kw in col for kw in ["BANK", "COMM"]) and all(kw not in col for kw in ["SUPERAGENT", "TOTAL"])]
         if bank_comm_cols:
             self.df['TOTAL NORMAL BANK COMMISSION'] = self.df[bank_comm_cols].sum(axis=1)   
         else:
             pass  #absent provider block is normal - each agent carries a different mix

         #SUPERAGENT BANK COMMISSION TOTAL
         sup_bank_comm_cols = [col for col in self.df.columns if all(kw in col for kw in ["BANK", "SUPERAGENT", "COMM"]) and all(kw not in col for kw in ["TOTAL"])]
         if sup_bank_comm_cols:
             self.df['TOTAL SUPERAGENT BANK COMMISSION'] = self.df[sup_bank_comm_cols].sum(axis=1)
         else:
             pass  #absent provider block is normal - each agent carries a different mix
    
         #TOTAL MOBILE COMMISSION
         cols = [col for col in self.df.columns if "COMM" in col and all(kw not in col for kw in ["BANK", "TOTAL", "SELCOM", "AGENCY", "Details"])]
         if cols: 
             self.df['TOTAL MOBILE COMMISSION'] = self.df[cols].sum(numeric_only=True, axis=1)
         else: 
             pass  #absent provider block is normal - each agent carries a different mix
    
         #TOTAL BANK COMMISSION
         cols = [col for col in self.df.columns if all(kw in col for kw in ["BANK", "COMM"]) and all(kw not in col for kw in ["TOTAL"])]
         if cols: 
             self.df['TOTAL BANK COMMISSION'] = self.df[cols].sum(axis=1)
         else: 
             pass  #absent provider block is normal - each agent carries a different mix
    
         #TOTAL COMMISSION
         cols = [col for col in self.df.columns if "COMM" in col and all(kw not in col for kw in ["TOTAL"])]
         if cols: 
             self.df['TOTAL COMMISSION'] = self.df[cols].sum(axis=1)
         else: 
             pass  #absent provider block is normal - each agent carries a different mix
    
         #TOTAL MOBILE FLOAT
         cols = [col for col in self.df.columns if not any(kw in col for kw in ["BANK", "TOTAL", "COMM", "SELCOM", "AGENCY", 
                                                                  "INFUSION","TRANSFER", "SALARIES", "EXPENDITURES", "HARD", "Timestamp", "Submitter", "Details", "INCIDENTS", "Transaction", 'CREDIT', 'DEBIT'])]
         if cols:
             self.df['TOTAL MOBILE FLOAT'] = self.df[cols].sum(axis=1)
         else:
             pass  #absent provider block is normal - each agent carries a different mix
    
         #TOTAL BANK FLOAT
         cols = [col for col in self.df.columns if "BANK" in col and all(kw not in col for kw in ["TOTAL", "COMM"])]
         if cols: 
             self.df['TOTAL BANK FLOAT'] = self.df[cols].sum(axis=1)
         else: 
             pass  #absent provider block is normal - each agent carries a different mix
    
         #TOTAL FLOAT
         cols = [col for col in self.df.columns if not any(keyword in col for keyword in ["COMM", "TOTAL", "INFUSION","TRANSFER", "SALARIES", "EXPENDITURES", "HARD", "Timestamp", "Submitter", "Details", "INCIDENTS", "Transaction", 'CREDIT', 'DEBIT'])]
         if cols: 
             self.df['TOTAL FLOAT'] = self.df[cols].sum(axis=1)
         else: 
             pass  #absent provider block is normal - each agent carries a different mix
         
         #Sort the dataframe alphabetically
         #self.df = self.rearrange_columns(self.df) #Apply this method or
         #self.df = self.df.reindex(sorted(self.df.columns), axis=1)  #Most efficient one liner
         self.df = self.df[[col for col in self.df.columns if col not in ['CAPITAL INFUSION', 'SALARIES', 'EXPENDITURES', 'TRANSFER FEES', 'HARD CASH', 'TOTAL FLOAT']] + ['CAPITAL INFUSION', 'SALARIES', 'EXPENDITURES', 'TRANSFER FEES', 'HARD CASH'] + ['TOTAL FLOAT']]
         
         #Calculate the sum of the specified columns
         self.df['ACTUAL OPERATING CAPITAL'] = self.df['HARD CASH'] + self.df['TOTAL FLOAT']
         self.df = self.update_operating_capital(self.df)
         self.df['ACTUAL OPERATING CAPITAL'] = self.df['ACTUAL OPERATING CAPITAL']
         
         if report_type == 'comp':
             self.df = self.calculate_expected_capital(self.df)
             self.df = self.consolidate_transactions_insert_new_summed_row(self.df)
             
         else:
             self.df = self.consolidate_transactions(self.df) #Consolidate multiple rows for the same date
             #Expected here means the capital that you should have after some changes compared to the previous one
             #EXPECTED OPERATING CAPITAL
             self.df.insert(self.df.columns.get_loc('ACTUAL OPERATING CAPITAL') + 1, 'EXPECTED OPERATING CAPITAL', self.df.loc[1:, ['TOTAL COMMISSION', 'CAPITAL INFUSION', 'CREDIT']].sum(numeric_only=True, axis=1) - self.df.loc[1:, ['TRANSFER FEES', 'SALARIES', 'EXPENDITURES', 'CREDIT PAID']].sum(numeric_only=True, axis=1) + self.df['ACTUAL OPERATING CAPITAL'].shift(1))
             self.df.at[0, 'EXPECTED OPERATING CAPITAL'] = self.df.at[0, 'ACTUAL OPERATING CAPITAL']

         #Excess/Loss
         self.df.insert(self.df.columns.get_loc('EXPECTED OPERATING CAPITAL') + 1, 'EXCESS/LOSS', self.df['ACTUAL OPERATING CAPITAL'] - self.df['EXPECTED OPERATING CAPITAL'])

         #Excess
         self.df.insert(self.df.columns.get_loc('EXCESS/LOSS') + 1, 'EXCESS', self.df['EXCESS/LOSS'].apply(lambda x: x if x > 0 else 0))
         #Loss
         self.df.insert(self.df.columns.get_loc('EXCESS') + 1, 'LOSS', self.df['EXCESS/LOSS'].apply(lambda x: abs(x) if x < 0 else 0))

         #Move EXCESS/LOSS column two steps further
         self.df.insert(self.df.columns.get_loc('EXCESS/LOSS') + 2, 'EXCESS/LOSS', self.df.pop('EXCESS/LOSS'))

         #Total cash inflow
         self.df.insert(self.df.columns.get_loc('TOTAL FLOAT') + 1, 'TOTAL CASH INFLOW', self.df.loc[:, ['TOTAL COMMISSION', 'CAPITAL INFUSION', 'EXCESS']].sum(numeric_only=True, axis=1))
         #Total cash outflow
         self.df.insert(self.df.columns.get_loc('TOTAL CASH INFLOW') + 1, 'TOTAL CASH OUTFLOW', self.df.loc[:, ['TRANSFER FEES', 'SALARIES','EXPENDITURES']].sum(numeric_only=True, axis=1))
         
         self.df.rename(columns={'Timestamp': 'Date of Submission'}, inplace=True) 
         #Move and rearrange columns
         cols_to_left = ['Date of Submission', 'Name of Submitter', 'Date of Transaction']
         cols_to_right = ['DEBIT Details', 'DEBIT PAID Details', 'CREDIT Details', 'CREDIT PAID Details', 'CAPITAL INFUSION Details', 'TRANSFER FEES Details', 'SALARIES Details', 'EXPENDITURES Details', 'Transaction Anomalies and Irregularities Details', 'INCIDENTS']
         
         #Move CREDIT and DEBIT columns next to EXCESS/LOSS column
         self.df = self.df.reindex(columns=(list(self.df.columns.drop(['CREDIT', 'DEBIT', 'CREDIT PAID', 'DEBIT PAID', 'EXCESS/LOSS'])) + ['EXCESS/LOSS', 'DEBIT', 'DEBIT PAID', 'CREDIT', 'CREDIT PAID']))
         
         #Check if 'MOBILE BUNDLES and SHARES Details' exists and add it to the list of cols_to_right
         if 'MOBILE BUNDLES and SHARES Details' in self.df.columns:
             cols_to_right.insert(4, 'MOBILE BUNDLES and SHARES Details')
         other_cols = [col for col in self.df.columns if col not in cols_to_left + cols_to_right]

         self.df = self.df[cols_to_left + other_cols + cols_to_right]
         
         self.df = self.df.reset_index(drop=True)
         self.df.index = self.df.index + 1
         
         if report_type != 'comp':  #Skip calculations if report_type is 'comprehensive'
             df = self.calculations(self.df)
         else:
            df = self.df  #Directly use self.df without calculations
            
         df = df.map(self.format_data)
         df = self.date_time(df)
         
         #Reports
         if report_type == 'mini':
             df = self.mini_report(df)
             output_file = 'Mini_DFB_Report.html'
             
         elif report_type == 'brief':
            df = self.subset_df(df)
            output_file = 'Brief_DFB_Report.html'
            
         elif report_type == 'summary':
             df = self.summary_df(df)
             output_file = 'Summary_DFB_Report.html'
        
         elif report_type == 'comp':
             output_file = 'Extended_DFB_Report.html'
        
         else:
            output_file = 'Full_DFB_Report.html'

         #Rendering is the expensive half - build_table writes an inline style= on
         #every cell (~58,000 for a full dataset): ~10s and ~9MB. Callers that only
         #need the calculated frame pass render=False.
         if render:
             self.generate_html_table(df, output_file)
         
         return self.df
    
    #COMM Report - Month Year
    def comm_report(self):
        """
        Generates a COMM report summarizing columns with 'COMM' in their names for each month and year.

        Args:
            self:  The class instance containing a Pandas DataFrame called self.df.
        """
        df = self.df.copy()

        #Ensure 'Date of Transaction' is datetime
        try:
            df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'], format='%m/%d/%Y', errors='coerce')
        except ValueError as e:
            print(f"Error converting 'Date of Transaction' column: {e}. Check date format.")
            return  #Exit if date conversion fails

        #Identify COMM columns
        comm_cols = [col for col in df.columns if 'COMM' in col]
        if not comm_cols:
            pass  #absent provider block is normal - each agent carries a different mix
            return

        #Group by month and year, sum COMM columns
        df['MONTH YEAR'] = df['Date of Transaction'].dt.strftime('%B %Y')
        df['Year'] = df['Date of Transaction'].dt.year
        df['Month'] = df['Date of Transaction'].dt.month
    
        df = df.groupby(['Year', 'Month', 'MONTH YEAR'])[comm_cols].sum().reset_index()

        #Sort chronologically
        df = df.sort_values(by=['Year', 'Month'])
        
        #Delete columns
        df = df.drop(['Year', 'Month'], axis=1)
        
        #Calculate totals, averages, etc.
        cal = pd.DataFrame({
        'TOTAL COMM': df.loc[:, df.columns != 'MONTH YEAR'].sum(),
        'AVERAGE COMM': df.loc[:, df.columns != 'MONTH YEAR'].mean(),
        'HIGHEST COMM': df.loc[:, df.columns != 'MONTH YEAR'].max(),
        'LOWEST COMM': df.loc[:, df.columns != 'MONTH YEAR'].min()
        }).T
        
        #Change the index to start with 1
        df.index = np.arange(1, len(df) + 1)
        
        df = pd.concat([df, cal])
        df = df.map(self.format_data)
        
        output_file = 'COMM_Report.html'
        self.generate_html_table(df, output_file)
        
    #Slice a dataframe based on a month for all years - full report
    def specific_month_for_all_years_report(self, month, report_type):
        df = self.df
    
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'], format='%m/%d/%Y')
        
        target_month = month
        
        #Filter rows based on the target month in 'Date of Transaction' column
        df = df[df['Date of Transaction'].dt.month == target_month] 
        
        df = self.calculations(df)
        df = df.map(self.format_data)
        df = self.date_time(df)
        #Report
        if report_type == 'brief':
            df = self.subset_df(df)
            output_file = 'Target_Month_Brief_DFB_Report.html'
        else:
            output_file = 'Target_Month_Full_DFB_Report.html'

        self.generate_html_table(df, output_file)
        
    #Target month and target year   
    def specific_month_of_year_report(self, month, year, report_type):
        df = self.df
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'], format='%m/%d/%Y')
        #Specify the target month and year
        target_month = month  #January
        target_year = year
        #Create a mask for rows within the target month and year
        mask = (df['Date of Transaction'].dt.month == target_month) & (df['Date of Transaction'].dt.year == target_year)
        #Filter the DataFrame based on the mask
        df = df.loc[mask]
        
        df = self.calculations(df)
        df = df.map(self.format_data)
        df = self.date_time(df)
        #Report
        if report_type == 'brief':
            df = self.subset_df(df)
            output_file = 'Target_Month_and_Year_Brief_DFB_Finance_Report.html'
        else:
            output_file = 'Target_Month_and_Year_Full_DFB_Finance_Report.html'

        self.generate_html_table(df, output_file)
        
    #Slice based on the week days
    def weekdays_report(self, report_type):
        df = self.df
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'], format='%m/%d/%Y')
        #Create a new column 'DAY NAME' with weekday names
        df['DAY NAME'] = df['Date of Transaction'].dt.day_name()
        #Specify the target weekday(s) you want to filter for
        target_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        #Create a mask for rows with the desired weekdays
        mask = df['DAY NAME'].isin(target_weekdays)
        #Filter the DataFrame based on the mask
        df = df.loc[mask]
        df = df[['DAY NAME'] + [col for col in df.columns if col != 'DAY NAME']]
        
        #Compute sums, averages, maxs and mins
        df = self.calculations(df)
    
        df = df.map(self.format_data)
        df = self.date_time(df)
        #Report
        if report_type == 'brief':
            df = self.subset_df(df)
            output_file = 'WeekDays_Brief_DFB_Finance_Report.html'
        else:
            output_file = 'WeekDays_Full_DFB_Finance_Report.html'

        self.generate_html_table(df, output_file)
    
    def weekdays_of_target_month_and_year_report(self, month, year, report_type):
        df = self.df
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'], format='%m/%d/%Y')
        #Specify the target year and month
        target_year = year
        target_month = month
        #Create a mask for rows within the target year and month
        mask = (df['Date of Transaction'].dt.year == target_year) & (df['Date of Transaction'].dt.month == target_month)
        #Filter the DataFrame based on the mask
        df = df.loc[mask]
        #Create a new column 'DAY NAME' with weekday names
        df['DAY NAME'] = df['Date of Transaction'].dt.day_name()
        #Specify the target weekday(s) you want to filter for
        target_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        #Create a mask for rows with the desired weekdays
        weekday_mask = df['DAY NAME'].isin(target_weekdays)
        #Filter the sub DataFrame based on the weekday mask
        df = df.loc[weekday_mask]
        df = df[['DAY NAME'] + [col for col in df.columns if col != 'DAY NAME']]
        
        #Compute sums, averages, maxs and mins
        df = self.calculations(df)  #Right shift by +1

        df = df.map(self.format_data)
        df = self.date_time(df)
        
        #Report
        if report_type == 'brief':
            df = self.subset_df(df)
            output_file = 'WeekDays_of_target_Month_and_Year_Brief_DFB_Report.html'
        else:
            output_file = 'WeekDays_of_target_Month_and_Year_Full_DFB_Report.html'

        self.generate_html_table(df, output_file)
    
    
    def date_range_report(self, start_date, end_date, report_type):
        df = self.df.copy()
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'], format='%d/%m/%Y')
        #Convert start_date and end_date strings to datetime objects
        start_date = pd.to_datetime(start_date, format='%d/%m/%Y')
        end_date = pd.to_datetime(end_date, format='%d/%m/%Y')
        #Filter rows based on the date range
        df = df[(df['Date of Transaction'] >= start_date) & (df['Date of Transaction'] <= end_date)]

        #If the dataframe is empty after filtering, return early
        if df.empty:
            print("No data found for the specified date range.")
            return

        df = self.calculations(df)
        df = df.map(self.format_data)  #Assuming format_data works element-wise or with .apply()
        df = self.date_time(df)
        
        #Reports
        if report_type == 'brief':
            df = self.subset_df(df)
            output_file = f'Brief_DFB_Report_{start_date.strftime("%d%m%Y")}_{end_date.strftime("%d%m%Y")}.html'
            
        elif report_type == 'full':
            output_file = f'Full_DFB_Report_{start_date.strftime("%d%m%Y")}_{end_date.strftime("%d%m%Y")}.html' 
            
        elif report_type == 'summary':
            otput_file = f'Summary_DFB_Report_{start_date.strftime("%d%m%Y")}_{end_date.strftime("%d%m%Y")}.html'
            
        else:
            output_file = f'Extended_DFB_Report_{start_date.strftime("%d%m%Y")}_{end_date.strftime("%d%m%Y")}.html'

        self.generate_html_table(df, output_file)

    def graphs(self, date, report_type):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pretty_html_table import build_table
        #top_ten, lower_ten, full
        df = self.df
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'], format='%d/%m/%Y')
        most_recent_date = df['Date of Transaction'].max().strftime('%d/%m/%Y')
        df = self.date_time(df)
        
        date = date
        duplicate_count = 0
        if date is None:
            date = most_recent_date
            df_selected_initial = df[df['Date of Transaction'] == most_recent_date].copy()  # Use most recent date if none provided
            #Filter for rows where 'Name of Submitter' has two strings separated by ';'
            df_selected = df_selected_initial[df_selected_initial['Name of Submitter'].str.contains(';', na=False)]

            #If multiple rows match the criteria, pick the last one
            if len(df_selected) > 1:
                df_selected = df_selected.head(-1) 
            #If no rows match criteria after the semicolon check, fall back to the original behavior.
            if df_selected.empty:
                df_selected = df_selected_initial.sample(n=1)
        else:
            filtered_df = df[df['Date of Transaction'] == date]
            if filtered_df.empty:
                #Handle the case where there are no transactions on the specified date
                print(f"No transactions found for date: {date}") #Print only the date part
                return  #Exit the function early
            else:
                duplicate_count = filtered_df.duplicated(subset='Date of Transaction').sum()
                df_selected = filtered_df.sample(n=1)

        print("Number of duplicate rows found:", duplicate_count)
       
        #Filter or subset the df to exclude columns names with the given keywords
        df_selected = df_selected[[col for col in df_selected.columns if not any(keyword in col for keyword in ["Submission", "Submitter", "Transaction", "Details", "INCIDENTS"])]]
        if df_selected.empty:
            print("No valid data to display.")
            return  #Exit the function early if no valid data
        df = df_selected.T
        #df.columns = ['Amount']
        df = df[['Amount']] if 'Amount' in df.columns else df.iloc[:, [0]].rename(columns={df.columns[0]: 'Amount'}) #Take last row
        #Reset the index and name the index column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Description'}, inplace=True)
        #keep rows where the 'Amount' column is neither zero, NaN, nor a string
        df = df[~((df['Amount'] == 0) | (df['Amount'].isna()) | (df['Amount'].apply(lambda x: isinstance(x, str))))]
            
        df = df.reset_index(drop=True)  #Reset the existing index
        df.index = df.index + 1       #Add 1 to the reset index
         
        df1 = df #Redefine df for tabular formatted data
        df1.rename(columns={0: 'Description'}, inplace=True)
        df1 = df1.map(self.format_data)
        df1.columns = pd.MultiIndex.from_product([[(f"Transaction Date: {date}; Generated on: {pd.Timestamp.now(tz='Africa/Nairobi').strftime('%d-%m-%Y %H:%M:%S')}")], df1.columns])
        table = build_table(df1, 'green_light', font_size='large', font_family='Open Sans, sans-serif', text_align='left', width='auto', index=True, even_color='darkblue',   even_bg_color='#c3d9ff')
        with open("Compact_Report.html","w+") as file:
            file.write(table)
            #HTML(string=table).write_pdf("Compact_Report.pdf", stylesheets=[CSS(string='@page { size: landscape }')])
            
        df.rename(columns={0: 'Description'}, inplace=True)    
        #plt.style.use('ggplot')
        sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=18)     #Fontsize of the axes title
        plt.rc('axes', labelsize=14)    #Fontsize of the x and y labels
        plt.rc('xtick', labelsize=13)    #Fontsize of the tick labels
        plt.rc('ytick', labelsize=13)    #Fontsize of the tick labels
        plt.rc('legend', fontsize=13)    #Legend fontsize
        plt.rc('font', size=13)
 
        colors1 = sns.color_palette('pastel')
        colors2 = sns.color_palette('deep')
        #colors = sns.color_palette("Set2")
        
        #Calculate the range of values in v
        vh = 5000000
        textstr = 'Created at \nwww.tssfl.com'
        if report_type == 'top':
            #Plot 1
            df_sorted = df.sort_values('Amount',ascending=False)
            #df_sorted['Description'] = df_sorted['Description'].str.wrap(13)  #Wrap at 13 character
            #Top ten highest amount
            plt.figure(figsize=(12, 8), tight_layout=True)
            sns.barplot(x=df_sorted['Amount'].head(10),y=df_sorted['Description'].head(10),data=df_sorted, color="yellowgreen")
            plt.xticks(rotation=90)
            plt.title("Ten Highest Amounts")
            for i, v in enumerate(df_sorted['Amount'].head(10)):
                plt.text(v, i, str(round(v, 4)), color='steelblue', va="center")
                plt.text(v+vh, i, str(i+1), color='black', va="center")
                print("i & v:", i,v)
            #plt.subplots_adjust(right=0.3)    
            #plt.text(0.02, 0.5, textstr, fontsize=14, transform=plt.gcf().transFigure)
            plt.gcf().text(0.7, 0.3, textstr, fontsize=14, color='green') #(0,0) is bottom left, (1,1) is top right
            plt.xlabel("Amount")
            plt.ylabel("Description")
            plt.show()
            plt.close()
            
            #Plot 2
            df_sorted = df.sort_values('Amount',ascending=False)
            df_sorted['Description'] = df_sorted['Description'].str.wrap(13)
            #Top ten highest amounts
            plt.figure(figsize=(12, 10), tight_layout=True)
            sns.barplot(x=df_sorted['Description'].head(10), y=df_sorted['Amount'].head(10),data=df_sorted, color="yellowgreen")
            plt.xticks(rotation=45, ha='center', va='top')
            plt.title("Ten Highest Amounts", y = 1.08)
            xlocs, xlabs = plt.xticks()
            for i, v in enumerate(df_sorted['Amount'].head(10)):
                plt.text(xlocs[i]-0.1, v + 0.05, str(round(v, 4)), color='red', va="center", rotation=45)
            plt.gcf().text(0.7, 0.7, textstr, fontsize=14, color='green')
            plt.xlabel("Description")
            plt.ylabel("Amount")
            plt.show()
            plt.close()
            
        elif report_type == 'low':
            #Lowest amounts
            #Plot 1
            vh = 100
            df_sorted = df.copy().sort_values('Amount',ascending=True)
            vf = 0.05*df_sorted['Amount'].head(10).min()
            df_sorted['Description'] = df_sorted['Description'].str.wrap(13)
            plt.figure(figsize=(12,10), tight_layout=True)
            sns.barplot(x=df_sorted['Amount'].head(10),y=df_sorted['Description'].head(10),data=df_sorted, color="cadetblue")
            plt.xticks(rotation=90)
            plt.title("Ten Lowest Amounts")
            for i, v in enumerate(df_sorted['Amount'].head(10)):
                plt.text(v + vh, i, str(round(v, 4)), color='crimson', va="center") #teal
            plt.gcf().text(0.7, 0.7, textstr, fontsize=14, color='green')
            plt.xlabel("Amount")
            plt.ylabel("Description")
            plt.show()
            plt.close()
            
            #Plot 2
            df_sorted = df.sort_values('Amount',ascending=True)
            df_sorted['Description'] = df_sorted['Description'].str.wrap(13)
            plt.figure(figsize=(12,10), tight_layout=True)
            sns.barplot(x=df_sorted['Description'].head(10), y=df_sorted['Amount'].head(10),data=df_sorted, color="cadetblue")
            plt.xticks(rotation=45, ha='center', va='top')
            plt.title("Ten Lowest Amounts", y = 1.0)
            xlocs, xlabs = plt.xticks()
            for i, v in enumerate(df_sorted['Amount'].head(10)):
                plt.text(xlocs[i]-0.0, v+400, str(round(v, 4)), color='crimson', va="center", rotation=90)
            plt.gcf().text(0.2, 0.7, textstr, fontsize=14, color='green')
            plt.xlabel("Description")
            plt.ylabel("Amount")
            plt.show()
            plt.close()
            
        elif report_type == 'ascend':
            #Full report
            #Plot 1
            df_sorted = df.sort_values('Amount',ascending=True)
            plt.figure(figsize=(12,25), tight_layout=True)
            sns.barplot(x=df_sorted['Amount'],y=df_sorted['Description'],data=df_sorted, color="deepskyblue")
            plt.xticks(rotation=90)
            plt.title("Amounts in TZS")
            for i, v in enumerate(df_sorted['Amount']):
                plt.text(v+10, i, str(round(v, 4)), color='teal', va="center")
                plt.text(v + vh, i, str((i+1)), color='black', va="center")
            plt.gcf().text(0.69, 0.7, textstr, fontsize=14, color='green')
            plt.xlabel("Amount")
            plt.ylabel("Description")
            plt.show()
            plt.close()
            
        elif report_type == 'descend': #Plot 2
            df_sorted = df.sort_values('Amount',ascending=False)
            plt.figure(figsize=(10,25), tight_layout=True)
            sns.barplot(x=df_sorted['Amount'],y=df_sorted['Description'],data=df_sorted, color="deepskyblue")
            plt.xticks(rotation=90)
            plt.title("Amounts in TZS")
            for i, v in enumerate(df_sorted['Amount']):
                plt.text(v+10, i, str(round(v, 4)), color='teal', va="center")
                #plt.text(v+vh, i, str(i+1), color='black', va="center")
                
            #Define the GMT+3 timezone
            gmt_plus_3 = timezone(timedelta(hours=3))

            #Get the current time in GMT+3
            now = datetime.now(gmt_plus_3)
            #Generate the timestamp string
            timestamp = now.strftime("%d-%m-%Y %H:%M:%S")
            textstr = (
                f"Generated at\n" 
                f"TSSFL Technology Stack\n"
                f"www.tssfl.com\n"
                f"on {timestamp}"
            )
            #Add Most Recent Transaction Date
            plt.gcf().text(0.685, 0.35, textstr, fontsize=14, color='green')
            plt.gcf().text(0.685, 0.30, f"Transaction Date:\n{date}", fontsize=14, color='blue') #Adjust position as needed

            plt.xlabel("Amount")
            plt.ylabel("Description")
            plt.show()
            plt.close()
        else:
           pass #Do nothing


    # ══════════════════════════════════════════════════════════════════════════
    # NEXTGAMIS ABS parity layer
    # Mirrors src/engine.py + src/config_manager.py of NEXTGAMIS ABS so the
    # Cloud reports are identical in both figures and presentation.
    # ══════════════════════════════════════════════════════════════════════════

    SELCOM = 'SELCOM'
    MOBILE_POOL = ['AIRTEL MONEY', 'AIRTEL LIPA', 'AZAM PESA', 'HALO LIPA', 'HALO-PESA',
                   'M-PESA', 'T-PESA', 'TIGO LIPA', 'TIGO-PESA', 'VODA LIPA']
    BANK_POOL = ['ABSA BANK', 'ACCESS BANK', 'ACB BANK', 'AMANA BANK', 'AZANIA BANK',
                 'BARODA BANK', 'BOA BANK', 'CBT BANK', 'CHINADASHENG BANK', 'CITI BANK',
                 'CRDB BANK', 'DCB BANK', 'DTB BANK', 'ECO BANK', 'EQUITY BANK', 'EXIM BANK',
                 'FINCA BANK', 'FIRSTHOUSING BANK', 'GT BANK', 'HABIB BANK', 'I&M BANK',
                 'ICB BANK', 'INDIA BANK', 'KCB BANK', 'LETSHEGO BANK', 'MAENDELEO BANK',
                 'MKOMBOZI BANK', 'MUCOBA BANK', 'MWALIMU BANK', 'MWANGA BANK', 'NBC BANK',
                 'NCBA BANK', 'NMB BANK', 'PBZ BANK', 'SMB BANK', 'STANBIC BANK',
                 'STANDARDCHARTERED BANK', 'TADB BANK', 'TCB BANK', 'TIB BANK', 'TMRC BANK',
                 'UBA BANK', 'UCHUMI BANK', 'VISIONFUND BANK', 'YETU BANK']
    AGENCY_POOL = ['MONEYGRAM AGENCY', 'RIA AGENCY', 'WESTERN UNION AGENCY', 'WORLDREMIT AGENCY']

    #Legacy form headers that have since been renamed. Old submissions keep the
    #old wording, so they are mapped onto the current provider name at read time.
    PROVIDER_ALIASES = {'AKIBA BANK': 'ACB BANK'}

    #Fields that are never provider accounts.
    ABS_NON_PROVIDER = {
        'Timestamp', 'Name of Submitter', 'Date of Transaction', 'Date of Submission',
        'MOBILE BUNDLES COMM and SHARES', 'CAPITAL INFUSION', 'SALARIES', 'EXPENDITURES',
        'TRANSFER FEES', 'HARD CASH', 'CREDIT', 'DEBIT', 'CREDIT PAID', 'DEBIT PAID',
        'INCIDENTS', 'TOTAL CASH INFLOW', 'TOTAL CASH OUTFLOW', 'DAY NAME',
        'ACTUAL OPERATING CAPITAL', 'EXPECTED OPERATING CAPITAL',
        'EXCESS', 'LOSS', 'EXCESS/LOSS', 'S/N',
    }

    #Short labels: the summary block sits in the sticky first column, so its
    #width is set by the longest label. "COLUMN TOTALS" forced that column wide
    #enough to crowd out the data on a phone.
    ABS_SUMMARY_LABELS = ('TOTALS', 'AVERAGE', 'MAXIMUM', 'MINIMUM')

    ABS_TEXT_FIELDS = ['MOBILE BUNDLES and SHARES Details', 'CAPITAL INFUSION Details',
                       'TRANSFER FEES Details', 'SALARIES Details', 'EXPENDITURES Details',
                       'CREDIT Details', 'DEBIT Details', 'CREDIT PAID Details',
                       'DEBIT PAID Details',
                       'Transaction Anomalies and Irregularities Details', 'INCIDENTS']

    # ── Column naming ─────────────────────────────────────────────────────────

    @classmethod
    def _abs_is_non_provider(cls, col):
        if col in cls.ABS_NON_PROVIDER or col.endswith(' Details') or col.startswith('TOTAL '):
            return True
        return ' TOTAL' in col or 'GRAND TOTAL' in col

    @classmethod
    def normalise_column(cls, col):
        """Legacy Cloud column name -> NEXTGAMIS ABS naming. Idempotent.

        'AIRTEL MONEY 1 SUPERAGENT COMM' -> 'AIRTEL MONEY SUPERAGENT 1 COMM'
        'AKIBA BANK'                     -> 'ACB BANK 1'
        'SELCOM COMM'                    -> 'SELCOM 1 COMM'

        Old forms emit unindexed provider names and put the index before
        SUPERAGENT; ABS always indexes and puts the index last. Future forms
        follow ABS naming, which this leaves untouched.
        """
        c = re.sub(r'\s+', ' ', str(col)).strip()
        if cls._abs_is_non_provider(c):
            return c

        comm = c.endswith(' COMM')
        if comm:
            c = c[:-5]

        sup, idx = False, '1'
        m = re.match(r'^(?P<base>.+?) (?P<n>\d+) SUPERAGENT$', c)            # legacy Cloud
        if m:
            sup, idx, c = True, m.group('n'), m.group('base')
        else:
            m = re.match(r'^(?P<base>.+?) SUPERAGENT(?: (?P<n>\d+))?$', c)   # ABS
            if m:
                sup, idx, c = True, m.group('n') or '1', m.group('base')
            else:
                m = re.match(r'^(?P<base>.+?) (?P<n>\d+)$', c)
                if m:
                    idx, c = m.group('n'), m.group('base')

        c = cls.PROVIDER_ALIASES.get(c, c)
        out = "{} SUPERAGENT {}".format(c, idx) if sup else "{} {}".format(c, idx)
        return out + " COMM" if comm else out

    def normalise_columns(self, df):
        """Rename every column to ABS naming, and report unrostered providers."""
        before = list(df.columns)
        df = df.rename(columns={c: self.normalise_column(c) for c in df.columns})
        renamed = sum(1 for a, b in zip(before, df.columns) if a != b)
        self._p("Columns normalised to ABS naming",
                sub="{} columns, {} renamed".format(len(df.columns), renamed))
        #ABS calls the submission stamp 'Date of Submission'; the forms emit 'Timestamp'.
        if 'Timestamp' in df.columns and 'Date of Submission' not in df.columns:
            df = df.rename(columns={'Timestamp': 'Date of Submission'})
        pools = set([self.SELCOM] + self.MOBILE_POOL + self.BANK_POOL + self.AGENCY_POOL)
        unknown = set()
        for col in df.columns:
            if self._abs_is_non_provider(col):
                continue
            base = re.sub(r' COMM$', '', col)
            base = re.sub(r' (?:SUPERAGENT )?\d+$', '', base)
            if base not in pools:
                unknown.add(base)
        if unknown:
            self._warn("Providers not in the roster (totals still computed): "
                       + ", ".join(sorted(unknown)))
        return df

    # ── Progress reporting ────────────────────────────────────────────────────

    ABS_VERBOSE = True

    #Continuation marker for a sub-line. Six spaces put it directly under the
    #first letter of the headline above it, so the detail reads as belonging to
    #that step rather than as a step of its own.
    _SUB = '      \u2514 '

    @classmethod
    def _p(cls, msg, done=False, meta=None, sub=None):
        """One progress line per real step, so a long run shows where it is.

        Plain text only. ANSI colour was tried and taken out: SageCell is where
        this runs, and it does not interpret the escapes - it prints them, so a
        green [OK] arrives as a literal [1;32m[OK][0m. Structure carries the
        weight instead: the [OK]/... column, the indent, and the sub-line.

        Output paths are deliberately not printed. They are long enough to wrap,
        and a wrapped line pushes the part worth reading - what was built, over
        what period, how many rows - off the left of the screen. Every report
        method returns its path, so nothing is lost by leaving it out.

        `meta` is the trailing (size, time) note. `sub` is one detail line, or a
        list of them, printed under the headline: a step that both names what it
        did and quantifies it is two facts, and two facts on one line is where
        these started to run long enough to wrap.
        """
        if not cls.ABS_VERBOSE:
            return
        line = ('  [OK] ' if done else '  ... ') + msg
        if meta:
            line += '  ' + meta
        print(line, flush=True)
        for s in ([sub] if isinstance(sub, str) else (sub or [])):
            print(cls._SUB + s, flush=True)

    @classmethod
    def _warn(cls, msg):
        """Same channel as _p, one column narrower."""
        print('  [!] ' + msg, flush=True)

    @classmethod
    def _size_time(cls, path, t0):
        """The trailing note every completion line carries."""
        mb = os.path.getsize(path) / 1048576.0 if os.path.exists(path) else 0
        return "({:.1f} MB, {:.1f}s)".format(mb, time.time() - t0)

    # ── Master column order (ABS get_dynamic_columns, derived from the data) ───

    def abs_master_columns(self, df):
        """Rebuild ABS's g1..g7 column order from whatever the form supplied.

        ABS reads the provider counts from sys_config.json; the Cloud has no
        config, so the accounts actually present in the data are discovered
        instead. Ordering and total names are otherwise identical.
        """
        cols = set(df.columns)
        pools = [self.SELCOM] + self.MOBILE_POOL + self.BANK_POOL + self.AGENCY_POOL

        #Any provider present in the data but absent from the roster still gets
        #its own block, appended after the known ones so nothing is dropped.
        extra = []
        for col in df.columns:
            if self._abs_is_non_provider(col):
                continue
            base = re.sub(r' COMM$', '', col)
            base = re.sub(r' (?:SUPERAGENT )?\d+$', '', base)
            if base not in pools and base not in extra:
                extra.append(base)

        g1 = ['Date of Submission', 'Name of Submitter', 'Date of Transaction']
        g2, g4 = [], []

        for key in pools + extra:
            n_idx = sorted(int(m.group(1)) for m in
                           (re.match(r'^' + re.escape(key) + r' (\d+)$', c) for c in cols) if m)
            s_idx = sorted(int(m.group(1)) for m in
                           (re.match(r'^' + re.escape(key) + r' SUPERAGENT (\d+)$', c) for c in cols) if m)
            if not n_idx and not s_idx:
                continue
            n_cnt, sa_cnt = (max(n_idx) if n_idx else 0), (max(s_idx) if s_idx else 0)
            single_tier = 'LIPA' in key or 'AGENCY' in key

            for n in range(1, n_cnt + 1):
                g2.append("{} {}".format(key, n))
            if single_tier:
                if n_cnt > 1:
                    g2.append("{} GRAND TOTAL FLOAT".format(key))
            else:
                if n_cnt > 1:
                    g2.append("{} NORMAL FLOAT TOTAL".format(key))
                for n in range(1, sa_cnt + 1):
                    g2.append("{} SUPERAGENT {}".format(key, n))
                if sa_cnt > 1:
                    g2.append("{} SUPERAGENT FLOAT TOTAL".format(key))
                if (n_cnt > 0 and sa_cnt > 0) or (n_cnt > 1 or sa_cnt > 1):
                    g2.append("{} GRAND TOTAL FLOAT".format(key))

            for n in range(1, n_cnt + 1):
                g4.append("{} {} COMM".format(key, n))
            if single_tier:
                if n_cnt > 1:
                    g4.append("{} GRAND TOTAL COMM".format(key))
            else:
                if n_cnt > 1:
                    g4.append("{} NORMAL COMM TOTAL".format(key))
                for n in range(1, sa_cnt + 1):
                    g4.append("{} SUPERAGENT {} COMM".format(key, n))
                if sa_cnt > 1:
                    g4.append("{} SUPERAGENT COMM TOTAL".format(key))
                if (n_cnt > 0 and sa_cnt > 0) or (n_cnt > 1 or sa_cnt > 1):
                    g4.append("{} GRAND TOTAL COMM".format(key))

        g3 = ['TOTAL NORMAL MOBILE FLOAT', 'TOTAL SUPERAGENT MOBILE FLOAT',
              'TOTAL LIPA MOBILE FLOAT', 'SELCOM FLOAT TOTAL', 'TOTAL AGENCY FLOAT',
              'TOTAL NORMAL BANK FLOAT', 'TOTAL SUPERAGENT BANK FLOAT',
              'TOTAL MOBILE FLOAT', 'TOTAL BANK FLOAT', 'TOTAL FLOAT']
        g5 = ['MOBILE BUNDLES COMM and SHARES', 'CAPITAL INFUSION', 'SALARIES',
              'EXPENDITURES', 'TRANSFER FEES', 'HARD CASH',
              'TOTAL CASH INFLOW', 'TOTAL CASH OUTFLOW']
        g6 = ['TOTAL NORMAL MOBILE COMMISSION', 'TOTAL LIPA MOBILE COMMISSION',
              'TOTAL SELCOM COMMISSION', 'TOTAL NORMAL BANK COMMISSION',
              'TOTAL AGENCY COMMISSION', 'TOTAL MOBILE COMMISSION',
              'TOTAL BANK COMMISSION', 'TOTAL COMMISSION',
              'ACTUAL OPERATING CAPITAL', 'EXPECTED OPERATING CAPITAL',
              'EXCESS', 'LOSS', 'EXCESS/LOSS',
              'CREDIT', 'DEBIT', 'CREDIT PAID', 'DEBIT PAID']
        g7 = list(self.ABS_TEXT_FIELDS)

        return g1 + g2 + g3 + g4 + g5 + g6 + g7

    @staticmethod
    def _abs_is_text_col(col):
        return 'Details' in col or any(x in col for x in ['INCIDENTS', 'Name', 'Date'])

    # ── Financial engine (port of ABS ReportingEngine.run_calculations) ───────

    def abs_run_calculations(self, raw_df):
        """Provider totals, aggregates and per-date capital reconciliation.

        Identical to NEXTGAMIS ABS: EXPECTED is chained date-to-date across the
        whole business (not per submitter), DEBIT is an outflow and DEBIT PAID an
        inflow, and EXCESS/LOSS is written only onto the last row of each date
        group so a later sum over the group reproduces it exactly.
        """
        master = self.abs_master_columns(raw_df)
        df = raw_df.reindex(columns=master, fill_value=0.0).copy()

        protected = ['Date of Submission', 'Name of Submitter', 'Date of Transaction']
        numeric_cols = [c for c in df.columns
                        if c not in protected and 'Details' not in c and 'INCIDENTS' not in c]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        keywords = [self.SELCOM] + self.MOBILE_POOL + self.BANK_POOL + self.AGENCY_POOL
        for key in keywords:
            p_base = key + ' '
            norm_f = [c for c in df.columns if c.startswith(p_base) and 'SUPERAGENT' not in c and 'COMM' not in c and 'TOTAL' not in c]
            sa_f   = [c for c in df.columns if c.startswith(key + ' SUPERAGENT ') and 'COMM' not in c and 'TOTAL' not in c]
            norm_c = [c for c in df.columns if c.startswith(p_base) and 'SUPERAGENT' not in c and 'COMM' in c and 'TOTAL' not in c]
            sa_c   = [c for c in df.columns if c.startswith(key + ' SUPERAGENT ') and 'COMM' in c and 'TOTAL' not in c]

            if key + ' NORMAL FLOAT TOTAL' in df.columns:
                df[key + ' NORMAL FLOAT TOTAL'] = df[norm_f].sum(axis=1)
            if key + ' SUPERAGENT FLOAT TOTAL' in df.columns:
                df[key + ' SUPERAGENT FLOAT TOTAL'] = df[sa_f].sum(axis=1)
            if key + ' NORMAL COMM TOTAL' in df.columns:
                df[key + ' NORMAL COMM TOTAL'] = df[norm_c].sum(axis=1)
            if key + ' SUPERAGENT COMM TOTAL' in df.columns:
                df[key + ' SUPERAGENT COMM TOTAL'] = df[sa_c].sum(axis=1)
            if key + ' GRAND TOTAL FLOAT' in df.columns:
                df[key + ' GRAND TOTAL FLOAT'] = df[norm_f + sa_f].sum(axis=1)
            if key + ' GRAND TOTAL COMM' in df.columns:
                df[key + ' GRAND TOTAL COMM'] = df[norm_c + sa_c].sum(axis=1)

        def _sum(include_all=None, include_any=None, exclude=None):
            sel = []
            for c in df.columns:
                u = c.upper()
                if include_all and not all(k in u for k in include_all):
                    continue
                if include_any and not any(k in u for k in include_any):
                    continue
                if exclude and any(k in u for k in exclude):
                    continue
                sel.append(c)
            return df[sel].sum(axis=1) if sel else 0.0

        base_ex = ['INFUSION', 'TRANSFER', 'SALARIES', 'EXPENDITURES', 'HARD',
                   'TIMESTAMP', 'SUBMITTER', 'DETAILS', 'INCIDENTS', 'DATE',
                   'CREDIT', 'DEBIT']

        df['TOTAL NORMAL MOBILE FLOAT']     = _sum(exclude=['BANK', 'COMM', 'SUPERAGENT', 'LIPA', 'TOTAL', 'GRAND', 'SELCOM', 'AGENCY'] + base_ex)
        df['TOTAL SUPERAGENT MOBILE FLOAT'] = _sum(include_all=['SUPERAGENT'], exclude=['BANK', 'COMM', 'LIPA', 'TOTAL', 'GRAND', 'SELCOM'])
        df['TOTAL LIPA MOBILE FLOAT']       = _sum(include_all=['LIPA'], exclude=['COMM', 'TOTAL', 'GRAND'])
        df['SELCOM FLOAT TOTAL']            = _sum(include_all=['SELCOM'], exclude=['COMM', 'TOTAL', 'GRAND'])
        df['TOTAL NORMAL BANK FLOAT']       = _sum(include_all=['BANK'], exclude=['SUPERAGENT', 'TOTAL', 'COMM', 'GRAND'])
        df['TOTAL AGENCY FLOAT']            = _sum(include_all=['AGENCY'], exclude=['COMM', 'TOTAL', 'GRAND'])
        df['TOTAL SUPERAGENT BANK FLOAT']   = _sum(include_all=['BANK', 'SUPERAGENT'], exclude=['TOTAL', 'COMM', 'GRAND'])
        df['TOTAL MOBILE FLOAT']            = _sum(exclude=['BANK', 'TOTAL', 'GRAND', 'COMM', 'SELCOM', 'AGENCY', 'TRANSACTION'] + base_ex)
        df['TOTAL BANK FLOAT']              = _sum(include_all=['BANK'], exclude=['TOTAL', 'GRAND', 'COMM'])
        df['TOTAL FLOAT']                   = _sum(exclude=['COMM', 'TOTAL', 'GRAND', 'TRANSACTION'] + base_ex)

        df['TOTAL NORMAL MOBILE COMMISSION']     = _sum(include_all=['COMM'], exclude=['BANK', 'SUPERAGENT', 'LIPA', 'TOTAL', 'GRAND', 'SELCOM', 'AGENCY', 'DETAILS'])
        df['TOTAL SUPERAGENT MOBILE COMMISSION'] = _sum(include_all=['SUPERAGENT', 'COMM'], exclude=['BANK', 'TOTAL', 'GRAND', 'SELCOM', 'AGENCY'])
        df['TOTAL LIPA MOBILE COMMISSION']       = _sum(include_all=['LIPA', 'COMM'], exclude=['TOTAL', 'GRAND'])
        df['TOTAL SELCOM COMMISSION']            = _sum(include_all=['SELCOM', 'COMM'], exclude=['TOTAL', 'GRAND'])
        df['TOTAL NORMAL BANK COMMISSION']       = _sum(include_all=['BANK', 'COMM'], exclude=['SUPERAGENT', 'TOTAL', 'GRAND'])
        df['TOTAL AGENCY COMMISSION']            = _sum(include_all=['AGENCY', 'COMM'], exclude=['TOTAL', 'GRAND'])
        df['TOTAL SUPERAGENT BANK COMMISSION']   = _sum(include_all=['BANK', 'SUPERAGENT', 'COMM'], exclude=['TOTAL', 'GRAND'])
        df['TOTAL MOBILE COMMISSION']            = _sum(include_all=['COMM'], exclude=['BANK', 'TOTAL', 'GRAND', 'SELCOM', 'AGENCY', 'DETAILS'])
        df['TOTAL BANK COMMISSION']              = _sum(include_all=['BANK', 'COMM'], exclude=['TOTAL', 'GRAND'])
        df['TOTAL COMMISSION']                   = _sum(include_all=['COMM'], exclude=['TOTAL', 'GRAND', 'DETAILS'])

        df['ACTUAL OPERATING CAPITAL'] = df['HARD CASH'] + df['TOTAL FLOAT']

        df['_dp'] = self._abs_parse_dates(df['Date of Transaction'])
        df = df.sort_values('_dp', na_position='last').reset_index(drop=True)

        date_info = {}
        for d, grp in df.groupby('_dp', sort=True, dropna=True):
            date_info[d] = {
                'last_idx':     grp.index[-1],
                'total_actual': grp['ACTUAL OPERATING CAPITAL'].sum(),
                'inflow':  (grp['TOTAL COMMISSION'].sum() + grp['CAPITAL INFUSION'].sum()
                            + grp['CREDIT'].sum() + grp['DEBIT PAID'].sum()),
                'outflow': (grp['TRANSFER FEES'].sum() + grp['SALARIES'].sum()
                            + grp['EXPENDITURES'].sum() + grp['CREDIT PAID'].sum()
                            + grp['DEBIT'].sum()),
            }
        dates_sorted = sorted(date_info.keys())

        for col in ['EXPECTED OPERATING CAPITAL', 'EXCESS/LOSS', 'EXCESS', 'LOSS']:
            df[col] = 0.0

        for j, d in enumerate(dates_sorted):
            info = date_info[d]
            if j == 0:
                expected = info['total_actual']
            else:
                prev = date_info[dates_sorted[j - 1]]
                expected = prev['total_actual'] + info['inflow'] - info['outflow']
            el = info['total_actual'] - expected
            last = info['last_idx']
            df.at[last, 'EXPECTED OPERATING CAPITAL'] = expected
            df.at[last, 'EXCESS/LOSS'] = el
            df.at[last, 'EXCESS'] = el if el > 0 else 0.0
            df.at[last, 'LOSS'] = abs(el) if el < 0 else 0.0

        df['TOTAL CASH INFLOW']  = df['TOTAL COMMISSION'] + df['CAPITAL INFUSION'] + df['EXCESS']
        df['TOTAL CASH OUTFLOW'] = df['TRANSFER FEES'] + df['SALARIES'] + df['EXPENDITURES']
        df.drop(columns=['_dp'], inplace=True)
        return df.copy()

    # ── Date consolidation (port of ABS consolidate_by_date) ──────────────────

    def abs_consolidate_by_date(self, df, is_snapshot=False):
        df = df.copy()
        df['Date of Transaction'] = self._abs_parse_dates(
            df['Date of Transaction']).dt.strftime('%d/%m/%Y')

        final_rows = []
        for date, group in df.groupby('Date of Transaction', sort=False):
            for _, row in group.iterrows():
                final_rows.append(row.to_dict())
            if len(group) > 1:
                summary = group.select_dtypes(include=[np.number]).sum().to_dict()
                names = []
                for n in group['Name of Submitter']:
                    if pd.notnull(n):
                        clean = str(n).replace('Mr. ', '').replace('Ms. ', '') \
                                      .replace('Mrs. ', '').replace('Dr. ', '').strip()
                        if clean:
                            names.append(clean.split()[0])
                summary['Name of Submitter'] = 'COMBINED ' + '; '.join(dict.fromkeys(names))
                for col in group.columns:
                    if 'Details' in col or col == 'INCIDENTS':
                        vals = [str(x).strip() for x in group[col]
                                if pd.notnull(x) and str(x).strip().lower()
                                not in ['nan', '', 'none', '0', '0.0']]
                        summary[col] = ' ; '.join(vals) if vals else ''
                summary.update({'Date of Transaction': date,
                                'Date of Submission': datetime.now().strftime('%d/%m/%Y %H:%M:%S')})
                final_rows.append(summary)

        res = pd.DataFrame(final_rows)
        res['S/N'] = range(1, len(res) + 1)
        master = self.abs_master_columns(res)
        head = (['Date of Submission'] if is_snapshot else []) + ['Date of Transaction', 'Name of Submitter']
        other = [c for c in master if c in res.columns and c not in head and c != 'S/N']
        return res[[c for c in (head + other + ['S/N']) if c in res.columns]]

    # ── Summary rows (port of ABS add_summary_rows) ───────────────────────────

    def abs_add_summary_rows(self, df, label_col):
        calc_df = df.copy()
        exclude = list(self.ABS_SUMMARY_LABELS)
        calc_df = calc_df[~calc_df[label_col].isin(exclude)]
        if 'Name of Submitter' in calc_df.columns:
            calc_df = calc_df[~calc_df['Name of Submitter'].astype(str).str.startswith('COMBINED', na=False)]

        protected = ['Reporting Date', 'Name of Submitter', 'Date of Transaction', 'Month Year', 'S/N']
        numeric_cols = [c for c in calc_df.columns
                        if c not in protected and 'Details' not in c and 'INCIDENTS' not in c]
        for col in numeric_cols:
            calc_df[col] = pd.to_numeric(calc_df[col], errors='coerce').fillna(0.0)
        if calc_df.empty:
            return df

        daily_agg = calc_df.groupby('Date of Transaction')[numeric_cols].sum()

        #AVERAGE and MINIMUM count only dates on which a column actually moved:
        #a provider account added mid-history reads 0 for every earlier date, and
        #averaging over all of them divides a real total by the whole period.
        #Filter is != 0 so signed columns keep their loss days.
        #Tolerance rather than != 0: float sums leave residues like 1.86e-09 on
        #dates that are genuinely zero. Money is carried to 2dp, so under half a
        #cent is inactive.
        _ZERO_TOL = 0.005
        active = daily_agg.where(daily_agg.abs() > _ZERO_TOL)
        sums = daily_agg.sum()
        avgs = active.mean().fillna(0.0)
        maxs = daily_agg.max()
        mins = active.min().fillna(0.0)
        mins = mins.where(maxs.abs() > _ZERO_TOL, 0.0)

        allow_sum = ['COMM', 'SALARIES', 'EXPENDITURES', 'INFUSION', 'FEES', 'TOTAL CASH',
                     'EXCESS', 'LOSS', 'CREDIT', 'DEBIT', 'SHARES', 'HARD CASH']
        for col in numeric_cols:
            is_flow = any(kw in col for kw in allow_sum)
            is_balance = 'FLOAT' in col or 'OPERATING CAPITAL' in col
            if not is_flow or is_balance:
                sums[col] = np.nan

        rows = []
        for label, series in zip(self.ABS_SUMMARY_LABELS, (sums, avgs, maxs, mins)):
            row = {c: '' for c in df.columns}
            for col in numeric_cols:
                val = series.get(col)
                row[col] = val if pd.notnull(val) else ''
            row[label_col] = label
            row['S/N'] = ''
            rows.append(row)
        return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    # ── Presentation ──────────────────────────────────────────────────────────

    #House palette (TSSFL Technology Stack)
    HOUSE_NAVY = '#0B2A5B'
    HOUSE_DEEP = '#071E44'
    HOUSE_GOLD = '#F0B429'
    HOUSE_BAND = ['#096EFF', '#F97316', '#10B981', '#F59E0B', '#EC4899', '#15803D']

    #Liberation Sans is metrically identical to Arial, so this renders exactly as
    #NEXTGAMIS ABS on Linux and falls back cleanly elsewhere - with no webfont
    #fetch, which a sandboxed SageMath Cell page cannot rely on.
    ABS_FONT = "'Liberation Sans', Arial, Helvetica, sans-serif"

    def _band_html(self):
        """The brand strip as six laid-out cells rather than six painted stops.

        A gradient stop is a paint-time event with no layout box, so where it
        lands is whatever the rasteriser makes of a fraction of a pixel. Under
        display scaling the boundaries drift and the last stripe absorbs the
        accumulated error - which is exactly what showed on screen: a long
        first stripe and a short last one, six percent apart.

        Six table cells are laid out, not painted. The engine sizes them from
        table-layout:fixed and hands out the remainder a pixel at a time, so no
        stripe is ever more than one pixel off its neighbours, at any width, in
        any renderer. Table layout rather than flex because WeasyPrint lays out
        tables but not flex - flex is what made the strip vanish from every PDF
        to begin with. The gradient stays on as the cells' background, so the
        strip still paints if a renderer ever ignores the cells.
        """
        return ''.join("<span style='background:%s'></span>" % c
                       for c in self.HOUSE_BAND)

    def _abs_css(self):
        """NEXTGAMIS Cloud palette.

        Light theme by design: dark ink on pale grounds, never white-on-dark.
        That is what makes a wide financial table readable, and it removes the
        whole class of "text disappears on hover/selection" faults - the ink is
        dark in every state, so only the ground changes.

        Navy + gold (the TSSFL house colours), deliberately distinct from
        NEXTGAMIS ABS green + crimson. Cloud mirrors ABS in structure and logic,
        not in colour.
        """
        band = self._band_html()   #six cells; .ngc-band below lays them out
        css = """
        <style>
        .ngc-body { font-family: %(font)s; background:#eceff3; margin:0;
                    padding:34px 16px; color:#16243a;
                    -webkit-font-smoothing:antialiased;
                    text-rendering:optimizeLegibility; }
        ::selection     { background:%(gold)s; color:#0B2A5B; }
        ::-moz-selection{ background:%(gold)s; color:#0B2A5B; }

        .ngc-container { max-width:fit-content; margin:0 auto; border-radius:10px;
                         overflow:hidden; background:#fff;
                         box-shadow:0 10px 34px rgba(11,42,91,.20); }
        /* No background on the strip itself. A linear-gradient is painted, and a
           painted box whose left edge lands on a fractional device pixel fills
           that partial pixel at nearly full colour, while the solid background of
           the navy head beside it snaps to the pixel grid - so the strip sat one
           pixel proud on the left. The cells carry solid colours, which snap the
           same way the head does. */
        .ngc-band { display:table; table-layout:fixed; width:100%%; height:7px;
                    border-collapse:collapse; }
        .ngc-band span { display:table-cell; height:7px; }
        /* Each band is closed by a thin gold rule: brand colours on top, navy
           body, gold at the lower boundary - header and footer mirror each other. */
        .ngc-head { background:%(navy)s; color:#fff; padding:20px 28px; text-align:center;
                    font-size:1.2rem; font-weight:700; letter-spacing:.15px;
                    word-wrap:break-word; border-bottom:3px solid %(gold)s; }
        .ngc-head .ngc-sub { display:block; margin-top:7px; font-weight:400;
                             font-size:.92rem; color:#c7d4e6; letter-spacing:.2px; }
        .ngc-head .ngc-co { color:%(gold)s; font-weight:600; }

        .dataframe-div { max-height:78vh; overflow:auto; position:relative; background:#fff; }
        table.ngc { border-collapse:separate; border-spacing:0; width:auto; margin:0;
                    font-size:1.08rem; font-variant-numeric:tabular-nums;
                    font-feature-settings:"tnum" 1; }

        /* Header: white ground, navy ink, gold rule. Borders belong to the cell
           because border-collapse is separate, so they stay put while it is sticky. */
        table.ngc thead th { position:sticky; top:0; z-index:20;
            background:#ffffff; color:#0B2A5B; padding:14px 13px;
            border-bottom:2px solid %(gold)s; border-right:1px solid #b7c7da;
            white-space:normal; word-wrap:break-word; overflow-wrap:anywhere;
            min-width:138px; max-width:210px; vertical-align:middle;
            text-align:center; line-height:1.32; font-weight:700;
            font-size:1.06rem; letter-spacing:.2px; }
        table.ngc thead th:first-child { left:0; z-index:30; background:#fff; }

        /* Body: dark ink throughout, pale alternating grounds. */
        /* Ink is a near-black navy at semibold: on Liberation Sans / Arial weight
           600 maps to the Bold face, so figures read sharply on both grounds.
           The column rule is a definite blue-grey - the old #eef2f7 was invisible
           against white and pale blue alike. */
        table.ngc tbody td { padding:11px 14px; border-bottom:1px solid #d3ddea;
            border-right:1px solid #b7c7da; text-align:right; white-space:nowrap;
            line-height:1.5; color:#0b1c33; font-weight:600; background:#ffffff; }
        table.ngc tbody tr:nth-child(even) td { background:#e9f0fa; }

        /* Sticky index column: pale gold so it separates from the data without
           inverting to white ink. */
        table.ngc tbody td:first-child { position:sticky; left:0; z-index:10;
            background:#fff6e0; color:#0B2A5B; font-weight:700; text-align:left;
            letter-spacing:.2px; border-right:2px solid %(gold)s;
            box-shadow:2px 0 6px rgba(11,42,91,.10); }
        table.ngc tbody tr:nth-child(even) td:first-child { background:#fdeec8; }

        /* Hover only deepens the ground - the ink stays dark, so nothing can
           disappear. (The old rule repainted the navy index cell in a pale tint
           and left its white text invisible.) */
        table.ngc tbody tr:hover td { background:#d6e4f6; color:#08182c; }
        table.ngc tbody tr:hover td:first-child { background:#f7dfa4; color:#0B2A5B; }

        /* Summary block reads as a footer, not as more data. */
        table.ngc tbody tr:nth-last-child(-n+4) td { background:#dfe8f4;
            font-weight:700; color:#08182c; border-top:2px solid #9fb4cd; }
        table.ngc tbody tr:nth-last-child(-n+4) td:first-child { background:#f4e3b4; }
        table.ngc tbody tr:nth-last-child(-n+4):hover td { background:#d2e0f2; }
        table.ngc tbody tr:nth-last-child(-n+4):hover td:first-child { background:#f7dfa4; }

        table.ngc tbody td:last-child { background:#f2f5f9; font-weight:700;
            text-align:center; color:#12386f; }
        table.ngc tbody td:nth-child(2), table.ngc tbody td:nth-child(3) {
            text-align:left; min-width:174px; }
        /* The date column only has to hold dd/mm/yyyy and the short summary
           labels; the header wraps onto two lines rather than setting the width. */
        table.ngc thead th:first-child { min-width:118px; max-width:150px; }
        table.ngc tbody td:first-child { min-width:118px; max-width:150px;
            white-space:nowrap; }

        .ngc-foot { background:%(deep)s; color:#c7d4e6; padding:18px 26px;
                    text-align:center; font-size:.88rem; line-height:1.6;
                    border-bottom:3px solid %(gold)s; }
        .ngc-foot a { color:%(gold)s; text-decoration:none; font-weight:600; }

        /* Mobile: type goes UP, not down. A 100-column financial table scrolls
           sideways on a phone whatever we do, so the win is legibility per cell,
           not fitting more columns on screen. */
        @media screen and (max-width:767px) {
            .ngc-body { padding:10px 4px; }
            .ngc-container { border-radius:6px; }
            .ngc-head { font-size:1.24rem; padding:18px 16px; }
            .ngc-head .ngc-sub { font-size:.98rem; }
            table.ngc { font-size:1.22rem; }
            table.ngc thead th { min-width:158px; padding:15px 13px; font-size:1.16rem;
                                 line-height:1.35; }
            table.ngc thead th:first-child,
            table.ngc tbody td:first-child { min-width:126px; max-width:150px;
                                             padding-left:10px; padding-right:10px; }
            table.ngc tbody td { padding:15px 15px; line-height:1.55; }
            table.ngc tbody td:nth-child(2), table.ngc tbody td:nth-child(3) {
                min-width:190px; }
            .ngc-foot { font-size:.95rem; padding:16px 14px; }
        }
        </style>
        """ % {'font': self.ABS_FONT, 'navy': self.HOUSE_NAVY, 'deep': self.HOUSE_DEEP,
               'gold': self.HOUSE_GOLD}
        return css, band

    def abs_generate_html_report(self, df, title, period_desc, company_name=None,
                                 output_file=None):
        """ABS-styled HTML report: same formatting, colouring and column widths."""
        fdf = df.copy()
        if 'Date of Submission' in fdf.columns:
            fdf.rename(columns={'Date of Submission': 'Reporting Date'}, inplace=True)
        fdf = self.abs_add_summary_rows(fdf, fdf.columns[0])

        def _esc(v):
            return (str(v).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))

        text_like = [c for c in fdf.columns
                     if self._abs_is_text_col(c) or c in ('Reporting Date', 'S/N')]
        for c in fdf.columns:
            if c in text_like:
                #pandas 3 keeps NaN as NaN through astype(str) instead of turning it
                #into the string 'nan', so the regex below never matched it and _esc
                #later rendered a literal "nan" in every empty Details cell.
                #fillna() has to run before the regex for the cell to come out blank.
                fdf[c] = fdf[c].astype(str).fillna('').replace(
                    r'^(nan|None|NaT|0\.0|0|0\.00)$', '', regex=True).map(_esc)
            else:
                num = pd.to_numeric(fdf[c], errors='coerce')
                if not num.isna().all():
                    fdf[c] = num.apply(
                        lambda x: "{:,.2f}".format(float(x)) if pd.notnull(x) else '')

        def get_val(x):
            try:
                return float(str(x).replace(',', ''))
            except (ValueError, TypeError):
                return 0.0

        def style_val(v, color):
            return "<span style='color: {}; font-weight: bold;'>{}</span>".format(color, v)

        AMBER, RED, GREEN = '#E65100', '#C62828', '#1B5E20'
        date_col = 'Date of Transaction' if 'Date of Transaction' in fdf.columns else fdf.columns[0]
        multi_dates = fdf[fdf.duplicated(subset=[date_col], keep=False)][date_col].unique()
        summary_labels = list(self.ABS_SUMMARY_LABELS)

        for i in fdf.index:
            sub = str(fdf.at[i, 'Name of Submitter']) if 'Name of Submitter' in fdf.columns else ''
            tdate = fdf.at[i, date_col]
            is_combined = sub.startswith('COMBINED')
            #On a multi-submission date the per-counter rows carry no reconciliation
            #of their own - only the COMBINED row does - so blank them out.
            if tdate in multi_dates and not is_combined and tdate not in summary_labels:
                for col in ['EXPECTED OPERATING CAPITAL', 'EXCESS', 'LOSS', 'EXCESS/LOSS']:
                    if col in fdf.columns:
                        fdf.at[i, col] = ''
            elif 'ACTUAL OPERATING CAPITAL' in fdf.columns and 'EXPECTED OPERATING CAPITAL' in fdf.columns:
                act, exp = get_val(fdf.at[i, 'ACTUAL OPERATING CAPITAL']), get_val(fdf.at[i, 'EXPECTED OPERATING CAPITAL'])
                color = GREEN if abs(act - exp) < 0.01 else RED
                fdf.at[i, 'ACTUAL OPERATING CAPITAL'] = style_val(fdf.at[i, 'ACTUAL OPERATING CAPITAL'], color)
                fdf.at[i, 'EXPECTED OPERATING CAPITAL'] = style_val(fdf.at[i, 'EXPECTED OPERATING CAPITAL'], color)
                if 'EXCESS' in fdf.columns and get_val(fdf.at[i, 'EXCESS']) > 0.01:
                    fdf.at[i, 'EXCESS'] = style_val(fdf.at[i, 'EXCESS'], AMBER)
                if 'LOSS' in fdf.columns and get_val(fdf.at[i, 'LOSS']) > 0.01:
                    fdf.at[i, 'LOSS'] = style_val(fdf.at[i, 'LOSS'], RED)
                if 'EXCESS/LOSS' in fdf.columns:
                    v = get_val(fdf.at[i, 'EXCESS/LOSS'])
                    if v > 0.01:
                        fdf.at[i, 'EXCESS/LOSS'] = style_val(fdf.at[i, 'EXCESS/LOSS'], AMBER)
                    elif v < -0.01:
                        fdf.at[i, 'EXCESS/LOSS'] = style_val(fdf.at[i, 'EXCESS/LOSS'], RED)

        #Wide tier for free-text columns, moderate tier for submitter names -
        #this is the string-breaking that the old Cloud layout could not do.
        detail_css = ''
        for i, col in enumerate(fdf.columns, start=1):
            if 'Details' in col or 'INCIDENTS' in col:
                detail_css += (".dataframe td:nth-child(%d), .dataframe th:nth-child(%d)"
                               "{min-width:400px !important;max-width:650px !important;"
                               "white-space:normal !important;word-wrap:break-word !important;"
                               "text-align:left !important;line-height:1.35 !important;}\n" % (i, i))
            elif 'Name of Submitter' in col:
                detail_css += (".dataframe td:nth-child(%d), .dataframe th:nth-child(%d)"
                               "{min-width:200px !important;max-width:320px !important;"
                               "white-space:normal !important;word-wrap:break-word !important;"
                               "text-align:left !important;line-height:1.2 !important;}\n" % (i, i))

        #pretty_html_table writes an inline style= on every cell: for a 550x106
        #frame that is ~58,000 attributes, 22x slower and 6x larger than to_html,
        #and the inline styles override the stylesheet. Cell text is escaped above,
        #so escape=False here only lets our own colour spans through.
        df_html = fdf.to_html(index=False, border=0, classes='ngc',
                              escape=False, na_rep='')

        css, band = self._abs_css()
        comp = (company_name or 'NEXTGAMIS CLOUD').upper()
        stamp = datetime.now().strftime('%d/%m/%Y at %H:%M:%S')
        head = ("{} {}<span class='ngc-sub'>Generated for "
                "<span class='ngc-co'>{}</span> on {}</span>").format(title, period_desc, comp, stamp)
        foot = ("NEXTGAMIS Cloud &nbsp;|&nbsp; {} &nbsp;|&nbsp; Generated on {}<br>"
                "Automated Agency Banking Reporting &nbsp;&copy; 2026 "
                "<a href='https://www.tssfl.com'>TSSFL Technology Stack</a>").format(comp, stamp)

        html = ("<html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "{css}<style>{detail}</style></head><body class='ngc-body'>"
                "<div class='ngc-container'>"
                "<div class='ngc-band'>{band}</div>"
                "<div class='ngc-head'>{head}</div>"
                "<div class='dataframe-div'>{table}</div>"
                "<div class='ngc-band'>{band}</div>"
                "<div class='ngc-foot'>{foot}</div>"
                "</div></body></html>").format(css=css, detail=detail_css, band=band,
                                               head=head, table=df_html, foot=foot)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
        return html

    # ── Report 1: Comprehensive Extended ─────────────────────────────────────

    @staticmethod
    def _abs_parse_dates(series):
        """The loaders normalise dates to %m/%d/%Y; ABS-sourced data is day-first.
        Try the pipeline's format first, then fall back to day-first parsing."""
        out = pd.to_datetime(series, format='%m/%d/%Y', errors='coerce')
        if out.isna().all():
            out = pd.to_datetime(series, dayfirst=True, errors='coerce')
        else:
            miss = out.isna() & series.notna()
            if miss.any():
                out.loc[miss] = pd.to_datetime(series[miss], dayfirst=True, errors='coerce')
        return out

    def _abs_filter_period(self, df, period, year=None, start=None, end=None):
        d = df.copy()
        d['Date of Transaction'] = self._abs_parse_dates(d['Date of Transaction'])
        d = d.dropna(subset=['Date of Transaction'])
        if period == 'year':
            d = d[d['Date of Transaction'].dt.year == int(year)]
            desc = "for Year {}".format(year)
        elif period == 'range':
            s = pd.to_datetime(start, dayfirst=True).replace(hour=0, minute=0, second=0)
            e = pd.to_datetime(end, dayfirst=True).replace(hour=23, minute=59, second=59)
            d = d.dropna(subset=['Date of Transaction'])
            d = d[(d['Date of Transaction'] >= s) & (d['Date of Transaction'] <= e)]
            desc = "From {} to {}".format(pd.to_datetime(start, dayfirst=True).strftime('%d-%m-%Y'),
                                          pd.to_datetime(end, dayfirst=True).strftime('%d-%m-%Y'))
        else:
            desc = "All Time"
        return d.sort_values('Date of Transaction').reset_index(drop=True), desc

    def comprehensive_report(self, period='all', year=None, start=None, end=None,
                             company_name=None, output_file=None):
        """Report 1 - Comprehensive Extended (All Columns), ABS-identical.

        period: 'all' | 'year' (with year=) | 'range' (with start=, end=)
        """
        base = self.normalise_columns(self.raw_df.copy())
        filtered, desc = self._abs_filter_period(base, period, year, start, end)
        if filtered.empty:
            self._warn("No transaction data for {}.".format(desc))
            return None
        calc = self.abs_run_calculations(filtered)
        final = self.abs_consolidate_by_date(calc)
        out = output_file or self._abs_outfile('Comprehensive_Report', desc, company_name)
        self.abs_generate_html_report(final, 'Comprehensive Report', desc,
                                      company_name=company_name, output_file=out)
        print("[OK] Report saved -> {}".format(out))
        return out

    # ── Report slicing (port of ABS slice_report) ─────────────────────────────

    def abs_slice_report(self, df, mode):
        fdf = df.copy()
        if mode == 'full_clean':
            drop = [c for c in fdf.columns if 'TOTAL' in c and not any(
                x in c for x in ['TOTAL NORMAL', 'TOTAL SUPERAGENT', 'TOTAL MOBILE',
                                 'TOTAL BANK', 'TOTAL FLOAT', 'TOTAL CASH', 'TOTAL AGENCY'])]
            fdf.drop(columns=drop, inplace=True)
        elif mode == 'monthly_comm':
            fdf['Date of Transaction'] = self._abs_parse_dates(fdf['Date of Transaction'])
            #Group on a real month period, not on the formatted label. Grouping by
            #the "%B %Y" string sorts it alphabetically - April 2025, April 2026,
            #August 2025 ... - instead of earliest month to most recent.
            fdf['Month Period'] = fdf['Date of Transaction'].dt.to_period('M')
            comm_cols = [c for c in fdf.columns if 'COMM' in c and 'TOTAL' not in c
                         and 'Details' not in c]
            prov_totals = [c for c in fdf.columns if any(
                x in c for x in ['NORMAL COMM TOTAL', 'SUPERAGENT COMM TOTAL', 'GRAND TOTAL COMM'])]
            grand = ['TOTAL NORMAL MOBILE COMMISSION', 'TOTAL LIPA MOBILE COMMISSION',
                     'TOTAL SELCOM COMMISSION', 'TOTAL NORMAL BANK COMMISSION',
                     'TOTAL AGENCY COMMISSION', 'TOTAL MOBILE COMMISSION',
                     'TOTAL BANK COMMISSION']
            agg = [c for c in dict.fromkeys(comm_cols + prov_totals + grand) if c in fdf.columns]
            res = fdf.groupby('Month Period')[agg + ['TOTAL COMMISSION']].sum().reset_index()
            res = res.sort_values('Month Period').reset_index(drop=True)
            res['Date of Transaction'] = res['Month Period'].dt.strftime('%B %Y')
            res = res.drop(columns=['Month Period'])
            res = res[['Date of Transaction'] + [c for c in res.columns
                                                 if c != 'Date of Transaction']]
            res['S/N'] = range(1, len(res) + 1)
            return res
        elif mode == 'mini':
            include = ['Date of Transaction', 'Date of Submission', 'Name of Submitter',
                       'TOTAL MOBILE FLOAT', 'TOTAL BANK FLOAT', 'TOTAL FLOAT',
                       'SELCOM FLOAT TOTAL', 'HARD CASH', 'ACTUAL OPERATING CAPITAL',
                       'EXPECTED OPERATING CAPITAL', 'EXCESS/LOSS']
            fdf = fdf[[c for c in fdf.columns if c in include]]
        elif mode == 'compact':
            fdf = fdf[[c for c in fdf.columns if 'TOTAL' not in c]]
        return fdf

    # ── Reports 1-6 ───────────────────────────────────────────────────────────

    ABS_REPORTS = {
        1: ('Comprehensive Report',   None,           'Comprehensive_Report'),
        2: ('Clean Full Report',      'full_clean',   'Clean_Full_Report'),
        3: ('Monthly Commission Report', 'monthly_comm', 'Monthly_Comm_Report'),
        4: ('Mini Totals Dashboard',  'mini',         'Mini_Totals_Dashboard'),
        5: ('Compact Report',         'compact',      'Compact_Accounts_Report'),
    }

    # ── Cached pipeline stages ────────────────────────────────────────────────
    # Normalisation and the calculation pass depend only on the source data and the
    # period, not on which report is drawn from them. Running several reports used
    # to repeat both for every one - the same progress lines over and over, and the
    # same work behind them. Computed once here and reused.

    def _abs_base_frame(self):
        if getattr(self, '_abs_base_cache', None) is None:
            self._abs_base_cache = self.normalise_columns(self.raw_df.copy())
        return self._abs_base_cache

    def _abs_calc_frame(self, period, year=None, start=None, end=None):
        """Returns (calculated_frame_or_None, period_description)."""
        key = (period, str(year), str(start), str(end))
        cache = getattr(self, '_abs_calc_cache', None)
        if cache is None:
            cache = self._abs_calc_cache = {}
        if key not in cache:
            filtered, desc = self._abs_filter_period(
                self._abs_base_frame(), period, year, start, end)
            if filtered.empty:
                cache[key] = (None, desc)
            else:
                self._p("{}: {} submissions across {} date(s)".format(
                            desc, len(filtered),
                            filtered['Date of Transaction'].dt.date.nunique()),
                        sub="Totals and reconciliation computed")
                cache[key] = (self.abs_run_calculations(filtered), desc)
        return cache[key]

    def abs_report(self, number=1, period='all', year=None, start=None, end=None,
                   company_name=None, output_file=None):
        """NEXTGAMIS ABS reports 1-5 over All Time / a Year / a Date Range.

            report.abs_report(1)                                  # Comprehensive, all time
            report.abs_report(3, period='year', year=2025)        # Monthly commission, 2025
            report.abs_report(4, period='range',
                              start='01/11/2024', end='30/11/2024')
        """
        _t0 = time.time()
        if number not in self.ABS_REPORTS:
            self._warn("Report {} is not one of 1-5. Use daily_snapshot_report() "
                       "for 6 or quick_report() for 7.".format(number))
            return None
        title, mode, slug_base = self.ABS_REPORTS[number]

        calc, desc = self._abs_calc_frame(period, year, start, end)
        if calc is None:
            self._warn("No transaction data for {}.".format(desc))
            return None

        if mode == 'monthly_comm':
            final = self.abs_slice_report(calc, mode)
        elif mode:
            final = self.abs_consolidate_by_date(self.abs_slice_report(calc, mode))
        else:
            final = self.abs_consolidate_by_date(calc)

        out = output_file or self._abs_outfile(slug_base, desc, company_name)
        self.abs_generate_html_report(final, title, desc,
                                      company_name=company_name, output_file=out)
        self._p("{} \u00b7 {} \u00b7 {} rows".format(title, desc, len(final)),
                done=True, meta=self._size_time(out, _t0))
        return out

    def _abs_outfile(self, slug_base, desc, company_name):
        """Build the filename: <Company>-<Report>_<Scope>-DD_MM_YYYY-HH_MM_SS.html

        Dashes separate the fields, underscores live inside them, so the
        date and the clock each read as a date and a clock instead of running
        together into one fourteen-digit block.

        Single-date reports otherwise carry their date twice - once as the
        scope, once as the generation stamp. When the two fall on the same day
        the scope is dropped rather than repeated; a scope covering some other
        day is kept, because then it says something the stamp does not.
        """
        now = datetime.now()
        slug = re.sub(r'[^A-Za-z0-9]+', '_', (company_name or 'NEXTGAMIS')).strip('_')[:25]
        scope = desc.replace(' ', '_')
        if re.sub(r'\D', '', scope) == now.strftime('%d%m%Y'):
            scope = ''       #single-date report for today - the stamp already says so
        return "{}-{}{}-{}.html".format(slug, slug_base,
                                        '_' + scope if scope else '',
                                        now.strftime('%d_%m_%Y-%H_%M_%S'))

    # ── Report 6: Daily Snapshot (single date) ────────────────────────────────

    def _abs_pick_date(self, calc, date):
        """Resolve the date a single-date report should cover.

        date=None  -> the latest date that holds data.
        date given -> exactly that date, never substituted.
        Returns (dd/mm/yyyy, rows), or (target, None) when there is nothing to show.
        """
        parsed = self._abs_parse_dates(calc['Date of Transaction'])
        if date is None:
            latest = parsed.max()
            if pd.isna(latest):
                self._warn("No transaction data available.")
                return None, None
            target = latest.strftime('%d/%m/%Y')
        else:
            target = pd.to_datetime(date, dayfirst=True).strftime('%d/%m/%Y')

        day = calc[parsed.dt.strftime('%d/%m/%Y') == target].copy()
        if day.empty:
            self._warn("No transaction data for {}.".format(target))
            return target, None
        return target, day

    def daily_snapshot_report(self, date=None, company_name=None, output_file=None):
        """Report 6 - every submission for one date, plus the COMBINED row.

        With no date it reports on the latest date that holds data.
        """
        _t0 = time.time()
        calc, _ = self._abs_calc_frame('all')
        if calc is None:
            self._warn("No transaction data available.")
            return None
        target, day = self._abs_pick_date(calc, date)
        if day is None:
            return None
        final = self.abs_consolidate_by_date(day, is_snapshot=True)
        desc = "for {}".format(target)
        out = output_file or self._abs_outfile('Daily_Snapshot', target.replace('/', '-'), company_name)
        self.abs_generate_html_report(final, 'Daily Snapshot', desc,
                                      company_name=company_name, output_file=out)
        self._p("Daily Snapshot \u00b7 {} \u00b7 {} rows".format(target, len(final)),
                done=True, meta=self._size_time(out, _t0))
        return out

    # ── Report 7: Quick Report (single date, 4 columns) ───────────────────────

    ABS_SPECIAL_DETAIL = {'MOBILE BUNDLES COMM and SHARES': 'MOBILE BUNDLES and SHARES Details'}

    def quick_report(self, date=None, company_name=None, fmt='html', output_file=None):
        """Report 7 - one date as # | Description | Amount (TZS) | Details.

        With no date it reports on the latest date that holds data.
        Zone A: numeric rows that are non-zero.  Zone B: non-empty free text.
        Zone C: the balance rows, always shown even when zero.
        fmt='pdf' renders A4 portrait via WeasyPrint, falling back to HTML.
        """
        _t0 = time.time()
        calc, _ = self._abs_calc_frame('all')
        if calc is None:
            self._warn("No transaction data available.")
            return None
        #No date given means today, falling back to the most recent date that has
        #data when today's submission is not in yet. An explicitly requested date is
        #never substituted - the caller asked for that day and gets it or nothing.
        target, day = self._abs_pick_date(calc, date)
        if day is None:
            return None

        final = self.abs_consolidate_by_date(day, is_snapshot=True)
        row_data = final.iloc[-1].to_dict()      # COMBINED row when there is one
        all_cols = self.abs_master_columns(final)

        def get_float(col):
            v = row_data.get(col, 0)
            if v is None:
                return 0.0
            try:
                f = float(str(v).replace(',', ''))
                return 0.0 if f != f else f
            except (ValueError, TypeError):
                return 0.0

        def get_text(col):
            v = row_data.get(col, '')
            if v is None:
                return ''
            s = str(v).strip()
            return '' if s.lower() in ['nan', 'none', '', '0', '0.0', '0.00', ';', '; '] else s

        def detail_col_for(col):
            if col in self.ABS_SPECIAL_DETAIL:
                return self.ABS_SPECIAL_DETAIL[col]
            cand = col + ' Details'
            return cand if cand in all_cols else ''

        zone_c = ['ACTUAL OPERATING CAPITAL', 'EXPECTED OPERATING CAPITAL',
                  'EXCESS', 'LOSS', 'EXCESS/LOSS']
        zone_b = ['Transaction Anomalies and Irregularities Details', 'INCIDENTS']
        g1 = {'Date of Submission', 'Name of Submitter', 'Date of Transaction'}
        zone_a = [c for c in all_cols if c not in g1 and c not in zone_c and c not in zone_b
                  and 'Details' not in c and c != 'INCIDENTS']

        rows = []
        for col in zone_a:
            val = get_float(col)
            if val == 0.0:
                continue
            d = detail_col_for(col)
            rows.append(('a', col, val, get_text(d) if d else ''))
        for col in zone_b:
            t = get_text(col)
            if t:
                rows.append(('b', col, None, t))
        for col in zone_c:
            rows.append(('c', col, get_float(col), ''))

        act, exp = get_float('ACTUAL OPERATING CAPITAL'), get_float('EXPECTED OPERATING CAPITAL')
        balanced = abs(act - exp) < 0.01
        AMBER, RED, GREEN = '#E65100', '#C62828', '#1B5E20'

        def amt_cell(val, kind, col):
            if val is None:
                return "<td class='amt'></td>"
            s = "{:,.2f}".format(val)
            if kind == 'c':
                if col in ('ACTUAL OPERATING CAPITAL', 'EXPECTED OPERATING CAPITAL'):
                    c = GREEN if balanced else RED
                    return "<td class='amt'><b style='color:{}'>{}</b></td>".format(c, s)
                if col == 'EXCESS' and val > 0.01:
                    return "<td class='amt'><b style='color:{}'>{}</b></td>".format(AMBER, s)
                if col == 'LOSS' and val > 0.01:
                    return "<td class='amt'><b style='color:{}'>{}</b></td>".format(RED, s)
                if col == 'EXCESS/LOSS':
                    if val > 0.01:
                        return "<td class='amt'><b style='color:{}'>{}</b></td>".format(AMBER, s)
                    if val < -0.01:
                        return "<td class='amt'><b style='color:{}'>{}</b></td>".format(RED, s)
            if val < -0.01:
                return "<td class='amt'><span style='color:{}'>{}</span></td>".format(RED, s)
            return "<td class='amt'>{}</td>".format(s)

        body = []
        for i, (kind, col, val, dtext) in enumerate(rows, 1):
            safe = str(dtext).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            body.append("<tr class='z{}'><td class='sn'>{}</td><td class='desc'>{}</td>"
                        "{}<td class='det'>{}</td></tr>".format(kind, i, col,
                                                                amt_cell(val, kind, col), safe))

        comp = (company_name or 'NEXTGAMIS CLOUD').upper()
        stamp = datetime.now().strftime('%d/%m/%Y at %H:%M:%S')
        band = self._band_html()   #six cells; .qr-band below lays them out
        css = """
        <style>
        @page { size: A4 portrait; margin: 14mm; }
        body { font-family: %(font)s; background:#f4f7f6; padding:28px 14px; }
        .qr-wrap { max-width:980px; margin:0 auto; background:#fff;
                   box-shadow:0 8px 24px rgba(0,0,0,.12); }
        /* No width:100%% - a block box already fills its containing block
           exactly, and stays right whatever padding the wrap later grows. */
        /* No background here either - see .ngc-band. Solid cell colours only, so
           the strip snaps to the same pixel column as the head and the foot. */
        .qr-band { display:table; table-layout:fixed; width:100%%; height:7px;
                   border-collapse:collapse; }
        .qr-band span { display:table-cell; height:7px; }
        /* Both bands close with a thin gold rule, matching the main reports:
           brand colours on top, navy body, gold at the lower boundary. */
        .qr-head { background:%(navy)s; color:#fff; padding:18px 24px; text-align:center;
                   font-size:1.05rem; font-weight:700;
                   border-bottom:3px solid %(gold)s; }
        .qr-head .co { color:%(gold)s; }
        .qr-head .sub { display:block; margin-top:6px; font-weight:400; font-size:.85rem;
                        color:#B9C6DC; }
        table.qr { width:100%%; border-collapse:collapse; font-size:1.08rem;
                   font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1; }
        table.qr th { background:#2E7D32; color:#fff; padding:13px 11px; text-align:left;
                      font-size:1.08rem; letter-spacing:.2px;
                      border-top:3px solid #DC143C; border-bottom:3px solid #DC143C; }
        table.qr td { padding:12px 11px; border:1px solid #dbe2ea; vertical-align:top;
                      line-height:1.45; }
        /* border-collapse:collapse makes an outer cell border straddle the table
           edge - half of it paints outside the box. So the body rows sat a hair
           proud of the bands, the head and the foot, while the header row (which
           carries no side borders) sat a hair inside the body rows: three different
           vertical edges within one card. Dropping the outer side borders gives the
           table a zero-width outer border, so band, head, table and foot all share
           one straight line down each side. */
        table.qr th:first-child, table.qr td:first-child { border-left:0; }
        table.qr th:last-child,  table.qr td:last-child  { border-right:0; }
        td.sn { width:48px; text-align:center; color:#14356b; font-weight:700;
                font-size:1.02rem; background:#eef2f7; }
        td.desc { width:34%%; font-weight:700; color:#0B2A5B; font-size:1.06rem; }
        /* The amount column carried no weight or colour of its own, so the
           figures inherited regular weight while the descriptions beside them
           were bold - the numbers read as the faintest thing on the page.
           This is the one column people actually read, so it is now the
           strongest: largest, boldest, darkest. */
        td.amt { width:20%%; text-align:right; white-space:nowrap;
                 font-size:1.16rem; font-weight:700; color:#0b1c33;
                 letter-spacing:.2px; }
        td.det { width:40%%; white-space:normal; word-wrap:break-word; line-height:1.4;
                 color:#2b3646; font-size:1.02rem; }
        tr.zc td { background:#eef4ff; }
        tr.zb td.det { font-style:italic; }
        tr:hover td { background:#EADDCA; }
        .qr-foot { background:%(deep)s; color:#c7d4e6; padding:16px 24px; text-align:center;
                   font-size:.86rem; line-height:1.6;
                   border-bottom:3px solid %(gold)s; }
        /* The table needs ~471px of content before it stops shrinking, which is
           wider than any phone in portrait - so without a scroller it pushed past
           the card and the strips stopped short of the document width. Scrolling
           it inside the card keeps the card authoritative: strips, head, table and
           foot share one width at every viewport. Same pattern as .dataframe-div
           in the main reports. Print is exempt - WeasyPrint would clip. */
        .qr-scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; }
        @media print { .qr-scroll { overflow-x:visible; } }
        .qr-foot a { color:%(gold)s; text-decoration:none; font-weight:600; }
        </style>
        """ % {'font': self.ABS_FONT, 'navy': self.HOUSE_NAVY,
               'deep': self.HOUSE_DEEP, 'gold': self.HOUSE_GOLD}

        html = ("<html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "{css}</head><body><div class='qr-wrap'>"
                "<div class='qr-band'>{band}</div>"
                "<div class='qr-head'>Quick Report for {date}"
                "<span class='sub'>Generated for <span class='co'>{co}</span> on {stamp}</span></div>"
                "<div class='qr-scroll'>"
                "<table class='qr'><thead><tr><th>#</th><th>Description</th>"
                "<th style='text-align:right'>Amount (TZS)</th><th>Details</th></tr></thead>"
                "<tbody>{body}</tbody></table></div>"
                "<div class='qr-band'>{band}</div>"
                "<div class='qr-foot'>NEXTGAMIS Cloud &nbsp;|&nbsp; {co} &nbsp;|&nbsp; {stamp}<br>"
                "Automated Agency Banking Reporting &copy; 2026 "
                "<a href='https://www.tssfl.com'>TSSFL Technology Stack</a></div>"
                "</div></body></html>").format(css=css, band=band, date=target, co=comp,
                                               stamp=stamp, body=''.join(body))

        stem = output_file or self._abs_outfile('Quick_Report', target.replace('/', '-'),
                                                company_name).replace('.html', '')
        stem = stem[:-5] if stem.endswith('.html') else stem
        if fmt == 'pdf':
            try:
                #Imported here, not at module scope: this is the only place weasyprint
                #is needed, and an ImportError lands in the same except that already
                #handles a broken PDF engine - so an HTML report still works without it.
                from weasyprint import HTML
                HTML(string=html).write_pdf(stem + '.pdf')
                self._p("Quick Report (PDF) \u00b7 {}".format(target), done=True,
                        meta=self._size_time(stem + '.pdf', _t0))
                return stem + '.pdf'
            except Exception as e:
                self._warn("PDF engine unavailable ({}) - saving HTML instead.".format(e))
        with open(stem + '.html', 'w', encoding='utf-8') as f:
            f.write(html)
        self._p("Quick Report (HTML) \u00b7 {}".format(target), done=True,
                meta=self._size_time(stem + '.html', _t0))
        return stem + '.html'

