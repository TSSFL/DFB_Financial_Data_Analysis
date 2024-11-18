import gspread
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from pretty_html_table import build_table
from weasyprint import CSS
from weasyprint import HTML

from koboextractor import KoboExtractor

import regex as re

from datetime import date, datetime, timezone, timedelta

import warnings
warnings.filterwarnings('ignore')

class FinancialReport:
    def __init__(self, data_source, spreadsheet_id =None, service_account_file=None, range_name=None, file_path=None, file_name=None, token=None, url=None, asset_index=None):
        self.data_source = data_source
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
            self.kobo = KoboExtractor(token, url, debug=True)
            self.asset_index = asset_index
            self.asset_uid = None
            self.df = None
            self.df_copy = None
            self.df = self._get_data_from_kobo()
    
        self.df = self._full_report()

    def _get_data_from_google_drive(self):
        print(
        "🌟 Welcome to TSSFL Technology Stack! 🚀\n"
        "Embark on a journey of financial insights as we\n" 
        "process your data with precision and elegance.\n"
        "Please hold on while we craft your comprehensive report! 📊✨\n"
        "")
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
        url = self.file_url
        urllib.request.urlretrieve(url, self.file_name)
        data = pd.read_csv(self.file_name)
        return data
        
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
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%m/%d/%Y %H:%M:%S')
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction']).dt.strftime('%d/%m/%Y')
        
        return df
        
    def calculations(self, df):
        df = df.copy()
        df = df.reset_index(drop=True)  #Reset the existing index
        df.index = df.index + 1       #Add 1 to the reset index
        
        #Replace 0, '', nan with float 0.00
        keywords_include = ['COMM', 'LIPA', 'AGENCY', 'INFUSION', 'TRANSFER', 'SALARIES', 'EXPENDITURES', 'INFLOW', 'OUTFLOW', 'EXCESS', 'LOSS', 'EXCESS/LOSS', 'CREDIT', 'DEBIT']
        keywords_exclude = ['Details', 'INCIDENTS', 'Transaction', 'Submitter', 'Timestamp', 'DAY NAME']
        #Select relevant columns
        relevant_cols = [col for col in df.columns if any(keyword in col for keyword in keywords_include) and not any(keyword in col for keyword in keywords_exclude)]
        for col in relevant_cols:
            try:
                #Attempt direct conversion to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except TypeError:  # Handle cases where pd.to_numeric might fail
                print(f"Column '{col}' could not be converted directly. Attempting string cleaning.")
                #Add more sophisticated string cleaning below if needed
            #Replace NaNs, 0, and empty strings with 0.0
            df[col] = df[col].fillna(0.0).replace('', 0.0)
            
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
            keywords = []  # If you have any keywords to INCLUDE, put them here
            exclusions = ["DETAILS", "INCIDENTS", "TRANSACTION", "SUBMITTER", "TIMESTAMP", "DAY NAME"]

            #Exclude the last row from calculations
            df_calc = df.iloc[:-1, :]
            #Filter columns based on keywords and exclude keywords
            filtered_columns = []
            for col in df_calc.columns:
                if (not keywords or any(keyword in col.upper() for keyword in keywords)) and not any(exclude_keyword in col.upper() for exclude_keyword in exclusions):  # Case-insensitive matching
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
        df.columns = pd.MultiIndex.from_product([[(f"Automated Daily UWAKALA Business Financial Reports Computed at TSSFL Technology Stack - www.tssfl.com on {pd.Timestamp.now(tz='Africa/Nairobi').strftime('%d-%m-%Y %H:%M:%S')} Estern AFrica Time")], df.columns])

        #df.columns = pd.MultiIndex.from_product([[textstr], df.columns])
        df_html = build_table(df, 'green_light', font_size='large', font_family='Open Sans, sans-serif', text_align='left', width='auto', index=True, even_color='darkblue', even_bg_color='#c3d9ff')
        style = """
        <style scoped>
        .dataframe-div {
        max-height: 1240px;
        overflow: auto;
        position: relative;
        }
    
        .dataframe thead th {
        position: -webkit-sticky; /* for Safari */
        position: sticky;
        top: 0;
        background: green;
        color: darkblue;
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
        background: blue;
        color: green;
        vertical-align: top;
        }
        </style>
        """
        df_html = style + '<div class="dataframe-div">' + df_html + "\n</div>"
        
        with open(output_file, "w+") as file:
            file.write(df_html)
            
        return df_html
        
    #This method is for brief report - select a df subset
    def subset_df(self, df):
        #Subset a dataframe
        subset_df = df[[col for col in df.columns if any(kw in col for kw in ["Timestamp", "Submitter", "TOTAL", "AGENCY", "BUNDLES", "ACTUAL", "EXPECTED", "INFUSION", "TRANSFER", "SALARIES", "EXPENDITURES", "HARD", "Transaction", "INCIDENTS"])]]

            
        return subset_df
        
    #Subset a df for Float Summary
    def summary_df(self, df):
        #Remove columns that starts with TOTAL and those contained the keywords shown
        summary_df = df.loc[:, ~df.columns.str.startswith('TOTAL') & ~df.columns.str.contains('Submitter|COMM|LIPA|INFUSION|TRANSFER|SALARIES|EXPENDITURES|HARD|ACTUAL|EXPECTED|EXCESS|LOSS|CREDIT|DEBIT|Details|INCIDENTS', case=False)]
        cols = summary_df.columns.drop(['Timestamp', 'Date of Transaction'])
        sorted_cols = sorted(cols)
        summary_df = summary_df[['Timestamp'] + sorted_cols + ['Date of Transaction']]
        return summary_df
        
    def _full_report(self, report_type = "default_report_type"):
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
                 self.df[col] = self.df[col].astype(str).replace(r'^(0\.0|0|nan)$', '', regex=True)
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
         base_key = ['AIRTEL MONEY ', 'AIRTEL LIPA ', 'VODA LIPA ', 'TIGO-PESA ', 'M-PESA ', 'HALO-PESA ', 'AZAM PESA ', 'CRDB BANK ', 'NMB BANK ', 'NBC BANK ', 'EQUITY BANK ', 'SELCOM ', 'AZANIA BANK ']
 
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
          
         #Replace 0, '', nan with float 0.00
         keywords_include = ['BANK', 'COMM', 'LIPA', 'AGENCY', 'INFUSION', 'TRANSFER', 'SALARIES', 'EXPENDITURES', 'INFLOW', 'OUTFLOW', 'EXCESS', 'LOSS', 'EXCESS/LOSS', 'CREDIT', 'DEBIT']
         keywords_exclude = ['Details', 'INCIDENTS', 'Transaction', 'Submitter', 'Timestamp', 'DAY NAME']
         #Select relevant columns
         relevant_cols = [col for col in self.df.columns if any(keyword in col for keyword in keywords_include) and not any(keyword in col for keyword in keywords_exclude)]
         for col in relevant_cols:
             try:
                 #Attempt direct conversion to numeric, coercing errors to NaN
                 self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
             except TypeError:  # Handle cases where pd.to_numeric might fail
                 print(f"Column '{col}' could not be converted directly. Attempting string cleaning.")
                 #Add more sophisticated string cleaning below if needed
             #Replace NaNs, 0, and empty strings with 0.0
             self.df[col] = self.df[col].fillna(0.0).replace('', 0.0) 
             
         #NORMAL MOBILE FLOAT TOTAL
         #Keywords to exclude
         exclude_keywords = ["BANK", "COMM", "SUPERAGENT", "LIPA", "TOTAL", "SELCOM", 
                    "AGENCY", "INFUSION","TRANSFER", "SALARIES", "EXPENDITURES", "HARD", "Timestamp", "Submitter", "Details", "INCIDENTS", "Transaction", 'CREDIT', 'DEBIT']
                    
         #Columns to sum for "NORMAL MOBILE FLOAT TOTAL"
         normal_mobile_columns = [col for col in self.df.columns if not any(keyword in col for keyword in exclude_keywords)]  
         if normal_mobile_columns:
                self.df['TOTAL NORMAL MOBILE FLOAT'] = self.df[normal_mobile_columns].sum(axis=1) 
         else: 
             print("TOTAL NORMAL MOBILE FLOAT: No columns found")          
          
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
             print("TOTAL SUPERAGENT MOBILE FLOAT: No columns found")
             
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
             print("TOTAL LIPA MOBILE FLOA: No columns found")
       
         #TOTAL SELCOM FLOAT
         selcom_cols = [col for col in self.df.columns if "SELCOM" in col and all(kw not in col for kw in ["COMM", "TOTAL"])]
         if selcom_cols:
             self.df['SELCOM FLOAT TOTAL'] = self.df[selcom_cols].sum(axis=1)  
         else:
             print("SELCOM FLOAT TOTAL: No matching columns")
 
         #NORMAL BANK FLOAT TOTAL
         bank_cols = [col for col in self.df.columns if "BANK" in col and all(kw not in col for kw in ["SUPERAGENT", "TOTAL", "COMM"])]
         if bank_cols:
             self.df['TOTAL NORMAL BANK FLOAT'] = self.df[bank_cols ].sum(axis=1)   
         else:
             print("TOTAL NORMAL BANK FLOAT: No matching columns")
 
         #SUPERAGENT BANK FLOAT TOTAL
         sup_bank_cols = [col for col in self.df.columns if all(kw in col for kw in ["BANK", "SUPERAGENT"]) and all(kw not in col for kw in ["TOTAL", "COMM"])]
         if sup_bank_cols:
             self.df['TOTAL SUPERAGENT BANK FLOAT'] = self.df[sup_bank_cols].sum(axis=1) 
         else:
             print("TOTAL SUPERAGENT BANK FLOAT: No Matching columns")

         #NORMAL MOBILE COMMISSION TOTAL - includes MOBILE BUNDLES COMM and SHARES
         mobile_comm_cols = [col for col in self.df.columns if "COMM" in col and all(kw not in col for kw in ["BANK", "SUPERAGENT", "LIPA", "TOTAL", "SELCOM", "AGENCY", "Details"])]
         if mobile_comm_cols:
             self.df['TOTAL NORMAL MOBILE COMMISSION'] = self.df[mobile_comm_cols].sum(numeric_only=True, axis=1)
         else:
             print("TOTAL NORMAL MOBILE COMMISSION: No Matching columns")
 
         #SUPERAGENT MOBILE COMMISSION TOTAL
         sup_mobile_comm_cols = [col for col in self.df.columns if all(kw in col for kw in ["COMM", "SUPERAGENT"]) and all(kw not in col for kw in ["BANK", "LIPA", "TOTAL", "SELCOM"])]
         if sup_mobile_comm_cols:
             self.df['TOTAL SUPERAGENT MOBILE COMMISSION'] = self.df[sup_mobile_comm_cols ].sum(axis=1)   
         else:
             print("TOTAL NORMAL MOBILE COMMISSION: No Matching columns")

         #LIPA MOBILE COMMISSION TOTAL
         lipa_comm_cols = [col for col in self.df.columns if all(kw in col for kw in ["LIPA", "COMM"]) and all(kw not in col for kw in ["TOTAL"])]
         if lipa_comm_cols:
             self.df['TOTAL LIPA MOBILE COMMISSION'] = self.df[lipa_comm_cols].sum(axis=1)   
         else:
             print("TOTAL LIPA MOBILE COMMISSION: No Matching columns")

         #TOTAL SELCOM COMMISSION
         selcom_comm_cols = [col for col in self.df.columns if all(kw in col for kw in ["SELCOM", "COMM"]) and all(kw not in col for kw in ["TOTAL"])]
         if selcom_comm_cols:
             self.df['TOTAL SELCOM COMMISSION'] = self.df[selcom_comm_cols].sum(axis=1)   
         else:
             print("TOTAL SELCOM COMMISSION: No Matching columns")

         #NORMAL BANK COMMISSION TOTAL
         bank_comm_cols = [col for col in self.df.columns if all(kw in col for kw in ["BANK", "COMM"]) and all(kw not in col for kw in ["SUPERAGENT", "TOTAL"])]
         if bank_comm_cols:
             self.df['TOTAL NORMAL BANK COMMISSION'] = self.df[bank_comm_cols].sum(axis=1)   
         else:
             print("TOTAL NORMAL BANK COMMISSION: No Matching columns")

         #SUPERAGENT BANK COMMISSION TOTAL
         sup_bank_comm_cols = [col for col in self.df.columns if all(kw in col for kw in ["BANK", "SUPERAGENT", "COMM"]) and all(kw not in col for kw in ["TOTAL"])]
         if sup_bank_comm_cols:
             self.df['TOTAL SUPERAGENT BANK COMMISSION'] = self.df[sup_bank_comm_cols].sum(axis=1)
         else:
             print("TOTAL SUPERAGENT BANK COMMISSION: No matching columns") 
    
         #TOTAL MOBILE COMMISSION
         cols = [col for col in self.df.columns if "COMM" in col and all(kw not in col for kw in ["BANK", "TOTAL", "SELCOM", "AGENCY", "Details"])]
         if cols: 
             self.df['TOTAL MOBILE COMMISSION'] = self.df[cols].sum(numeric_only=True, axis=1)
         else: 
             print("TOTAL MOBILE COMMISSION: No Matching columns")
    
         #TOTAL BANK COMMISSION
         cols = [col for col in self.df.columns if all(kw in col for kw in ["BANK", "COMM"]) and all(kw not in col for kw in ["TOTAL"])]
         if cols: 
             self.df['TOTAL BANK COMMISSION'] = self.df[cols].sum(axis=1)
         else: 
             print("TOTAL BANK COMMISSION: No Matching columns")
    
         #TOTAL COMMISSION
         cols = [col for col in self.df.columns if "COMM" in col and all(kw not in col for kw in ["TOTAL"])]
         if cols: 
             self.df['TOTAL COMMISSION'] = self.df[cols].sum(axis=1)
         else: 
             print("TOTAL COMMISSION: No Matching columns")
    
         #TOTAL MOBILE FLOAT
         cols = [col for col in self.df.columns if not any(kw in col for kw in ["BANK", "TOTAL", "COMM", "SELCOM", "AGENCY", 
                                                                  "INFUSION","TRANSFER", "SALARIES", "EXPENDITURES", "HARD", "Timestamp", "Submitter", "Details", "INCIDENTS", "Transaction", 'CREDIT', 'DEBIT'])]
         if cols:
             self.df['TOTAL MOBILE FLOAT'] = self.df[cols].sum(axis=1)
         else:
             print("TOTAL MOBILE FLOAT: No Matching columns")
    
         #TOTAL BANK FLOAT
         cols = [col for col in self.df.columns if "BANK" in col and all(kw not in col for kw in ["TOTAL", "COMM"])]
         if cols: 
             self.df['TOTAL BANK FLOAT'] = self.df[cols].sum(axis=1)
         else: 
             print("TOTAL BANK FLOAT: No Matching columns")
    
         #TOTAL FLOAT
         cols = [col for col in self.df.columns if not any(keyword in col for keyword in ["COMM", "TOTAL", "INFUSION","TRANSFER", "SALARIES", "EXPENDITURES", "HARD", "Timestamp", "Submitter", "Details", "INCIDENTS", "Transaction", 'CREDIT', 'DEBIT'])]
         if cols: 
             self.df['TOTAL FLOAT'] = self.df[cols].sum(axis=1)
         else: 
             print("TOTAL FLOAT: No Matching columns")
    
         #Calculate the sum of the specified columns
         self.df['ACTUAL OPERATING CAPITAL'] = self.df['HARD CASH'] + self.df['TOTAL FLOAT']

         #Compute loss/excess
         self.df.insert(self.df.columns.get_loc('ACTUAL OPERATING CAPITAL') + 1, 'EXPECTED OPERATING CAPITAL', self.df.loc[1:, ['TOTAL COMMISSION', 'CAPITAL INFUSION', 'CREDIT']].sum(numeric_only=True, axis=1) - self.df.loc[1:, ['TRANSFER FEES', 'SALARIES', 'EXPENDITURES', 'DEBIT']].sum(numeric_only=True, axis=1) + self.df['ACTUAL OPERATING CAPITAL'].shift(1))
         
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
         self.df.insert(self.df.columns.get_loc('TOTAL FLOAT') + 1, 'TOTAL CASH INFLOW', self.df.loc[:, ['TOTAL COMMISSION', 'CAPITAL INFUSION', 'EXCESS', 'CREDIT']].sum(numeric_only=True, axis=1))
         #Total cash outflow
         self.df.insert(self.df.columns.get_loc('TOTAL CASH INFLOW') + 1, 'TOTAL CASH OUTFLOW', self.df.loc[:, ['TRANSFER FEES', 'SALARIES','EXPENDITURES', 'DEBIT']].sum(numeric_only=True, axis=1))
         
         #Move and rearrange columns
         cols_to_left = ['Timestamp', 'Name of Submitter']
         cols_to_right = ['CAPITAL INFUSION Details', 'TRANSFER FEES Details', 'SALARIES Details', 'EXPENDITURES Details', 'CREDIT Details', 'DEBIT Details', 'Transaction Anomalies and Irregularities Details', 'INCIDENTS', 'Date of Transaction']
         
         #Check if 'MOBILE BUNDLES and SHARES Details' exists and add it to the beginning of cols_to_right
         if 'MOBILE BUNDLES and SHARES Details' in self.df.columns:
             cols_to_right.insert(0, 'MOBILE BUNDLES and SHARES Details')
         other_cols = [col for col in self.df.columns if col not in cols_to_left + cols_to_right]

         self.df = self.df[cols_to_left + other_cols + cols_to_right]
         
         df = self.calculations(self.df)
         df = df.map(self.format_data)
         df = self.date_time(df)
         
         #Full report
         if report_type == 'brief':
            df = self.subset_df(df)
            output_file = 'Brief_DFB_Report.html'
            
         elif report_type == 'summary':
             df = self.summary_df(df)
             output_file = 'Summary_DFB_Report.html'
         
         else:
            output_file = 'Full_DFB_Report.html'

         self.generate_html_table(df, output_file)
        
         return self.df
        
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
            output_file = f'Brief_DFB_Report_{start_date.strftime("%d%m%Y")}_{end_date.strftime("%d%m%Y")}.html' #Dynamic file name
        else:
            output_file = f'Full_DFB_Report_{start_date.strftime("%d%m%Y")}_{end_date.strftime("%d%m%Y")}.html'  #Dynamic file name

        self.generate_html_table(df, output_file)

    def graphs(self, date, report_type):
        #top_ten, lower_ten, full
        df = self.df
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'], format='%d/%m/%Y')
        most_recent_date = df['Date of Transaction'].max()
        df = self.date_time(df)
        
        date = date
        #df = df.tail(1) if date is None else df[df['Date of Transaction'] == date].sample(n=1)
        duplicate_count = 0
        if date is None:
            df_selected = df.tail(1)
        else:
            filtered_df = df[df['Date of Transaction'] == date]
            if filtered_df.empty:
                #Handle the case where there are no transactions on the specified date.
                #For example, you might set df_selected to an empty DataFrame or raise an exception.
                df_selected = pd.DataFrame()  # Or raise an appropriate exception
                print(f"No transactions found for date: {date}")
            else:
                duplicate_count = filtered_df.duplicated(subset='Date of Transaction').sum()
                df_selected = filtered_df.sample(n=1)

        print("Number of duplicate rows found:", duplicate_count)
        
        #Filter or subset the df to exclude columns names with the given keywords
        df_selected = df_selected[[col for col in df_selected.columns if not any(keyword in col for keyword in ["Timestamp", "Submitter", "Transaction", "Details", "INCIDENTS"])]]
        df = df_selected.T
        df.columns = ['Amount']
        #Reset the index and name the index column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Description'}, inplace=True)
        #keep rows where the 'Amount' column is neither zero, NaN, nor a string
        df = df[~((df['Amount'] == 0) | (df['Amount'].isna()) | (df['Amount'].apply(lambda x: isinstance(x, str))))]
            
        df = df.reset_index(drop=True)  #Reset the existing index
        df.index = df.index + 1       #Add 1 to the reset index
        
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
            plt.gcf().text(0.685, 0.30, f"Transaction Date:\n{most_recent_date.strftime('%d-%m-%Y')}", fontsize=14, color='blue') #Adjust position as needed

            plt.xlabel("Amount")
            plt.ylabel("Description")
            plt.show()
            plt.close()
        else:
            df = df.map(self.format_data)
            table = build_table(df, 'green_light', font_size='large', font_family='Open Sans, sans-serif', text_align='left', width='auto', index=True, even_color='darkblue',   even_bg_color='#c3d9ff')
        
            with open("Compact_Report.html","w+") as file:
                file.write(table)
            #HTML(string=table).write_pdf("Compact_Report.pdf", stylesheets=[CSS(string='@page { size: landscape }')])

