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

class FinancialReport:
    def __init__(self, data_source, spreadsheet_key=None, json_file=None, file_path=None, file_name=None, token=None, url=None, asset_index=None):
        self.data_source = data_source
        if self.data_source == 'google_drive':
            self.spreadsheet_key = spreadsheet_key
            self.json_file = json_file
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
        urllib.request.urlretrieve(self.json_file, "tssfl-code-fleet.json")
        gc = gspread.service_account(filename="tssfl-code-fleet.json")
        sh = gc.open_by_key(self.spreadsheet_key)
        worksheet = sh.sheet1
        data = worksheet.get_all_records()
        #Filter out rows with all empty values
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
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction']).dt.strftime('%d/%m/%Y')
        
        return df
        
    def calculations(self, df):
        df = df.copy()
        df.loc['COLUMN TOTALS']= df.iloc[:,np.r_[13:21, 51:96, 98:115, 117:120]].sum(axis=0)
        df.loc['MAXIMUM CREDITS']= df.iloc[0:-1,np.r_[2:120]].max(axis=0)
        df.loc['MINIMUM CREDITS']= df.iloc[0:-2,np.r_[2:120]].min(numeric_only=True, axis=0)
        df.loc['AVERAGE CREDITS']= df.iloc[0:-3,np.r_[2:120]].mean(numeric_only=True, axis=0).round(2)
        
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
        textstr = "Automated Daily Business Financial Reports Computed at TSSFL Technology Stack: www.tssfl.com"
        df.columns = pd.MultiIndex.from_product([[textstr], df.columns])
        df_html = build_table(df, 'green_light', font_size='medium', font_family='Open Sans, sans-serif', text_align='left', width='auto', index=True, even_color='black', even_bg_color='gray')
        style = """
        <style scoped>
        .dataframe-div {
        max-height: 640px;
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
    
    def subset_df(self, df):
        subset_df = df.drop(columns=df.loc[:, 'AIRTEL MONEY':'AZAM PESA'].columns
                    .append(df.loc[:, 'AIRTEL LIPA COMM':'TIGO LIPA COMM'].columns)
                    .append(df.loc[:, 'TRANSFEE: AIRTEL LIPA - AIRTEL MONEY':'TRANSFEE: TIGO LIPA - TIGOPESA'].columns)
                    .append(df.loc[:, 'NMB':'LETSHEGO'].columns)
                    .append(df.loc[:, 'AIRTEL MONEY COMM':'TTCL BUNDLES COMM'].columns)
                    .append(df.loc[:, 'NMB COMM':'LETSHEGO COMM'].columns)
                    .append(df.loc[:, 'ELECTRICITY BILL':'UNFORESEEN EXPENSES'].columns)
                    )
            
        return subset_df
        
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
            
         #Convert 'Date of Transaction' column to datetime
         self.df['Date of Transaction'] = pd.to_datetime(self.df['Date of Transaction'], format='%m/%d/%Y')
         self.df['Grouped Date'] = self.df['Date of Transaction'].dt.date
         self.df['Grouped Date'] = self.df.groupby('Grouped Date')['Grouped Date'].transform('first')
         self.df['Date of Transaction'] = self.df['Grouped Date']
         self.df = self.df.sort_values('Date of Transaction')
         self.df = self.df.reset_index(drop=True) #Maintain date order from low to highest
         self.df = self.df.drop('Grouped Date', axis=1)

         #Total Lipa charges
         self.df.insert(self.df.columns.get_loc('TIGO LIPA COMM') + 1, 'TOTAL LIPA COMMISSION', self.df.loc[:, 'AIRTEL LIPA COMM':'TIGO LIPA COMM'].sum(axis=1))
         
         #Total Transfer fees
         self.df.insert(self.df.columns.get_loc('TRANSFEE: TIGO LIPA - TIGOPESA') + 1, 'TOTAL TRANSFER FEE', self.df.loc[:, 'TRANSFEE: AIRTEL LIPA - AIRTEL MONEY':'TRANSFEE: TIGO LIPA - TIGOPESA'].sum(axis=1))
         
         #Total Mobile commission without Lipa charges
         self.df.insert(self.df.columns.get_loc('TTCL BUNDLES COMM') + 1, 'TOTAL MOBILE COMMISSION', self.df.loc[:, 'AIRTEL MONEY COMM':'TTCL BUNDLES COMM'].sum(numeric_only=True, axis=1))
 
         #Total Bank commission
         self.df.insert(self.df.columns.get_loc('LETSHEGO COMM') + 1, 'TOTAL BANK COMMISSION', self.df.loc[:, 'NMB COMM':'LETSHEGO COMM'].sum(numeric_only=True, axis=1))

         #Total Mobile float
         self.df.insert(self.df.columns.get_loc('AZAM PESA') + 1, 'TOTAL MOBILE FLOAT', self.df.loc[:, 'AIRTEL MONEY':'AZAM PESA'].sum(numeric_only=True, axis=1))
 
         #Total Bank float
         self.df.insert(self.df.columns.get_loc('LETSHEGO') + 1, 'TOTAL BANK FLOAT', self.df.loc[:, 'NMB':'LETSHEGO'].sum(numeric_only=True, axis=1))
 
         #Total Expenditure
         #self.df.insert(self.df.columns.get_loc('RENT') + 1, 'TOTAL EXPENDITURE', self.df.loc[:, 'ELECTRICITY BILL':'RENT'].sum(numeric_only=True, axis=1))
         self.df.insert(self.df.columns.get_loc('RENT') + 1, 'TOTAL EXPENDITURE', self.df.loc[:, 'ELECTRICITY BILL':'RENT'].sum(numeric_only=True, axis=1) - self.df['ADDITIONAL CAPITAL']) 
         #Total float
         self.df.insert(self.df.columns.get_loc('TOTAL BANK COMMISSION') + 1, 'TOTAL FLOAT', self.df.loc[:, ['SELCOM', 'TOTAL MOBILE FLOAT', 'TOTAL BANK FLOAT']].sum(numeric_only=True, axis=1))

         #Total Commission
         self.df.insert(self.df.columns.get_loc('TOTAL BANK COMMISSION') + 1, 'TOTAL COMMISSION', self.df.loc[:, ['TOTAL LIPA COMMISSION', 'TOTAL MOBILE COMMISSION', 'TOTAL BANK COMMISSION']].sum(numeric_only=True, axis=1))
         
         #Total Salaries
         self.df.insert(self.df.columns.get_loc('SALARIES') + 1, 'TOTAL SALARIES', self.df.loc[:, 'OWNER\'S DRAW':'SALARIES'].sum(numeric_only=True, axis=1))

         #Actual Operating Capital
         self.df.insert(self.df.columns.get_loc('HARD CASH') + 1, 'ACTUAL OPERATING CAPITAL', self.df.loc[:, ['TOTAL FLOAT', 'HARD CASH']].sum(numeric_only=True, axis=1))
 
         #Compute loss/excess
         self.df.insert(self.df.columns.get_loc('ACTUAL OPERATING CAPITAL') + 1, 'EXPECTED OPERATING CAPITAL', self.df.loc[1:, ['TOTAL LIPA COMMISSION', 'SELCOM COMM', 'TOTAL MOBILE COMMISSION', 'TOTAL BANK COMMISSION', 'ADDITIONAL CAPITAL']].sum(numeric_only=True, axis=1) - self.df.loc[1:, ['TOTAL TRANSFER FEE', 'TOTAL EXPENDITURE']].sum(numeric_only=True, axis=1) + self.df['ACTUAL OPERATING CAPITAL'].shift(1))
         
         self.df.at[0, 'EXPECTED OPERATING CAPITAL'] = self.df.at[0, 'ACTUAL OPERATING CAPITAL']

         #Expected Operating Capital
         self.df.insert(self.df.columns.get_loc('EXPECTED OPERATING CAPITAL') + 1, 'EXCESS/LOSS', self.df['ACTUAL OPERATING CAPITAL'] - self.df['EXPECTED OPERATING CAPITAL'])

         #Excess
         self.df.insert(self.df.columns.get_loc('EXCESS/LOSS') + 1, 'EXCESS', self.df['EXCESS/LOSS'].apply(lambda x: x if x > 0 else 0))
         #Loss
         self.df.insert(self.df.columns.get_loc('EXCESS') + 1, 'LOSS', self.df['EXCESS/LOSS'].apply(lambda x: abs(x) if x < 0 else 0))

         #Move EXCESS/LOSS column two steps further
         self.df.insert(self.df.columns.get_loc('EXCESS/LOSS') + 2, 'EXCESS/LOSS', self.df.pop('EXCESS/LOSS'))

         #Total cash inflow
         self.df.insert(self.df.columns.get_loc('HARD CASH') + 1, 'TOTAL CASH INFLOW', self.df.loc[:, ['TOTAL COMMISSION', 'ADDITIONAL CAPITAL', 'EXCESS']].sum(numeric_only=True, axis=1))
         #Total cash outflow
         self.df.insert(self.df.columns.get_loc('TOTAL CASH INFLOW') + 1, 'TOTAL CASH OUTFLOW', self.df.loc[:, ['TOTAL TRANSFER FEE', 'TOTAL EXPENDITURE', 'TOTAL SALARIES', 'RENT']].sum(numeric_only=True, axis=1))
         
         #Move HARD CASH column next to the TOTAL FLOAT column
         self.df = self.df.reindex(columns=[col for col in self.df.columns if col != 'HARD CASH'][:self.df.columns.get_loc('TOTAL FLOAT') + 1] + ['HARD CASH'] + [col for col in self.df.columns if col != 'HARD CASH'][self.df.columns.get_loc('TOTAL FLOAT') + 1:])
         
         #Move ADDITIONAL CAPITAL column next to the TOTAL COMMISSION column
         self.df = self.df.reindex(columns=[col for col in self.df.columns if col != 'ADDITIONAL CAPITAL'][:self.df.columns.get_loc('TOTAL COMMISSION') + 1] + ['ADDITIONAL CAPITAL'] + [col for col in self.df.columns if col != 'ADDITIONAL CAPITAL'][self.df.columns.get_loc('TOTAL COMMISSION') + 1:])
         
         #Move TOTAL SALARIES column next to the RENT column
         self.df = self.df.reindex(columns=[col for col in self.df.columns if col != 'TOTAL SALARIES'][:self.df.columns.get_loc('RENT') + 1] + ['TOTAL SALARIES'] + [col for col in self.df.columns if col != 'TOTAL SALARIES'][self.df.columns.get_loc('RENT') + 1:])

         df = self.calculations(self.df)
         df = df.map(self.format_data)
         df = self.date_time(df)
         
         #Full report
         if report_type == 'brief':
            df = self.subset_df(df)
            output_file = 'Brief_DFB_Report.html'
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
        
        #df = self.calculations(df)
        df.loc['COLUMN TOTALS']= df.iloc[:,np.r_[14:22, 52:96, 97:113, 114:116, 118:121]].sum(axis=0)
        df.loc['MAXIMUM CREDITS']= df.iloc[0:-1,np.r_[3:121]].max(axis=0)
        df.loc['MINIMUM CREDITS']= df.iloc[0:-2,np.r_[3:121]].min(numeric_only=True, axis=0)
        df.loc['AVERAGE CREDITS']= df.iloc[0:-3,np.r_[3:121]].mean(numeric_only=True, axis=0).round(2)
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
        
        #df = self.calculations(df) - right shift by +1
        df.loc['COLUMN TOTALS']= df.iloc[:,np.r_[14:22, 52:96, 97:113, 114:116, 118:121]].sum(axis=0)
        df.loc['MAXIMUM CREDITS']= df.iloc[0:-1,np.r_[3:121]].max(axis=0)
        df.loc['MINIMUM CREDITS']= df.iloc[0:-2,np.r_[3:121]].min(numeric_only=True, axis=0)
        df.loc['AVERAGE CREDITS']= df.iloc[0:-3,np.r_[3:121]].mean(numeric_only=True, axis=0).round(2)
        df = df.map(self.format_data)
        df = self.date_time(df)
        
        #Report
        if report_type == 'brief':
            df = self.subset_df(df)
            output_file = 'WeekDays_of_target_Month_and_Year_Brief_DFB_Report.html'
        else:
            output_file = 'WeekDays_of_target_Month_and_Year_Full_DFB_Report.html'

        self.generate_html_table(df, output_file)
    
    def graphs(self, date, report_type):
        #top_ten, lower_ten, full
        df = self.df
        df['Date of Transaction'] = pd.to_datetime(df['Date of Transaction'], format='%m/%d/%Y')
        df = self.date_time(df)
        
        date = date
        #df = df.tail(1) if date is None else df[df['Date of Transaction'] == date].sample(n=1)
        duplicate_count = 0
        if date is None:
            df_selected = df.tail(1)
        else:
            filtered_df = df[df['Date of Transaction'] == date]
            duplicate_count = filtered_df.duplicated(subset='Date of Transaction').sum()
            df_selected = filtered_df.sample(n=1)

        print("Number of duplicate rows found:", duplicate_count)
            
        df = df_selected.T
        df.columns = ['Amount']
        #Reset the index and name the index column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Form of Currency'}, inplace=True)

        df = df[~((df['Amount'] == 0) | (df['Amount'].isna()) | (df['Amount'].apply(lambda x: isinstance(x, str))))]
        #df.loc['TOTAL AMOUNT'] = df.iloc[:, np.r_[1]].sum(axis=0)
        
        table = build_table(df, 'green_light', font_size='medium', font_family='Open Sans, sans-serif', text_align='left', width='auto', index=True, even_color='black',   even_bg_color='gray')
        
        with open("Compact_Report.html","w+") as file:
            file.write(table)
        
        #plt.style.use('ggplot')
        sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=18)     # fontsize of the axes title
        plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
        plt.rc('legend', fontsize=13)    # legend fontsize
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
            #df_sorted['Form of Currency'] = df_sorted['Form of Currency'].str.wrap(13)  #Wrap at 13 character
            #Top ten highest amount
            plt.figure(figsize=(12, 8), tight_layout=True)
            sns.barplot(x=df_sorted['Amount'].head(10),y=df_sorted['Form of Currency'].head(10),data=df_sorted, color="yellowgreen")
            plt.xticks(rotation=90)
            plt.title("Ten Highest Amounts")
            for i, v in enumerate(df_sorted['Amount'].head(10)):
                plt.text(v, i, str(round(v, 4)), color='steelblue', va="center")
                plt.text(v+vh, i, str(i+1), color='black', va="center")
                print("i & v:", i,v)
            #plt.subplots_adjust(right=0.3)    
            #plt.text(0.02, 0.5, textstr, fontsize=14, transform=plt.gcf().transFigure)
            plt.gcf().text(0.7, 0.3, textstr, fontsize=14, color='green') # (0,0) is bottom left, (1,1) is top right
            plt.xlabel("Amount")
            plt.ylabel("Form of Currency")
            plt.show()
            plt.close()
            
            #Plot 2
            df_sorted = df.sort_values('Amount',ascending=False)
            df_sorted['Form of Currency'] = df_sorted['Form of Currency'].str.wrap(13)
            #Top ten highest amounts
            plt.figure(figsize=(12,10), tight_layout=True)
            sns.barplot(x=df_sorted['Form of Currency'].head(10), y=df_sorted['Amount'].head(10),data=df_sorted, color="yellowgreen")
            plt.xticks(rotation=45, ha='center', va='top')
            plt.title("Ten Highest Amounts", y = 1.08)
            xlocs, xlabs = plt.xticks()
            for i, v in enumerate(df_sorted['Amount'].head(10)):
                plt.text(xlocs[i]-0.1, v + 0.05, str(round(v, 4)), color='red', va="center", rotation=45)
            plt.gcf().text(0.7, 0.7, textstr, fontsize=14, color='green')
            plt.xlabel("Form of Currency")
            plt.ylabel("Amount")
            plt.show()
            plt.close()
            
        elif report_type == 'low':
            #Lowest amounts
            #Plot 1
            vh = 100
            df_sorted = df.copy().sort_values('Amount',ascending=True)
            vf = 0.05*df_sorted['Amount'].head(10).min()
            df_sorted['Form of Currency'] = df_sorted['Form of Currency'].str.wrap(13)
            plt.figure(figsize=(12,10), tight_layout=True)
            sns.barplot(x=df_sorted['Amount'].head(10),y=df_sorted['Form of Currency'].head(10),data=df_sorted, color="cadetblue")
            plt.xticks(rotation=90)
            plt.title("Ten Lowest Amounts")
            for i, v in enumerate(df_sorted['Amount'].head(10)):
                plt.text(v + vh, i, str(round(v, 4)), color='crimson', va="center") #teal
            plt.gcf().text(0.7, 0.7, textstr, fontsize=14, color='green')
            plt.xlabel("Amount")
            plt.ylabel("Form of Currency")
            plt.show()
            plt.close()
            
            #Plot 2
            df_sorted = df.sort_values('Amount',ascending=True)
            df_sorted['Form of Currency'] = df_sorted['Form of Currency'].str.wrap(13)
            plt.figure(figsize=(12,10), tight_layout=True)
            sns.barplot(x=df_sorted['Form of Currency'].head(10), y=df_sorted['Amount'].head(10),data=df_sorted, color="cadetblue")
            plt.xticks(rotation=45, ha='center', va='top')
            plt.title("Ten Lowest Amounts", y = 1.0)
            xlocs, xlabs = plt.xticks()
            for i, v in enumerate(df_sorted['Amount'].head(10)):
                plt.text(xlocs[i]-0.0, v+400, str(round(v, 4)), color='crimson', va="center", rotation=90)
            plt.gcf().text(0.2, 0.7, textstr, fontsize=14, color='green')
            plt.xlabel("Form of Currency")
            plt.ylabel("Amount")
            plt.show()
            plt.close()
        else:
            #Full report
            #Plot 1
            df_sorted = df.sort_values('Amount',ascending=True)
            plt.figure(figsize=(12,10), tight_layout=True)
            sns.barplot(x=df_sorted['Amount'],y=df_sorted['Form of Currency'],data=df_sorted, color="deepskyblue")
            plt.xticks(rotation=90)
            plt.title("Amounts in Ascending Order")
            for i, v in enumerate(df_sorted['Amount']):
                plt.text(v+10, i, str(round(v, 4)), color='teal', va="center")
                plt.text(v + vh, i, str((i+1)), color='black', va="center")
            plt.gcf().text(0.69, 0.7, textstr, fontsize=14, color='green')
            plt.xlabel("Amount")
            plt.ylabel("Form of Currency")
            plt.show()
            plt.close()
            
            #Plot 2
            df_sorted = df.sort_values('Amount',ascending=False)
            plt.figure(figsize=(12,10), tight_layout=True)
            sns.barplot(x=df_sorted['Amount'],y=df_sorted['Form of Currency'],data=df_sorted, color="deepskyblue")
            plt.xticks(rotation=90)
            plt.title("Amounts in Descending Order")
            for i, v in enumerate(df_sorted['Amount']):
                plt.text(v+10, i, str(round(v, 4)), color='teal', va="center")
                plt.text(v+vh, i, str(i+1), color='black', va="center")
            plt.gcf().text(0.7, 0.3, textstr, fontsize=14, color='green')
            plt.xlabel("Amount")
            plt.ylabel("Form of Currency")
            plt.show()
            plt.close()

