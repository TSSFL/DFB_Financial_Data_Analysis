import requests
import os
import importlib.util

# --- STEP 1: Dynamically Load the Python Class from GitHub ---
url = "https://raw.githubusercontent.com/TSSFL/DFB_Financial_Data_Analysis/master/dfb_python_class.py"
response = requests.get(url)

if response.status_code == 200:
    with open("dfb_python_class.py", "w") as f:
        f.write(response.text)
    
    # Import the downloaded file as a module
    spec = importlib.util.spec_from_file_location("dfb_module", "dfb_python_class.py")
    dfb_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dfb_module)
    
    # Extract the FinancialReport class from the module
    FinancialReport = dfb_module.FinancialReport
    print("Successfully loaded FinancialReport class from GitHub.")
else:
    raise Exception(f"Failed to download the script. Status code: {response.status_code}")

# --- STEP 2: Initialize and Run Reports ---

try:
    # DropBox Configuration
    db_url = 'https://www.dropbox.com/scl/fi/62rtvedwsuv5myp2m6r7b/data.csv?rlkey=xoi9h5f0phvwhx6yf35nffxn9&st=x8944l0j&dl=1'
    
    # Create Instance
    report = FinancialReport(
        data_source='dropbox', 
        file_path=db_url, 
        file_name='data.csv'
    )

    # Generate Reports
    print("\n--- Generating Commission Report ---")
    report.comm_report()

    print("\n--- Generating Graphs ---")
    report.graphs(None, 'full')

finally:
    # --- STEP 3: Cleanup ---
    # Removes the temporary data file and the downloaded class file
    for temp_file in ['data.csv', 'dfb_python_class.py']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up: {temp_file}")
