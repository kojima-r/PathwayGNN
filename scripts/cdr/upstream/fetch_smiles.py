import pandas as pd
import requests

COMPOUND_INFO_DATA_FILE = "data_cdr/raw/Screened_Compounds.csv"
USED_COMPOUNDS_DATA_FILE = "data_cdr/raw/used_compounds.csv"
OUTPUT_FILE = "data_cdr/raw/drugs.smi"


general_drug_data = pd.read_csv(COMPOUND_INFO_DATA_FILE, index_col="Drug Name")
used_drug_data = pd.read_csv(USED_COMPOUNDS_DATA_FILE, index_col="name")
used_drug_data = used_drug_data.drop(columns=["sample_size"], errors="ignore")
used_drug_data = used_drug_data.dropna()

api_endpoint = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/ConnectivitySMILES/TXT"
target_url = api_endpoint.format(
    ",".join(used_drug_data["pubchem_id"].astype(int).astype(str))
)
response = requests.get(target_url, timeout=120)
response.raise_for_status()

with open(OUTPUT_FILE, "w") as write_handle:
    write_handle.write(response.text)
