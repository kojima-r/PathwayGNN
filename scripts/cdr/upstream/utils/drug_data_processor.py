import os
import csv
import math
import pandas as pd
from pandas.core.frame import DataFrame


class DrugDataProcessor(object):
    def __init__(
        self, compound_info_data_file: str, used_compounds_data_file: str
    ) -> None:
        print("Loading Drug Data...")
        self.general_drug_data = pd.read_csv(
            compound_info_data_file, index_col="Drug Name"
        )
        self.used_drug_data = pd.read_csv(
            used_compounds_data_file, index_col="name"
        )
        self.used_drug_data = self.used_drug_data.drop(
            columns=["sample_size"], errors="ignore"
        )
        self.used_drug_data = self.used_drug_data.dropna()

    def add_fingerprints(self, fingeprint_data_file: str) -> None:
        print("Loading Fingerprints...")
        fingerprints = pd.read_csv(fingeprint_data_file, dtype={"Name": str})
        if "Name" not in fingerprints.columns:
            raise ValueError("Fingerprint table must contain a Name column")
        fingerprints["Name"] = pd.to_numeric(fingerprints["Name"], errors="raise").astype(int)
        bit_columns = [column for column in fingerprints.columns if column != "Name"]
        if not bit_columns:
            raise ValueError("Fingerprint table contains no descriptor columns")
        fingerprint_strings = (
            fingerprints.set_index("Name")[bit_columns]
            .astype(int).astype(str).agg("".join, axis=1)
        )
        name_to_id = (self.general_drug_data.reset_index()
                      .drop_duplicates("Drug Name")
                      .set_index("Drug Name")["Drug ID"].to_dict())
        used = self.used_drug_data.copy()
        used["Drug ID"] = used.index.map(name_to_id)
        used["fingerprints"] = used["Drug ID"].map(fingerprint_strings)
        self.used_drug_data = used.dropna(subset=["Drug ID", "fingerprints"])

    def combine_drug_data(self, dosage_data_file: str) -> None:
        print("Combining Drug Data...")
        self.merged_drug_data = (self.used_drug_data[["Drug ID", "fingerprints"]]
                                 .drop_duplicates("Drug ID")
                                 .set_index("Drug ID"))
        dosage_data = pd.read_csv(dosage_data_file, index_col="DRUG_ID")
        dosage_data.index = pd.to_numeric(dosage_data.index, errors="raise").astype(int)
        self.merged_drug_data = dosage_data.join(self.merged_drug_data, how="inner")
        self.merged_drug_data = self.merged_drug_data[["COSMIC_ID", "LN_IC50", "fingerprints"]]
        self.merged_drug_data["drug_id"] = self.merged_drug_data.index

    def get_merged_drug_data(self) -> DataFrame:
        return self.merged_drug_data
