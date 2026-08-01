import argparse
import csv
import json
import os
import re
import subprocess
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    __package__ = "scripts.cdr.upstream"

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from . import DEFAULT_DATA_ROOT
from .utils.drug_data_processor import DrugDataProcessor
from .utils.graph_generator import GraphGeneration
from .utils.mutations_processor import MutationsProcessor
from .utils.signature_analizer import MutationalSignals
from .utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="Path to the JSON config file",
    default="configs/cdr/upstream.json",
)
parser.add_argument(
    "--data-root",
    help="Directory holding raw/ and processed/",
    default=DEFAULT_DATA_ROOT,
)
args = parser.parse_args()

with open(args.config) as json_file:
    parameters = json.load(json_file)

# Upstream GraphCDRScan hardcodes "data"; PathwayGNN keeps this corpus in
# data_cdr/ so that data_tr/ and data_cancer/ stay separate.
data_root = args.data_root
MAF_CACHE_FILE = "{}/raw/maf.csv".format(data_root)

# Features to Include
VARIANT_TYPE = parameters["VARIANT_TYPE"]
MUTATION_FEATURES = parameters["MUTATION_FEATURES"]
CANCER_TYPE = parameters["CANCER_TYPE"]
SPECTRA96 = parameters["SPECTRA96"]
SPECTRA78 = parameters["SPECTRA78"]
SPECTRA83 = parameters["SPECTRA83"]

# Folder to save the data (it will be created if it does not exist)
FOLDER = parameters["FOLDER"]

os.makedirs("{}/processed/{}".format(data_root, FOLDER), exist_ok=True)

GRAPH_DATA_FILE = "{}/{}".format(data_root, parameters["GRAPH_DATA_FILE"])
MUTATIONS_DATA_FILE = "{}/{}".format(
    data_root, parameters["MUTATIONS_DATA_FILE"]
)
ENSEMBLE_TO_HGNC_DATA_FILE = "{}/{}".format(
    data_root, parameters["ENSEMBLE_TO_HGNC_DATA_FILE"]
)
COMPOUND_INFO_DATA_FILE = "{}/{}".format(
    data_root, parameters["COMPOUND_INFO_DATA_FILE"]
)
DOSAGE_DATA_FILE = "{}/{}".format(data_root, parameters["DOSAGE_DATA_FILE"])
USED_COMPOUNDS_DATA_FILE = "{}/{}".format(
    data_root, parameters["USED_COMPOUNDS_DATA_FILE"]
)
USED_CELL_LINES_DATA_FILE = "{}/{}".format(
    data_root, parameters["USED_CELL_LINES_DATA_FILE"]
)
CANCER_GENE_CENSUS_DATA_FILE = "{}/{}".format(
    data_root, parameters["CANCER_GENE_CENSUS_DATA_FILE"]
)
FINGERPRINT_DATA_FILE = "{}/{}".format(
    data_root, parameters["FINGERPRINT_DATA_FILE"]
)
HG2BIT = "{}/{}".format(data_root, parameters["HG2BIT"])

# PaDEL is no longer required. Generate current RDKit fingerprints on demand.
if not os.path.isfile(FINGERPRINT_DATA_FILE):
    print("RDKit fingerprints not found; generating them from PubChem SMILES...")
    subprocess.run([sys.executable, "-m", "scripts.cdr.upstream.create_fingerprints",
                    "--compounds", COMPOUND_INFO_DATA_FILE,
                    "--used-compounds", USED_COMPOUNDS_DATA_FILE,
                    "--output", FINGERPRINT_DATA_FILE], check=True)

VERTICES_DIC = "{}/processed/{}/{}".format(
    data_root, FOLDER, parameters["VERTICES_DIC"]
)
RELATIONSHIPS_DIC = "{}/processed/{}/{}".format(
    data_root, FOLDER, parameters["RELATIONSHIPS_DIC"]
)
OUTPUT_GRAPH_FILE = "{}/processed/{}/{}".format(
    data_root, FOLDER, parameters["OUTPUT_GRAPH_FILE"]
)
OUTPUT_NODE_FEATURES_FILE = "{}/processed/{}/{}".format(
    data_root, FOLDER, parameters["OUTPUT_NODE_FEATURES_FILE"]
)
OUTPUT_LABELS_FILE = "{}/processed/{}/{}".format(
    data_root, FOLDER, parameters["OUTPUT_LABELS_FILE"]
)
OUTPUT_SAMPLE_FEATURES_FILE = "{}/processed/{}/{}".format(
    data_root, FOLDER, parameters["OUTPUT_SAMPLE_FEATURES_FILE"]
)

# Generate the graph
graph_generator = GraphGeneration(GRAPH_DATA_FILE)
graph_generator.clean_graph()
graph_generator.save_graph(
    ENSEMBLE_TO_HGNC_DATA_FILE,
    VERTICES_DIC,
    RELATIONSHIPS_DIC,
    OUTPUT_GRAPH_FILE,
)

# Process Drug Data
drug_data_processor = DrugDataProcessor(
    COMPOUND_INFO_DATA_FILE, USED_COMPOUNDS_DATA_FILE
)
drug_data_processor.add_fingerprints(FINGERPRINT_DATA_FILE)
drug_data_processor.combine_drug_data(DOSAGE_DATA_FILE)

mutations_processor = MutationsProcessor(MUTATIONS_DATA_FILE)

print("Calculating Mutational Spectra...")

if SPECTRA96 or SPECTRA78 or SPECTRA83:
    ensemble_to_hgnc = pd.read_csv(ENSEMBLE_TO_HGNC_DATA_FILE, sep="\t")
    if os.path.isfile(MAF_CACHE_FILE):
        ms = MutationalSignals.load_from_file(MAF_CACHE_FILE, HG2BIT)
    else:
        ms = MutationalSignals(
            HG2BIT, mutations_processor.get_mutation_data(), ensemble_to_hgnc
        )
        ms.save_maf(MAF_CACHE_FILE)

spectra = []

if SPECTRA96:
    spectra.append(ms.spectra_96_base())
if SPECTRA78:
    spectra.append(ms.spectra_78_base())
if SPECTRA83:
    spectra.append(ms.spectra_83_base())

if VARIANT_TYPE:
    mutations_processor.add_variant_type()

print("Combining Mutation Data...")
mutations_processor.filter_by_used_genes(CANCER_GENE_CENSUS_DATA_FILE)
mutations_processor.filter_by_used_cell_lines(USED_CELL_LINES_DATA_FILE)

if MUTATION_FEATURES:
    mutations_processor.add_mutation_features()

print("Extracting raw node features...")
node_features_df = extract_node_features(
    mutations_processor.get_mutation_data(), graph_generator.get_vertices_dic()
)


def extract_raw_sample_features(
    mutations_data: DataFrame, merged_drug_data: DataFrame
):
    print("Extracting raw sample features...")
    mutations_data = mutations_data[["Primary site", "ID_tumour"]]
    mutations_data = mutations_data[
        ~mutations_data.index.duplicated(keep="first")
    ]
    for spec in spectra:
        mutations_data = mutations_data.merge(
            spec, left_on="ID_tumour", right_index=True
        )

    merged_drug_data = merged_drug_data.set_index("COSMIC_ID")
    merged_data = mutations_data.merge(
        merged_drug_data, left_index=True, right_index=True
    )
    merged_data = merged_data.drop_duplicates()
    sample_features = merged_data.drop(
        columns=["ID_tumour", "drug_id"], errors="ignore"
    )
    return sample_features


sample_features = extract_raw_sample_features(
    mutations_processor.get_mutation_data(),
    drug_data_processor.get_merged_drug_data(),
)

# Select the samples that have enough features
final_id_sample = sample_features.index.intersection(node_features_df.index)
node_features_df = node_features_df.loc[final_id_sample].sort_index()
sample_features = sample_features.loc[final_id_sample].sort_index()

primary_site_dic = {
    v: k for k, v in enumerate(sample_features["Primary site"].unique())
}
sample_features["cancer_type"] = sample_features["Primary site"].map(
    primary_site_dic
)

# Encode cancer type
primary_site_one_hot = None
if CANCER_TYPE:
    mlb = MultiLabelBinarizer()
    primary_site_desc = pd.DataFrame(
        mlb.fit_transform([sample_features["Primary site"]]),
        columns=mlb.classes_,
    )
    primary_site_one_hot = pd.get_dummies(
        sample_features["Primary site"], prefix="Primary site", dtype=np.bool_
    )
    primary_site_one_hot = primary_site_one_hot[
        ~primary_site_one_hot.index.duplicated(keep="first")
    ]
    sample_features = sample_features.merge(
        primary_site_one_hot, left_index=True, right_index=True
    )

# Generate samples index
sample_features["COSMIC_ID"] = sample_features.index
sample_features = sample_features.reset_index(drop=True)
sample_to_cosmic = sample_features["COSMIC_ID"]

print("Generating Node Features...")
node_features_df = node_features_df.merge(
    sample_to_cosmic, right_on="COSMIC_ID", left_index=True
)
node_features_df = node_features_df.drop(columns=["COSMIC_ID"])
node_features_df = node_features_df.sort_index()
cols = node_features_df.columns
node_features_df = node_features_df[[*cols[-1:], *cols[:-1]]]
node_features_df.to_csv(
    OUTPUT_NODE_FEATURES_FILE,
    sep="\t",
    index=True,
    header=False,
    quoting=csv.QUOTE_NONE,
)

print("Generating Labels...")
labels = sample_features[["LN_IC50"]]
labels["IC50"] = np.exp(labels["LN_IC50"])
labels.to_csv(
    OUTPUT_LABELS_FILE,
    sep="\t",
    index=True,
    header=False,
    quoting=csv.QUOTE_NONE,
)

print("Generating Sample Features...")


def save_sample_features(
    file_path: str, data: DataFrame, sep: str = "\t"
) -> None:
    columns = data.columns
    with tqdm(total=len(data)) as progress_bar:
        with open(file_path, "w") as write_handle:

            def process_data(row):
                write_handle.write("{}{}".format(row["sample_id"], sep))
                write_handle.write("{}{}".format(row["cancer_type"], sep))
                [
                    write_handle.write("{}{}".format(int(row[x]), sep))
                    for x in row.index
                    if x not in ["sample_id", "cancer_type", "fingerprints"]
                ]
                write_handle.write(row["fingerprints"].replace("", sep)[1:-1])

                write_handle.write("\n")
                progress_bar.update(1)

            for index, row in data.iterrows():
                process_data(row)


# Remove last not needed features
sample_features = sample_features.drop(
    columns=["Primary site", "COSMIC_ID", "LN_IC50"], errors="ignore"
)

sample_features["sample_id"] = sample_features.index
print("Saving Data...")
save_sample_features(OUTPUT_SAMPLE_FEATURES_FILE, sample_features)
