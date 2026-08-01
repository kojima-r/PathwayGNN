import os
import csv
import math
import re
from typing import List

import pandas as pd
from .utils import *
from pandas.core.frame import DataFrame


class MutationsProcessor(object):
    def __init__(self, mutations_file: str) -> None:
        get_sub_sample = False  # NEVER turn on, only for debugging
        print("Loading Mutation Data...")
        columns_to_keep = [
            "Gene name",
            "ID_sample",
            "HGNC ID",
            "Mutation genome position",
            "Mutation Description",
            "Primary site",
            "ID_tumour",
            "Mutation CDS",
        ]

        if get_sub_sample:
            self.mutation_data = pd.read_csv(
                mutations_file, sep="\t", nrows=100000
            )
        else:
            self.mutation_data = pd.read_csv(mutations_file, sep="\t")
        self.mutation_data = self.mutation_data[columns_to_keep]
        self.mutation_data = self.mutation_data.dropna()

        self.index_change = False

    def get_mutation_data(self) -> DataFrame:
        return self.mutation_data

    def add_variant_type(self) -> None:
        mutation_desc_one_hot = pd.get_dummies(
            self.mutation_data["Mutation Description"],
            prefix="Mutation Description",
        )
        self.mutation_data = self.mutation_data.merge(
            mutation_desc_one_hot, left_index=True, right_index=True
        )
        self.remove_duplicates()

    def add_mutation_features(self) -> None:
        if self.index_change:
            print("Warning index has been changed.")
            print("This function should be performed ")
            print("before the index is changed to avoid broadcasting. ")
            print("Move this function higher in the execution order.")
        encoded_mutation_position = extract_genome_encoded_positions(
            self.mutation_data
        )
        self.mutation_data = self.mutation_data.merge(
            encoded_mutation_position,
            how="inner",
            left_on="Encoded Mutation Position",
            right_index=True,
        )
        self.remove_duplicates()

    def filter_by_used_genes(self, cancer_gene_census_file: str) -> None:
        self.index_change = True
        cancer_gene_census = pd.read_csv(cancer_gene_census_file)
        self.mutation_data = self.mutation_data.set_index("Gene name")
        self.mutation_data = self.mutation_data[
            self.mutation_data.index.isin(cancer_gene_census["Gene Symbol"])
        ]
        self.remove_duplicates()

    def filter_by_used_cell_lines(self, used_cell_lines_file: str) -> None:
        self.index_change = True
        used_cell_lines = pd.read_csv(
            used_cell_lines_file, index_col="cosmic_id"
        )
        self.mutation_data = self.mutation_data.set_index("ID_sample")
        self.mutation_data = used_cell_lines.merge(
            self.mutation_data, left_index=True, right_index=True
        )
        self.remove_duplicates()

    def remove_duplicates(self) -> None:
        self.mutation_data = self.mutation_data.drop_duplicates()
