import sys
import types
from collections.abc import Mapping


class _OrderedContext(Mapping):
    """Context table that pandas still accepts as a row indexer.

    signatureanalyzer does `spectra.loc[context96]` with a plain dict, which
    pandas >= 2 rejects outright. A Mapping that is not a dict subclass passes
    that check and is treated as list-like, so `.loc` iterates the keys — the
    legacy behaviour the package was written against. Lookup, membership and
    iteration order all stay identical to the original dict.
    """

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def __getitem__(self, key):
        return self._mapping[key]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)


def _import_signatureanalyzer():
    """Import signatureanalyzer, working around its unmaintained deps.

    signatureanalyzer.utils imports missingpy for its optional imputation
    helpers, and missingpy targets sklearn < 0.22 (sklearn.neighbors.base,
    _check_weights), which no longer exists. Only the spectra code is used
    here, so the imputation dependency is stubbed instead of pinning sklearn.
    """
    try:
        import signatureanalyzer as sa
    except ImportError:
        if "missingpy" not in sys.modules:

            def _missing(*args, **kwargs):
                raise ImportError(
                    "missingpy is unavailable; imputation is not used by "
                    "GraphCDRScan, only signatureanalyzer.spectra is."
                )

            stub = types.ModuleType("missingpy")
            stub.KNNImputer = _missing
            stub.MissForest = _missing
            sys.modules["missingpy"] = stub
        import signatureanalyzer as sa

    for name in ("context96", "context78", "context83", "context1536"):
        context = getattr(sa.spectra, name, None)
        if isinstance(context, dict):
            setattr(sa.spectra, name, _OrderedContext(context))
    return sa


sa = _import_signatureanalyzer()

import pandas as pd
from tqdm import tqdm
import csv
from twobitreader import TwoBitFile
from pandas.core.frame import DataFrame


class MutationalSignals:
    def __init__(self, hg_2bit, mutations_df="", ensemble_to_hgnc=""):
        self.hg_2bit = hg_2bit
        self.hg_2bit_obj = TwoBitFile(hg_2bit)
        self.maf = self.tranform_to_requirements(mutations_df).dropna()
        # The current HGNC export writes the ID as "HGNC:5" while the mutation
        # export uses the bare number, so strip the prefix before mapping.
        hgnc = ensemble_to_hgnc[["HGNC ID", "Approved symbol"]].dropna()
        hgnc_ids = pd.to_numeric(
            hgnc["HGNC ID"].astype(str).str.split("HGNC:", n=1).str[-1],
            errors="coerce",
        )
        hugo_dic = {
            int(hgnc_id): symbol
            for hgnc_id, symbol in zip(hgnc_ids, hgnc["Approved symbol"])
            if pd.notna(hgnc_id)
        }
        self.maf["Hugo_Symbol"] = self.maf["hugo"].map(hugo_dic)
        self.maf = self.maf.drop_duplicates()

    def tranform_to_requirements(self, mutations_df):
        formatted_row = []

        hugo_list = [int(x) for x in mutations_df["HGNC ID"]]
        indeces = [x for x in mutations_df["ID_tumour"]]
        variant_type = [
            self.format_variant_type(x, y)
            for x, y in zip(
                mutations_df["Mutation Description"],
                mutations_df["Mutation genome position"],
            )
        ]
        chromosome_start_position = [
            self.format_chromosome_and_start_position(x)
            for x in mutations_df["Mutation genome position"]
        ]
        allele_reference_allele_seq2 = [
            self.format_alleles(x, y, z)
            for x, y, z in zip(
                mutations_df["Mutation Description"],
                mutations_df["Mutation CDS"],
                mutations_df["Mutation genome position"],
            )
        ]
        with tqdm(total=len(hugo_list)) as progress_bar:
            for hugo, index, var_type, chrom_st_pos, alleles in zip(
                hugo_list,
                indeces,
                variant_type,
                chromosome_start_position,
                allele_reference_allele_seq2,
            ):
                formatted_row.append(
                    [hugo, index, *chrom_st_pos, *alleles, var_type]
                )
                progress_bar.update(1)
        columns = [
            "hugo",
            "Tumor_Sample_Barcode",
            "Chromosome",
            "Start_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
            "Variant_Type",
        ]
        return pd.DataFrame(formatted_row, columns=columns)

    @staticmethod
    def format_chromosome_and_start_position(row):
        chromosome, start_end_pos = row.split(":")
        start_position = start_end_pos.split("-")[0]

        return int(chromosome), int(start_position)

    def get_base(self, chr, start, end):
        if chr == "23":
            chr = "X"
        elif chr == "24":
            chr = "Y"
        elif chr == "MT":
            chr = "M"
        base = self.hg_2bit_obj["chr{}".format(chr)][int(start) - 1 : int(end)]
        return base

    def format_alleles(self, mut_desc, cds, position):
        # ["Mutation CDS"]

        def get_insertion_seq2():
            seq2 = "-"
            if "ins" in cds:
                seq2 = cds.split("ins")[1]
            elif "dup" in cds:
                chromosome, start_end_pos = position.split(":")
                start_position, end_position = start_end_pos.split("-")
                seq2 = self.get_base(chromosome, start_position, end_position)
            return seq2

        def get_deletion_ref():
            chromosome, start_end_pos = position.split(":")
            start_position, end_position = start_end_pos.split("-")
            ref = self.get_base(chromosome, start_position, end_position)
            return ref

        if "insertion" in mut_desc.lower():
            allele_reference = "-"
            allele_seq2 = get_insertion_seq2()
        elif "deletion" in mut_desc.lower():
            allele_reference = "-"
            allele_seq2 = get_deletion_ref()
        else:
            allele_reference = cds[-3]
            allele_seq2 = cds[-1]

        return allele_reference, allele_seq2

    @staticmethod
    def format_variant_type(mut_desc, mut_pos):
        # mut_desc = row["Mutation Description"]
        # mut_pos = row["Mutation genome position"]

        variant_type = ""
        if "insertion" in mut_desc.lower():
            variant_type = "INS"
        elif "deletion" in mut_desc.lower():
            variant_type = "DEL"
        elif "substitution" in mut_desc.lower():
            _, start_end_pos = mut_pos.split(":")
            start, end = start_end_pos.split("-")
            diff = int(end) - int(start) + 1

            if diff == 1:
                variant_type = "SNP"
            elif diff == 2:
                variant_type = "DNP"
            elif diff == 3:
                variant_type = "TNP"
            elif diff > 3:
                variant_type = "ONP"
        elif "nonstop extension" in mut_desc.lower():
            variant_type = "SNP"

        return variant_type

    @classmethod
    def load_from_file(cls, maf_file: str, hg_2bit: str):
        obj = cls.__new__(cls)  # Does not call __init__
        super(MutationalSignals, obj).__init__()
        obj.maf = pd.read_csv(maf_file, sep="\t").reset_index(drop=True)
        obj.hg_2bit = hg_2bit
        return obj

    def spectra_96_base(self) -> DataFrame:
        _, spectra_sbs = sa.spectra.get_spectra_from_maf(
            self.maf, cosmic="cosmic3_exome", hgfile=self.hg_2bit
        )
        return spectra_sbs.T

    def spectra_78_base(self) -> DataFrame:
        _, spectra_dbs = sa.spectra.get_spectra_from_maf(
            self.maf, cosmic="cosmic3_DBS"
        )
        return spectra_dbs.T

    def spectra_83_base(self) -> DataFrame:
        _, spectra_id = sa.spectra.get_spectra_from_maf(
            self.maf, cosmic="cosmic3_ID", hgfile=self.hg_2bit
        )
        return spectra_id.T

    def save_maf(self, path="data_cdr/raw/maf.csv"):
        self.maf.to_csv(
            path, sep="\t", index=False, header=True, quoting=csv.QUOTE_NONE
        )
