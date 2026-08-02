"""Raw-data acquisition for the target-repositioning corpus.

Downloads the public sources that ``pathwaygnn-data tr-build-processed`` reads
and writes them to ``data_tr/raw`` with a ``SHA256SUMS`` manifest:

    python -m scripts.tr.upstream.download_raw_data      # -> data_tr/raw

The module deliberately uses the standard library only, so acquiring the corpus
never depends on the training environment. The build stage that consumes these
files needs h5py (``pip install -e '.[tr-upstream]'``) because the LINCS Level 5
matrix is an HDF5 (GCTX) file.

The two label tables were published by the Kyutech group as
``labo.bio.kyutech.ac.jp/~yamani/target_repositioning/target_disease_data.zip``.
That URL no longer resolves and has no archived copy, so they are fetched from
this repository's release mirror instead. See ``data_tr/README.md``.
"""

DEFAULT_DATA_ROOT = "data_tr"
