"""GraphCDRScan's raw-data preparation, vendored so PathwayGNN is self-contained.

These modules are a copy of GraphCDRScan's ``scripts/`` tree. Only the data root
is changed: everything reads and writes under ``data_cdr/`` instead of ``data/``,
and the compatibility conversion, the encodings and the output layout are left
exactly as upstream produced them, so ``data_cdr/processed/<FOLDER>`` stays
byte-comparable with a GraphCDRScan checkout.

    python -m scripts.cdr.upstream.download_raw_data      # data_cdr/raw
    python -m scripts.cdr.upstream.prepare_data           # data_cdr/processed

This stage is only needed to rebuild the bundle. It has its own dependency set
(``pip install -e '.[cdr-upstream]'`` plus `pdftotext` and LibreOffice on PATH);
``pathwaygnn-data cdr-prepare`` and everything downstream of it read the bundle
with numpy alone.
"""

DEFAULT_DATA_ROOT = "data_cdr"
