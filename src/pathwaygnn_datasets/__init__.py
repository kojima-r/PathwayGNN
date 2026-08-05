"""Dataset-specific preprocessing and reporting for PathwayGNN.

Each subpackage owns one corpus and writes the generic dataset format defined in
``pathwaygnn.data.format``. The training engine never imports from here.
"""

__all__ = ["cancer", "cdr", "sample", "tr"]
