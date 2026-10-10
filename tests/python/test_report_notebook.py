"""NB-2: the notebook pulsatrix's Report writes is valid nbformat 4.5.

tests/report_test.cpp writes the sample report and compares it byte for byte with
tests/fixtures/report/sample.ipynb, so validating that file validates Report::to_ipynb().
Skipped where nbformat isn't installed.
"""
import os

import pytest

nbformat = pytest.importorskip("nbformat")

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "report", "sample.ipynb")


def test_the_sample_report_is_valid_nbformat_4_5():
    nb = nbformat.read(SAMPLE, as_version=nbformat.NO_CONVERT)
    assert (nb.nbformat, nb.nbformat_minor) == (4, 5)
    nbformat.validate(nb)


def test_outputs_are_in_their_mime_types():
    nb = nbformat.read(SAMPLE, as_version=nbformat.NO_CONVERT)
    data = [o["data"] for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "display_data"]
    kinds = {k for d in data for k in d}
    assert {"text/plain", "text/html", "image/svg+xml", "image/png", "application/vnd.vegalite.v5+json"} <= kinds
    for d in data:
        if "application/vnd.vegalite.v5+json" in d:
            assert isinstance(d["application/vnd.vegalite.v5+json"], dict)
