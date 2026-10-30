from pathlib import Path

import numpy as np

from faster_cky.tad_external_baselines import (
    parent_domains,
    parse_dedoc2,
    parse_jontads,
    parse_ontad,
    parse_supertad,
)


def test_parent_domains_uses_strict_containment() -> None:
    spans = [(0, 4), (0, 2), (2, 4), (4, 6)]
    assert parent_domains(spans) == [(0, 4)]


def test_external_parsers_normalize_to_half_open_bins(tmp_path: Path) -> None:
    ontad = tmp_path / "x.tad"
    ontad.write_text("1 10 0 0 0\n2 8 1 0 0\n3 4 2 0 0\n")
    assert parse_ontad(ontad, 10) == [(1, 8), (2, 4)]

    jontads = tmp_path / "x.jon"
    jontads.write_text("2, 8, 0.5, 7\n")
    assert parse_jontads(jontads, 10) == [(1, 8)]

    supertad = tmp_path / "x.super"
    supertad.write_text("chr1 2 1 2 chr1 8 7 8\n")
    assert parse_supertad(supertad, 10) == [(1, 8)]

    dedoc = tmp_path / "x.dedoc"
    dedoc.write_text("1\n2 3 4\n")
    assert parse_dedoc2(dedoc, 10) == [(1, 4)]


def test_parsers_drop_invalid_and_one_bin_domains(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_text("1\n0 1\n2 3\n")
    assert parse_dedoc2(path, 3) == [(1, 3)]

