"""Safety checks for ``prepare_output_dir``."""
from __future__ import annotations

import pytest

from cross.infrastructure.run_init import prepare_output_dir


def test_prepare_refuses_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="project root"):
        prepare_output_dir(".", clear=False, project_root=tmp_path)


def test_prepare_refuses_protected_name(tmp_path):
    d = tmp_path / "config"
    d.mkdir()
    with pytest.raises(ValueError, match="protected"):
        prepare_output_dir(d, clear=False, project_root=tmp_path, config_output_dir=None)


def test_prepare_refuses_name_without_out(tmp_path):
    d = tmp_path / "results_only"
    d.mkdir()
    with pytest.raises(ValueError, match="out"):
        prepare_output_dir(d, clear=False, project_root=tmp_path, config_output_dir=None)


def test_prepare_accepts_configured_relative_even_if_name_odd(tmp_path):
    """When ``config_output_dir`` matches resolved path, allow names without ``out``."""
    cfg_rel = "weirddir"
    d = tmp_path / cfg_rel
    d.mkdir()
    with pytest.raises(ValueError, match="out"):
        prepare_output_dir(d, clear=False, project_root=tmp_path, config_output_dir=None)
    out = prepare_output_dir(d, clear=False, project_root=tmp_path, config_output_dir=cfg_rel)
    assert out.resolve() == d.resolve()


def test_prepare_clears_children(tmp_path):
    out = tmp_path / "out_run"
    out.mkdir()
    (out / "stale.csv").write_text("x", encoding="utf-8")
    sub = out / "nested"
    sub.mkdir()
    (sub / "a.json").write_text("{}", encoding="utf-8")
    prepare_output_dir(out, clear=True, project_root=tmp_path, config_output_dir="out_run")
    assert out.is_dir()
    assert not (out / "stale.csv").exists()
    assert not (out / "nested").exists()
