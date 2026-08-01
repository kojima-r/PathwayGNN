from pathlib import Path

from pathwaygnn.config import load_config


def test_recursive_config_merge(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("a: 1\nnested:\n  x: 2\n  y: 3\n")
    (tmp_path / "child.yaml").write_text(
        "defaults: [base.yaml]\nnested:\n  y: 4\nvalue: ok\n"
    )
    assert load_config(tmp_path / "child.yaml") == {
        "a": 1,
        "nested": {"x": 2, "y": 4},
        "value": "ok",
    }

