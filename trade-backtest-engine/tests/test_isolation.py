"""futures/ and forex/ must never import each other; core/ imports neither."""

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "src" / "quant_lab"


def _imports(package: str):
    for path in (PKG / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    yield path, a.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield path, node.module


@pytest.mark.parametrize("package, forbidden", [
    ("futures", ("quant_lab.forex",)),
    ("forex", ("quant_lab.futures",)),
    ("core", ("quant_lab.futures", "quant_lab.forex")),
])
def test_no_cross_engine_imports(package, forbidden):
    bad = [(p.name, m) for p, m in _imports(package) if m.startswith(forbidden)]
    assert not bad, f"{package} imports {bad}"
