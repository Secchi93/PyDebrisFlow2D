from __future__ import annotations

"""Regression check for Windows legacy-console compatibility.

The complete publication orchestrator redirects child stdout.  On Windows,
older code pages such as cp1252 can be selected by Python for a pipe unless
UTF-8 is forced.  A literal character that cp1252 cannot encode can therefore
abort a simulation before the numerical kernel starts.  This test scans the
literal text contained in print(...) calls and ensures it is cp1252-safe.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _literal_print_text(node: ast.Call) -> str:
    chunks: list[str] = []
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            chunks.append(arg.value)
        elif isinstance(arg, ast.JoinedStr):
            for value in arg.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    chunks.append(value.value)
    return "".join(chunks)


def main() -> int:
    failures: list[str] = []
    for path in ROOT.rglob("*.py"):
        # Ignore caches/build products and archived user copies. Only active
        # source files belong to this regression check.
        ignored = {"__pycache__", "outputs", "cache", "vecchio", "old", "archive", "archived"}
        if any(part.startswith(".") or part.lower() in ignored for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                continue
            text = _literal_print_text(node)
            try:
                text.encode("cp1252")
            except UnicodeEncodeError as exc:
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno}: {exc}")

    if failures:
        print("Windows console compatibility: FAIL")
        for failure in failures:
            print("  " + failure)
        return 1
    print("Windows console compatibility: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
