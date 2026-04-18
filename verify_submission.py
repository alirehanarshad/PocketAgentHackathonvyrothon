from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

BANNED_IMPORTS = {"requests", "urllib", "http", "socket"}


def check_model_file() -> list[str]:
    errors: list[str] = []
    path = Path("model.gguf")
    if not path.exists():
        return ["model.gguf is missing"]
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"model.gguf size: {size_mb:.2f} MB")
    if size_mb > 500:
        errors.append("model.gguf exceeds 500 MB gate")
    return errors


def check_network_imports() -> list[str]:
    src = Path("inference.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                alias.name for alias in node.names if alias.name.split(".")[0] in BANNED_IMPORTS
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in BANNED_IMPORTS:
                found.append(node.module)
    if found:
        return [f"banned imports found in inference.py: {sorted(set(found))}"]
    print("No banned network imports found in inference.py")
    return []


def check_run_signature() -> list[str]:
    src = Path("inference.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    run_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_node = node
            break
    if run_node is None:
        return ["inference.py does not expose run"]

    params = [arg.arg for arg in run_node.args.args]
    if params != ["prompt", "history"]:
        return [f"run signature mismatch: ({', '.join(params)})"]
    print("run signature OK: run(prompt, history)")
    return []


def check_demo_import() -> list[str]:
    spec = importlib.util.find_spec("chatbot")
    if spec is None:
        return ["chatbot.py could not be imported as a module"]
    print("chatbot module import path found")
    return []


def main() -> int:
    checks = [
        check_model_file,
        check_network_imports,
        check_run_signature,
        check_demo_import,
    ]

    errors: list[str] = []
    for check in checks:
        errors.extend(check())

    if errors:
        print("\nVerification failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("\nVerification passed for all local checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
