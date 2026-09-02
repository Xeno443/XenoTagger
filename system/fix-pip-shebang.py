"""Reapply WinPython's relocatable-shebang patch to pip's vendored
pip\\_vendor\\distlib\\scripts.py, if a pip upgrade replaced it with the
official (non-relocatable) version.

Background: WinPython's bundled pip has a one-line patch making
ScriptMaker._get_shebang()'s fallback branch write a bare "python.exe"
shebang (resolved via PATH) instead of the full sys.executable path.
Official PyPI releases of pip do not have this patch - and pip's own
"pip install --upgrade pip" is a no-op as long as WinPython's bundled
version string still looks current, so the patched behavior can persist
indefinitely, then vanish without warning the moment PyPI publishes a
newer pip. See CLAUDE.md's "Wrapper-path self-heal" section for the
full story.

Rather than a blind text replacement (which could silently corrupt the
file if pip's internals shift in some unrelated way), this parses the
file and only acts if the exact assignment it expects - inside
ScriptMaker._get_shebang()'s `elif not sysconfig.is_python_build():`
branch - is confidently recognized as either the patched or unpatched
form. Anything else is left untouched with a warning; this is meant to
restore one known, specific patch, not to guess.

Usage: fix-pip-shebang.py
(operates on whichever pip is importable by the interpreter running
this script - run it with the same python.exe you want checked.)
"""
import ast
from pathlib import Path

UNPATCHED_SRC = "get_executable()"
PATCHED_SRC = "os.path.join(os.path.basename(get_executable()))"


def find_target_assign(tree: ast.Module):
    """Find `executable = ...` inside the
    `elif not sysconfig.is_python_build():` branch of _get_shebang()."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
            continue
        call = test.operand
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "is_python_build"
        ):
            continue
        if len(node.body) != 1:
            continue
        stmt = node.body[0]
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "executable"
        ):
            continue
        return stmt
    return None


def main():
    try:
        import pip._vendor.distlib.scripts as distlib_scripts
    except ImportError:
        print("pip (or its vendored distlib) is not importable here - nothing to check.")
        return

    scripts_py = Path(distlib_scripts.__file__)
    source = scripts_py.read_text(encoding="utf-8", newline="")

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"{scripts_py}: failed to parse ({e}). Leaving untouched.")
        return

    assign = find_target_assign(tree)
    if assign is None:
        print(
            f"{scripts_py}: could not locate the expected "
            f"`elif not sysconfig.is_python_build(): executable = ...` shape - "
            f"pip's internals may have changed. Leaving untouched."
        )
        return

    value_src = ast.get_source_segment(source, assign.value)

    if value_src == PATCHED_SRC:
        print(f"{scripts_py}: already WinPython-style, nothing to do.")
        return

    if value_src != UNPATCHED_SRC:
        print(
            f"{scripts_py}: found `executable = {value_src}`, which is neither "
            f"the known patched nor unpatched form. Leaving untouched - check manually."
        )
        return

    start_line, start_col = assign.value.lineno, assign.value.col_offset
    end_line, end_col = assign.value.end_lineno, assign.value.end_col_offset
    if start_line != end_line:
        print(f"{scripts_py}: matched expression spans multiple lines - unexpected shape, leaving untouched.")
        return

    lines = source.splitlines(keepends=True)
    line = lines[start_line - 1]
    lines[start_line - 1] = line[:start_col] + PATCHED_SRC + line[end_col:]

    scripts_py.write_text("".join(lines), encoding="utf-8", newline="")
    print(f"{scripts_py}: patched `{UNPATCHED_SRC}` -> `{PATCHED_SRC}`")


if __name__ == "__main__":
    main()
