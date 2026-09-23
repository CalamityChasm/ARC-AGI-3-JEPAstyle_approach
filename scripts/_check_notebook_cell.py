"""Static use-before-definition check for a generated notebook cell.

Why this exists: v5 of the backtest kernel died on
`NameError: name 'ARMS' is not defined` -- a reference placed above the
definition. The build already ran `ast.parse` on every cell, and
`ast.parse` accepts that happily, because it is valid syntax. Syntax
checking is not execution checking.

That bug cost a GPU session to discover something detectable in
milliseconds. So: walk the cell's module-level statements in order,
track what has been bound, and flag any module-level load of a name that
is not yet bound and is not a builtin.

Deliberately conservative -- it only inspects module level and skips
function/class bodies, since those run later and may legitimately refer
to names defined further down. That is enough to catch the whole class of
"I moved a block and broke the order" mistakes without false alarms.
"""

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

_BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}


def _bound_by(node: ast.AST) -> set[str]:
    """Names a statement binds at module level."""
    names: set[str] = set()
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            names |= _target_names(target)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        names |= _target_names(node.target)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        names |= _target_names(node.target)
    elif isinstance(node, ast.With):
        for item in node.items:
            if item.optional_vars is not None:
                names |= _target_names(item.optional_vars)
    elif isinstance(node, ast.Try):
        for handler in node.handlers:
            if handler.name:
                names.add(handler.name)
    return names


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for element in target.elts:
            out |= _target_names(element)
        return out
    return set()


def _loads(node: ast.AST) -> list[tuple[str, int]]:
    """Every Name load inside a statement, excluding nested func/class bodies.

    Three things bind *before* the code around them runs, and reporting
    them as undefined would be a false alarm:
      - a `for` target, bound before its body,
      - a `with ... as x` name, bound before its body,
      - a comprehension target, scoped to the comprehension.
    All three are tracked as locally bound rather than reported.
    """
    found: list[tuple[str, int]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.local: set[str] = set()

        def visit_FunctionDef(self, n):  # noqa: N802 - bodies run later
            for d in n.decorator_list:
                self.visit(d)
            for d in n.args.defaults:
                self.visit(d)

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

        def visit_Lambda(self, n):  # noqa: N802
            for d in n.args.defaults:
                self.visit(d)

        def visit_ClassDef(self, n):  # noqa: N802
            for b in n.bases:
                self.visit(b)
            for d in n.decorator_list:
                self.visit(d)

        def _comprehension(self, n):
            for gen in n.generators:
                self.visit(gen.iter)
                self.local |= _target_names(gen.target)
                for cond in gen.ifs:
                    self.visit(cond)
            for attr in ("elt", "key", "value"):
                child = getattr(n, attr, None)
                if child is not None:
                    self.visit(child)

        visit_ListComp = _comprehension  # noqa: N815
        visit_SetComp = _comprehension  # noqa: N815
        visit_GeneratorExp = _comprehension  # noqa: N815
        visit_DictComp = _comprehension  # noqa: N815

        def visit_For(self, n):  # noqa: N802
            self.visit(n.iter)
            self.local |= _target_names(n.target)
            for child in n.body + n.orelse:
                self.visit(child)

        visit_AsyncFor = visit_For  # noqa: N815

        def visit_With(self, n):  # noqa: N802
            for item in n.items:
                self.visit(item.context_expr)
                if item.optional_vars is not None:
                    self.local |= _target_names(item.optional_vars)
            for child in n.body:
                self.visit(child)

        visit_AsyncWith = visit_With  # noqa: N815

        def visit_ExceptHandler(self, n):  # noqa: N802
            if n.type is not None:
                self.visit(n.type)
            if n.name:
                self.local.add(n.name)
            for child in n.body:
                self.visit(child)

        def visit_Name(self, n):  # noqa: N802
            if isinstance(n.ctx, ast.Load) and n.id not in self.local:
                found.append((n.id, n.lineno))

    Visitor().visit(node)
    return found


def strip_magics(source: str) -> str:
    """Blank out IPython magics so the rest of a cell can be parsed.

    Notebook cells legitimately contain `!pip install ...`, `%cd`, and
    `%%bash`, none of which is Python. Lines are replaced with an empty
    line rather than removed, so reported line numbers still match the
    cell the author is looking at.
    """
    out = []
    in_cell_magic = False
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%%"):
            in_cell_magic = True
        if in_cell_magic or stripped.startswith(("!", "%")):
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def check_source(source: str, label: str = "<cell>") -> list[str]:
    """Return a list of problem descriptions; empty means clean."""
    source = strip_magics(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{label}: SyntaxError line {exc.lineno}: {exc.msg}"]

    bound: set[str] = set(_BUILTINS)
    problems: list[str] = []

    for statement in tree.body:
        # Everything this statement binds anywhere inside itself counts as
        # available while checking it. That makes the check deliberately
        # permissive *within* a statement -- it will not catch a
        # use-before-assign inside a loop body -- in exchange for zero
        # false alarms. The bug this exists to catch is module-level
        # ordering (a block referenced above where it is defined), and
        # that is still caught exactly.
        inner: set[str] = set()
        for child in ast.walk(statement):
            inner |= _bound_by(child)

        for name, lineno in _loads(statement):
            if name not in bound and name not in inner:
                problems.append(
                    f"{label}:{lineno}: '{name}' used before it is defined"
                )
        bound |= _bound_by(statement)
        # Names bound anywhere inside a compound statement's body are
        # visible afterwards at module level.
        for child in ast.walk(statement):
            if child is not statement:
                bound |= _bound_by(child)

    return problems


def check_notebook(path: Path) -> list[str]:
    import json

    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    # Each cell runs in one shared namespace, so names bound by an earlier
    # cell are available to a later one.
    carried: set[str] = set()
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        seeded = "".join(f"{n} = None\n" for n in sorted(carried))
        offset = len(seeded.splitlines())
        for problem in check_source(seeded + source, f"cell{index}"):
            try:
                _, lineno, rest = problem.split(":", 2)
                problems.append(f"cell{index}:{int(lineno) - offset}:{rest}")
            except ValueError:
                problems.append(problem)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for statement in tree.body:
            carried |= _bound_by(statement)
            for child in ast.walk(statement):
                carried |= _bound_by(child)
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: _check_notebook_cell.py <notebook.ipynb>")
        return 2
    problems = check_notebook(Path(argv[0]))
    if problems:
        print(f"FAILED -- {len(problems)} problem(s):")
        for problem in problems:
            print("  " + problem)
        return 1
    print("use-before-definition check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
