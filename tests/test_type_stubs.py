"""Type stubs ship next to the annotated modules.

A `.pyi` stub replaces its module for type checkers and IDEs, and the package ships
`py.typed`. A stub that drifts from its module gives users wrong defaults and flags
valid calls as errors. The modules carry their own annotations, so the package ships
no stubs. Any stub added later must match its module.
"""

import ast
from pathlib import Path

import fastforward

PACKAGE = Path(fastforward.__file__).parent


def _signatures(path: Path) -> dict:
    """Parameter names and defaults per function, keyed `func` or `Class.method`."""

    def sig(fn):
        a = fn.args
        pos = a.posonlyargs + a.args
        defaults = [None] * (len(pos) - len(a.defaults)) + [ast.unparse(d) for d in a.defaults]
        kw = [None if d is None else ast.unparse(d) for d in a.kw_defaults]
        return (
            [(p.arg, d) for p, d in zip(pos, defaults)],
            [(p.arg, d) for p, d in zip(a.kwonlyargs, kw)],
        )

    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = sig(node)
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{m.name}"] = sig(m)
    return out


def test_stubs_match_their_modules():
    for stub in sorted(PACKAGE.glob("*.pyi")):
        module = stub.with_suffix(".py")
        if not module.exists():
            continue  # a stub for a compiled module has no source to drift from
        stub_sigs, module_sigs = _signatures(stub), _signatures(module)
        public = {n for n in module_sigs if not n.split(".")[-1].startswith("_")}
        missing = sorted(public - set(stub_sigs))
        assert not missing, f"{stub.name} lacks {missing}, so type checkers reject them"
        for name, s in stub_sigs.items():
            assert s == module_sigs.get(name), f"{stub.name}: {name} differs from {module.name}"
