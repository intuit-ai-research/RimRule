from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import multiprocessing as mp
import queue as queue_mod
from typing import Any

# Identifiers that grant introspection/eval/IO capability. Rejected wherever
# they're bound or referenced -- as a bare name, an attribute access (e.g.
# operator.attrgetter), or an import alias source -- so untrusted tool source
# cannot reach them via any binding mechanism, with or without a dunder.
# attrgetter/methodcaller resolve a string argument to an attribute/method
# lookup, reaching the same dunders the attribute check blocks (e.g.
# "__class__", "__subclasses__") without ever producing an ast.Attribute
# node; checking the identifier at every binding site (not just call sites)
# means aliasing (`ag = attrgetter`) or `import ... as` renaming can't
# smuggle the reference past the check.
_FORBIDDEN_IDENTIFIERS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
        "object",
        "super",
        "classmethod",
        "staticmethod",
        "property",
        "memoryview",
        "help",
        "exit",
        "quit",
        "license",
        "copyright",
        "credits",
        "attrgetter",
        "methodcaller",
    }
)


class UnsafeToolSource(ValueError):
    """Raised when tool source uses a construct the sandbox forbids."""


def _reject_if_forbidden(identifier: str, context: str) -> None:
    if identifier in _FORBIDDEN_IDENTIFIERS:
        raise UnsafeToolSource(f"{context} {identifier!r} is not permitted")


class _SafetyVisitor(ast.NodeVisitor):
    """Reject dunder attribute access and capability-granting names.

    Blocks the pure-Python escape route (``().__class__.__base__.__subclasses__()``
    and ``getattr``/``type``/``object`` traversal) that a restricted-builtins
    ``exec`` cannot otherwise contain.
    """

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Reject any private/dunder attribute access. Blocks the
        # __subclasses__ traversal and single-underscore re-exports (e.g.
        # random._os -> os).
        if node.attr.startswith("_"):
            raise UnsafeToolSource(
                f"private attribute access is not permitted: .{node.attr}"
            )
        _reject_if_forbidden(node.attr, "access to")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        _reject_if_forbidden(node.id, "use of")
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        # `from operator import attrgetter as ag` binds the forbidden
        # identifier only in ast.alias.name, a plain string never wrapped in
        # a Name or Attribute node -- invisible to the two checks above.
        # Check the imported identifier itself regardless of what it's
        # renamed to.
        _reject_if_forbidden(node.name, "import of")
        self.generic_visit(node)


def _assert_safe_source(source: str) -> None:
    """Parse ``source`` and raise UnsafeToolSource on any forbidden construct."""
    _SafetyVisitor().visit(ast.parse(source))


# Builtins the sandboxed tool source is allowed to reference. Deliberately
# excludes open/eval/exec/compile/input/__import__ and everything else that
# grants filesystem, network, or import capability.
_SAFE_BUILTIN_NAMES = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "ArithmeticError",
        "RuntimeError",
        "StopIteration",
    }
)

# Standard-library modules ToolHop tool source is allowed to import. Covers the
# date/text/number utilities the dataset actually uses while excluding os, sys,
# subprocess, socket, shutil, pathlib, importlib, ctypes, and other capability
# grants. Third-party deps a tool declares (pytz, dateutil, ...) resolve if
# installed; anything off this list and not a submodule of it is blocked.
_ALLOWED_IMPORTS = frozenset(
    {
        "datetime",
        "pytz",
        "dateutil",
        "json",
        "re",
        "collections",
        "itertools",
        "calendar",
        "string",
        "math",
        "cmath",
        "fractions",
        "statistics",
        "numbers",
        "decimal",
        "unicodedata",
        "locale",
        "csv",
        "io",
        "base64",
        "binascii",
        "functools",
        "operator",
        "random",
        "time",
        "textwrap",
        "html",
        "xml",
        "babel",
        "dicttoxml",
        "holidays",
        "roman",
    }
)

# Per-tool-call resource caps applied in the worker process (best-effort;
# unavailable limits are skipped, e.g. RLIMIT_AS on some platforms).
_CPU_SECONDS = 15
_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024  # 1 GiB

# subprocess.Popen's underlying mechanism (fork/vfork/execve). Denied with
# ERRNO, not KILL_PROCESS: Popen's C-level fork/exec helper runs in this
# worker process itself (not a grandchild), and letting the syscall fail
# with a normal errno -- surfacing as an OSError Popen raises, caught by
# _worker's existing except Exception -- avoids any interaction between an
# unmaskable kill and Popen's own child-reaping/wait logic. Blocks the
# attack regardless of what Python-level trick (attrgetter, get_field, ...)
# reaches these; this is the syscall-level backstop for the AST layer's
# blind spot.
_SECCOMP_DENIED_WITH_ERRNO: tuple[str, ...] = (
    "execve",
    "execveat",
    "fork",
    "vfork",
)

# Syscalls with no legitimate use inside the worker under any code path --
# not even as a caught error -- so a hard kill is safe and simplest.
_SECCOMP_DENIED_WITH_KILL: tuple[str, ...] = (
    "socket",
    "socketpair",
    "connect",
    "bind",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
)

# clone() needs argument-level filtering, not a blanket deny:
# multiprocessing.Queue starts a feeder thread (via pthread_create -> clone)
# the first time queue.put() runs, which happens after this filter loads.
# CLONE_THREAD-flagged calls are ordinary thread creation and must stay
# allowed; only bare process-creation clones (what Popen needs) are denied,
# with ERRNO for the same Popen-internal reason as the exec-family calls
# above.
_CLONE_THREAD = 0x00010000


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not permitted in the sandbox")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    source = vars(builtins)
    safe = {name: source[name] for name in _SAFE_BUILTIN_NAMES if name in source}
    safe["__import__"] = _guarded_import
    return safe


def _apply_rlimits() -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return
    for res, limit in (
        (getattr(resource, "RLIMIT_CPU", None), _CPU_SECONDS),
        (getattr(resource, "RLIMIT_AS", None), _ADDRESS_SPACE_BYTES),
    ):
        if res is None:
            continue
        try:
            resource.setrlimit(res, (limit, limit))
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            continue


def _apply_seccomp_filter() -> None:
    try:
        try:
            import seccomp
        except ImportError:
            import pyseccomp as seccomp
    except Exception:  # pragma: no cover - non-Linux or libseccomp missing
        return
    try:
        import errno

        deny_with_errno = seccomp.ERRNO(errno.EPERM)
        f = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
        for syscall in _SECCOMP_DENIED_WITH_ERRNO:
            f.add_rule(deny_with_errno, syscall)
        for syscall in _SECCOMP_DENIED_WITH_KILL:
            f.add_rule(seccomp.KILL_PROCESS, syscall)
        # clone3's flags live in a struct pointed to by arg0, not in arg0
        # itself -- seccomp-BPF filters raw syscall arguments (registers), it
        # cannot dereference that pointer, so no CLONE_THREAD mask is
        # possible here. Left unfiltered (falls through to ALLOW): the
        # exploit path this layer targets (glibc's posix_spawn fast path,
        # used by subprocess.Popen) uses legacy clone/vfork, both covered
        # above.
        f.add_rule(
            deny_with_errno,
            "clone",
            seccomp.Arg(0, seccomp.MASKED_EQ, _CLONE_THREAD, 0),
        )
        f.load()
    except Exception:  # pragma: no cover - platform/container dependent
        return


def _strip_source(source: str) -> str:
    lines = source.splitlines()
    kept = []
    in_def = False
    for line in lines:
        if line.startswith("def "):
            in_def = True
        if in_def:
            kept.append(line)
    return "\n".join(kept)


def _worker(functions: list[str], name: str, args: dict[str, Any], queue) -> None:
    _apply_rlimits()
    _apply_seccomp_filter()
    namespace: dict[str, Any] = {"json": json}
    sandbox_globals = {"__builtins__": _safe_builtins()}
    try:
        for source in functions:
            stripped = _strip_source(source)
            _assert_safe_source(stripped)
            exec(stripped, sandbox_globals, namespace)
        fn = namespace.get(name)
        if not callable(fn):
            raise NameError(f"Unknown tool: {name}")
        # ToolHop functions sometimes print intermediate values. Keep worker
        # stdout/stderr out of the CLI so tqdm remains readable.
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            value = fn(**args)
        queue.put((True, value))
    except Exception as exc:
        queue.put((False, f"{type(exc).__name__}: {exc}"))


def execute_tool(
    functions: list[str], name: str, args: dict[str, Any], timeout: float = 10.0
) -> Any:
    """Run tool ``name`` from ``functions`` in a restricted worker process.

    Defence in depth, three independent layers:

    1. AST validation (cross-platform, always applied): tool source is first
       parsed to reject private and dunder attribute access (which is how a
       restricted-builtins ``exec`` is otherwise escaped via
       ``__subclasses__``/``__globals__`` traversal), capability-granting
       names (``getattr``/``eval``/``exec``/``open``/...), and
       ``attrgetter``/``methodcaller`` (which resolve the same dunders from a
       string argument, including via aliasing or ``import ... as``
       renaming).
    2. Process isolation (POSIX, best-effort): the call runs in a separate
       ``spawn`` process with a restricted builtins allowlist, a guarded
       import allowlist, CPU/address-space rlimits, and a wall-clock
       timeout.
    3. Seccomp syscall filter (Linux-only, best-effort): on Linux, with
       ``pyseccomp`` installed and a working ``libseccomp``, the worker
       additionally installs a kernel-enforced syscall filter. ``execve``/
       ``fork``/``vfork``/bare-process ``clone`` (subprocess.Popen's
       underlying mechanism, which runs in this worker process itself) are
       denied with ``EPERM`` rather than killed, so the failure surfaces as
       an ordinary ``OSError`` Popen raises -- caught like any other tool
       exception -- instead of an unmaskable process kill racing Popen's
       own child-reaping logic. ``socket``/``connect``/``bind``/``ptrace``
       and similar, which have no legitimate use under any code path here,
       are killed outright. This closes the syscall-level gap for *any*
       Python-level trick that reaches those syscalls, regardless of
       whether it produces an ``ast.Attribute``/``ast.Name`` node the AST
       layer can see -- including the ``string.Formatter().get_field()`` gap
       below.

    Known AST-layer gap: the AST check is a syntactic blocklist, not a
    semantic sandbox. Any allowed-stdlib API that resolves an attribute path
    from a *string* at runtime bypasses it, since no ``ast.Attribute``/
    ``ast.Name`` node for the target ever appears in source.
    ``string.Formatter().get_field()`` is one such API (reachable via the
    allowed ``string`` import, or from any string via the builtin
    ``str.format``) and is a confirmed escape to the same
    ``__subclasses__``-to-``subprocess.Popen`` chain the AST checks close for
    ``attrgetter``/``methodcaller``. On Linux with the seccomp layer active,
    this gap is closed at the syscall level even though the Python-level
    trick still runs: ``subprocess.Popen`` still gets invoked, but the
    ``execve``/``fork``/``clone`` syscall it depends on fails with
    ``EPERM``, raising inside the worker rather than executing a shell
    command.

    Residual gap on non-Linux platforms (macOS, Windows): only the AST and
    process-isolation layers apply there; the seccomp layer is a no-op
    (silently skipped, see ``_apply_seccomp_filter``), so the
    ``get_field``-style bypass above is NOT closed on those platforms. Treat
    this executor as containment against accidents and unsophisticated
    inputs only on non-Linux, and as containment additionally hardened
    against the disclosed syscall-level escape on Linux with ``pyseccomp``
    installed -- not as an unconditional security boundary against a
    determined adversary on any platform, until it runs under OS-level
    isolation (a container/gVisor boundary) everywhere.

    Args:
        functions: Python source strings, one per available tool.
        name: The tool function to call.
        args: Keyword arguments for the tool.
        timeout: Seconds before the worker process is killed.

    Returns:
        The tool's return value.

    Raises:
        TimeoutError: If the tool exceeds ``timeout``.
        RuntimeError: If the tool raises or exits without a result.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(functions, name, args, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise TimeoutError(f"Tool {name} timed out")
    try:
        ok, value = queue.get(timeout=1.0)
    except queue_mod.Empty:
        raise RuntimeError(f"Tool {name} exited without a result") from None
    if not ok:
        raise RuntimeError(value)
    return value
