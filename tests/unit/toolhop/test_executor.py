from __future__ import annotations

import sys
import unittest

from rimrule.toolhop.executor import (
    _CLONE_THREAD,
    _SECCOMP_DENIED_WITH_ERRNO,
    _SECCOMP_DENIED_WITH_KILL,
    _strip_source,
    execute_tool,
)

try:
    import seccomp  # noqa: F401

    _HAS_SECCOMP = True
except ImportError:
    try:
        import pyseccomp  # noqa: F401

        _HAS_SECCOMP = True
    except Exception:
        _HAS_SECCOMP = False

_seccomp_available = sys.platform == "linux" and _HAS_SECCOMP


class StripSourceTest(unittest.TestCase):
    def test_removes_preamble(self):
        source = "import math\nx = 1\ndef add(a, b):\n    return a + b\n"
        result = _strip_source(source)
        self.assertTrue(result.startswith("def add"))
        self.assertNotIn("import math", result)

    def test_no_def(self):
        self.assertEqual(_strip_source("x = 1\ny = 2"), "")


class ExecuteToolTest(unittest.TestCase):
    def test_simple_function(self):
        functions = ["def add(a, b):\n    return a + b\n"]
        result = execute_tool(functions, "add", {"a": 2, "b": 3}, timeout=5.0)
        self.assertEqual(result, 5)

    def test_unknown_function(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown tool"):
            execute_tool([], "nonexistent", {}, timeout=5.0)

    def test_runtime_error(self):
        functions = ["def fail():\n    raise ValueError('boom')\n"]
        with self.assertRaisesRegex(RuntimeError, "boom"):
            execute_tool(functions, "fail", {}, timeout=5.0)

    def test_timeout(self):
        functions = [
            "def slow():\n    x = 0\n    while True:\n        x += 1\n    return x\n"
        ]
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            execute_tool(functions, "slow", {}, timeout=0.5)

    def test_allowed_import_works(self):
        functions = [
            "def d():\n    import datetime\n    return datetime.date(2020, 1, 2).isoformat()\n"
        ]
        result = execute_tool(functions, "d", {}, timeout=5.0)
        self.assertEqual(result, "2020-01-02")


class SandboxBoundaryTest(unittest.TestCase):
    def test_open_is_unavailable(self):
        functions = ["def leak():\n    return open('/etc/passwd').read()\n"]
        with self.assertRaisesRegex(RuntimeError, "open"):
            execute_tool(functions, "leak", {}, timeout=5.0)

    def test_os_import_is_blocked(self):
        functions = ["def esc():\n    import os\n    return os.getcwd()\n"]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_subprocess_import_is_blocked(self):
        functions = [
            "def shell():\n"
            "    import subprocess\n"
            "    return subprocess.run(['echo', 'hi'])\n"
        ]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "shell", {}, timeout=5.0)

    def test_dunder_import_is_blocked(self):
        functions = ["def esc():\n    return __import__('os').getcwd()\n"]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_subclasses_traversal_escape_is_blocked(self):
        functions = [
            "def esc():\n"
            "    for c in ().__class__.__base__.__subclasses__():\n"
            "        if c.__name__ == 'Popen':\n"
            "            return c(['id']).communicate()\n"
            "    return None\n"
        ]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_globals_traversal_escape_is_blocked(self):
        functions = [
            "def esc():\n"
            "    for c in ().__class__.__base__.__subclasses__():\n"
            "        g = getattr(c.__init__, '__globals__', {})\n"
            "        if 'os' in g:\n"
            "            return g['os'].getcwd()\n"
            "    return None\n"
        ]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_private_reexport_escape_is_blocked(self):
        # random._os re-exports the os module; single-underscore access is denied.
        functions = ["def esc():\n    import random\n    return random._os.getcwd()\n"]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_getattr_is_blocked(self):
        functions = ["def esc():\n    return getattr(().__class__, 'mro')()\n"]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_operator_attrgetter_subclasses_escape_is_blocked(self):
        functions = [
            "def esc():\n"
            "    import operator\n"
            "    get_class = operator.attrgetter('__class__')\n"
            "    get_base = operator.attrgetter('__base__')\n"
            "    get_subs = operator.methodcaller('__subclasses__')\n"
            "    for c in get_subs(get_base(get_class(()))):\n"
            "        if c.__name__ == 'Popen':\n"
            "            return operator.methodcaller('communicate')(\n"
            "                c(['whoami'], stdout=-1)\n"
            "            )\n"
            "    return None\n"
        ]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_operator_methodcaller_from_import_escape_is_blocked(self):
        functions = [
            "def esc():\n"
            "    from operator import attrgetter, methodcaller\n"
            "    get_class = attrgetter('__class__')\n"
            "    get_base = attrgetter('__base__')\n"
            "    get_subs = methodcaller('__subclasses__')\n"
            "    return get_subs(get_base(get_class(())))\n"
        ]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_operator_attrgetter_alias_escape_is_blocked(self):
        functions = [
            "def esc():\n"
            "    import operator\n"
            "    ag = operator.attrgetter\n"
            "    mc = operator.methodcaller\n"
            "    get_class = ag('__class__')\n"
            "    get_base = ag('__base__')\n"
            "    get_subs = mc('__subclasses__')\n"
            "    get_name = ag('__name__')\n"
            "    for c in get_subs(get_base(get_class(()))):\n"
            "        if get_name(c) == 'Popen':\n"
            "            return mc('communicate')(c(['whoami'], stdout=-1))\n"
            "    return None\n"
        ]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_operator_import_as_rename_escape_is_blocked(self):
        functions = [
            "def esc():\n"
            "    from operator import attrgetter as ag, methodcaller as mc\n"
            "    get_class = ag('__class__')\n"
            "    get_base = ag('__base__')\n"
            "    get_subs = mc('__subclasses__')\n"
            "    return get_subs(get_base(get_class(())))\n"
        ]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_operator_module_alias_attribute_access_escape_is_blocked(self):
        # Renames the module (operator -> op), not a forbidden identifier, so
        # this exercises visit_Attribute on op.attrgetter/op.methodcaller --
        # not visit_alias, which only fires when the imported name itself
        # (e.g. attrgetter) is forbidden. Module names are never forbidden.
        functions = [
            "def esc():\n"
            "    import operator as op\n"
            "    get_subs = op.methodcaller('__subclasses__')\n"
            "    return get_subs(op.attrgetter('__base__')(op.attrgetter('__class__')(())))\n"
        ]
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            execute_tool(functions, "esc", {}, timeout=5.0)

    def test_operator_add_still_works(self):
        functions = [
            "def f(a, b):\n    import operator\n    return operator.add(a, b)\n"
        ]
        result = execute_tool(functions, "f", {"a": 2, "b": 3}, timeout=5.0)
        self.assertEqual(result, 5)

    def test_operator_itemgetter_still_works(self):
        functions = [
            'def f(d):\n    import operator\n    return operator.itemgetter("x")(d)\n'
        ]
        result = execute_tool(functions, "f", {"d": {"x": 99}}, timeout=5.0)
        self.assertEqual(result, 99)

    def test_unrelated_import_as_rename_is_allowed(self):
        functions = [
            "def f():\n"
            "    import datetime as dt\n"
            "    return dt.date(2020, 1, 2).isoformat()\n"
        ]
        result = execute_tool(functions, "f", {}, timeout=5.0)
        self.assertEqual(result, "2020-01-02")

    def test_underscore_string_argument_to_ordinary_call_is_allowed(self):
        functions = ['def f(record):\n    return record.get("_id")\n']
        result = execute_tool(functions, "f", {"record": {"_id": 42}}, timeout=5.0)
        self.assertEqual(result, 42)

    def test_underscore_prefixed_string_literal_is_allowed(self):
        functions = ['def f(s):\n    return s.startswith("_")\n']
        result = execute_tool(functions, "f", {"s": "_id"}, timeout=5.0)
        self.assertTrue(result)


class SeccompConstantsTest(unittest.TestCase):
    # Platform-independent: catches a typo'd syscall name or wrong flag
    # value before it ever reaches Linux CI, where the gated tests below
    # are the only ones that can exercise the filter for real.
    def test_errno_denied_syscalls_are_the_expected_set(self):
        self.assertEqual(
            set(_SECCOMP_DENIED_WITH_ERRNO),
            {"execve", "execveat", "fork", "vfork"},
        )

    def test_kill_denied_syscalls_are_the_expected_set(self):
        self.assertEqual(
            set(_SECCOMP_DENIED_WITH_KILL),
            {
                "socket",
                "socketpair",
                "connect",
                "bind",
                "ptrace",
                "process_vm_readv",
                "process_vm_writev",
            },
        )

    def test_clone_thread_matches_linux_kernel_flag(self):
        self.assertEqual(_CLONE_THREAD, 0x00010000)


@unittest.skipUnless(_seccomp_available, "requires Linux + pyseccomp")
class SeccompFilterTest(unittest.TestCase):
    def test_simple_function_still_works_with_filter_active(self):
        # Guards against a wrong CLONE_THREAD mask: multiprocessing.Queue
        # starts its feeder thread (clone) on the first queue.put(), which
        # happens after the seccomp filter loads. If the mask is wrong, this
        # times out or exits without a result instead of returning 5.
        functions = ["def add(a, b):\n    return a + b\n"]
        result = execute_tool(functions, "add", {"a": 2, "b": 3}, timeout=5.0)
        self.assertEqual(result, 5)


@unittest.skipUnless(_seccomp_available, "requires Linux + pyseccomp")
class SeccompRegressionTest(unittest.TestCase):
    def test_get_field_subprocess_escape_is_blocked(self):
        # The AST layer cannot see this: get_field resolves the dotted path
        # from a runtime string, never producing an ast.Attribute/ast.Name
        # node. The exec-family syscalls Popen depends on are denied with
        # ERRNO (not a kill), so this surfaces as the OSError Popen raises
        # inside the worker, caught by _worker's except Exception -- the
        # usual caught-exception path, not a killed-process timeout.
        functions = [
            "def esc():\n"
            "    import string\n"
            "    fmt = string.Formatter()\n"
            "    subs, _ = fmt.get_field(\n"
            "        '0.__class__.__base__.__subclasses__', [()], {}\n"
            "    )\n"
            "    for c in subs():\n"
            "        name, _ = fmt.get_field('0.__name__', [c], {})\n"
            "        if name == 'Popen':\n"
            "            proc = c(['whoami'], stdout=-1)\n"
            "            out, _ = fmt.get_field('0.communicate', [proc], {})\n"
            "            return out()\n"
            "    return None\n"
        ]
        with self.assertRaisesRegex(RuntimeError, "OSError|PermissionError"):
            execute_tool(functions, "esc", {}, timeout=5.0)
