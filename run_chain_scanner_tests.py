"""
Standalone test runner for tests/test_chain_scanner.py.

The sandbox can't install pytest (proxy 403) so we emulate just enough
of pytest's fixture / approx / raises / monkeypatch surface to drive the
test file directly. Returns exit code 0 on full pass, 1 on any failure.
"""

from __future__ import annotations

import inspect
import sys
import traceback
import types
from pathlib import Path

# -----------------------------------------------------------------------------
# Stub heavy / unavailable deps BEFORE importing anything from trading_agent
# -----------------------------------------------------------------------------

# pandas_market_calendars — the test monkeypatches calendar_utils.next_weekly_expiration
# but the import chain still has to succeed.
pmc = types.ModuleType("pandas_market_calendars")


class _FakeCal:
    def schedule(self, start_date, end_date):
        import pandas as pd
        idx = pd.bdate_range(start=start_date, end=end_date)
        return pd.DataFrame(index=idx, data={"market_open": idx, "market_close": idx})


def _get_calendar(name):
    return _FakeCal()


pmc.get_calendar = _get_calendar
sys.modules["pandas_market_calendars"] = pmc

# scipy / scipy.stats — used downstream of regime.py if anything pulls it in.
scipy_mod = types.ModuleType("scipy")
scipy_stats = types.ModuleType("scipy.stats")
scipy_stats.percentileofscore = lambda a, score, kind="rank": 50.0
sys.modules["scipy"] = scipy_mod
sys.modules["scipy.stats"] = scipy_stats

# -----------------------------------------------------------------------------
# Stub pytest
# -----------------------------------------------------------------------------

class _Approx:
    def __init__(self, expected, rel=1e-9, abs_tol=1e-9):
        self.expected = expected
        self.rel = rel
        self.abs_tol = abs_tol

    def __eq__(self, other):
        if self.expected == 0:
            return abs(other) <= max(self.rel, self.abs_tol)
        return abs(other - self.expected) <= max(
            self.abs_tol,
            self.rel * max(abs(self.expected), abs(other)),
        )

    def __repr__(self):
        return f"approx({self.expected})"


def _approx(expected, rel=1e-6, abs_tol=1e-9):
    return _Approx(expected, rel=rel, abs_tol=abs_tol)


class _Raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} not raised")
        return issubclass(exc_type, self.exc_type)


pytest_mod = types.ModuleType("pytest")
pytest_mod.approx = _approx
pytest_mod.raises = _Raises


def _fixture(*args, **kwargs):
    """Decorator that just tags the function — our runner calls it as a fn."""
    def _wrap(fn):
        fn._is_fixture = True
        fn._autouse = kwargs.get("autouse", False)
        return fn
    if len(args) == 1 and callable(args[0]):
        return _wrap(args[0])
    return _wrap


pytest_mod.fixture = _fixture
sys.modules["pytest"] = pytest_mod


# -----------------------------------------------------------------------------
# Monkeypatch fixture stub
# -----------------------------------------------------------------------------

class _Monkeypatch:
    def __init__(self):
        self._undos = []

    def setattr(self, target, name, value, raising=True):
        old = getattr(target, name)
        self._undos.append((target, name, old))
        setattr(target, name, value)

    def undo_all(self):
        for target, name, old in reversed(self._undos):
            setattr(target, name, old)


# -----------------------------------------------------------------------------
# Test discovery + execution
# -----------------------------------------------------------------------------

def _import_test_module():
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent / "tests"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_chain_scanner",
        Path(__file__).parent / "tests" / "test_chain_scanner.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_fixtures(mod):
    fixtures = {}
    for name, obj in vars(mod).items():
        if callable(obj) and getattr(obj, "_is_fixture", False):
            fixtures[name] = obj
    return fixtures


def _resolve_fixture(name, fixtures, cache, mp):
    """Resolve a fixture by name, recursively resolving its own deps."""
    if name in cache:
        return cache[name]
    if name == "monkeypatch":
        cache[name] = mp
        return mp
    if name not in fixtures:
        raise KeyError(f"unknown fixture {name!r}")
    fn = fixtures[name]
    sig = inspect.signature(fn)
    args = {}
    for p in sig.parameters.values():
        args[p.name] = _resolve_fixture(p.name, fixtures, cache, mp)
    val = fn(**args)
    cache[name] = val
    return val


def _run_tests(mod):
    fixtures = _collect_fixtures(mod)
    autouse = [n for n, f in fixtures.items() if getattr(f, "_autouse", False)]
    failures = []
    passed = 0

    test_classes = [
        (name, cls) for name, cls in vars(mod).items()
        if inspect.isclass(cls) and name.startswith("Test")
    ]

    for cls_name, cls in test_classes:
        instance = cls()
        for meth_name, meth in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not meth_name.startswith("test_"):
                continue
            full = f"{cls_name}::{meth_name}"
            mp = _Monkeypatch()
            cache = {}
            try:
                # Run autouse fixtures first
                for au in autouse:
                    _resolve_fixture(au, fixtures, cache, mp)
                # Build kwargs from method signature
                sig = inspect.signature(meth)
                kwargs = {}
                for p in sig.parameters.values():
                    kwargs[p.name] = _resolve_fixture(p.name, fixtures, cache, mp)
                meth(**kwargs)
                passed += 1
                print(f"  PASS  {full}")
            except Exception as exc:
                failures.append((full, exc, traceback.format_exc()))
                print(f"  FAIL  {full}: {exc}")
            finally:
                mp.undo_all()

    print()
    print(f"Result: {passed} passed, {len(failures)} failed")
    if failures:
        print()
        print("=" * 70)
        for name, exc, tb in failures:
            print(f"FAIL: {name}")
            print(tb)
            print("-" * 70)
    return 0 if not failures else 1


if __name__ == "__main__":
    mod = _import_test_module()
    sys.exit(_run_tests(mod))
