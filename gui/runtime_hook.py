"""
Runtime hook to ensure src modules can be imported and PyTorch works with PyInstaller
"""
import sys
import os
from pathlib import Path
import inspect

# Get the directory where the executable is located
if hasattr(sys, '_MEIPASS'):
    # Running in PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running in normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))

# For macOS app bundles, also check Resources directory
if sys.platform == 'darwin' and hasattr(sys, '_MEIPASS'):
    # Check both Frameworks and Resources for src
    possible_paths = [
        os.path.join(base_path, 'src'),
        os.path.join(base_path, '..', 'Resources', 'src'),
        os.path.join(base_path, '..', 'Frameworks', 'src'),
    ]
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)
            # Also add parent so "src.models" can be imported
            parent = os.path.dirname(abs_path)
            if parent not in sys.path:
                sys.path.insert(0, parent)
else:
    # Add src directory to Python path
    src_path = os.path.join(base_path, 'src')
    if os.path.exists(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)

# Also add the base path
if base_path not in sys.path:
    sys.path.insert(0, base_path)

# PyTorch compatibility fixes for PyInstaller
# PyTorch's source code introspection fails in frozen apps, so we patch it
if hasattr(sys, '_MEIPASS'):
    # Patch inspect module
    _original_getsource = inspect.getsource
    _original_getsourcelines = inspect.getsourcelines
    _original_findsource = inspect.findsource

    def _patched_getsource(object):
        """Patched getsource that returns empty string instead of failing"""
        try:
            return _original_getsource(object)
        except (OSError, TypeError):
            # Return empty source code when running in PyInstaller
            return ""

    def _patched_getsourcelines(object):
        """Patched getsourcelines that returns empty list instead of failing"""
        try:
            return _original_getsourcelines(object)
        except (OSError, TypeError):
            # Return empty source lines when running in PyInstaller
            return ([], 0)

    def _patched_findsource(object):
        """Patched findsource that returns empty source instead of failing"""
        try:
            return _original_findsource(object)
        except (OSError, TypeError):
            # Return empty source when running in PyInstaller
            return ([], 0)

    inspect.getsource = _patched_getsource
    inspect.getsourcelines = _patched_getsourcelines
    inspect.findsource = _patched_findsource

    # Patch torch._sources module which PyTorch uses for JIT compilation
    # This needs to be done after torch is imported, so we use an import hook
    def _patch_torch_sources():
        """Patch torch._sources.parse_def to handle missing source files"""
        try:
            import torch._sources
            _original_parse_def = torch._sources.parse_def

            def _patched_parse_def(fn):
                """Patched parse_def that returns None instead of failing"""
                try:
                    return _original_parse_def(fn)
                except (RuntimeError, OSError, TypeError):
                    # Return None when source parsing fails in PyInstaller
                    return None

            torch._sources.parse_def = _patched_parse_def
        except (ImportError, AttributeError):
            # torch._sources may not be available yet or at all
            pass

    # Patch torch JIT internal checking
    def _patch_torch_jit_internal():
        """Patch torch._jit_internal to skip source code checks"""
        try:
            import torch._jit_internal

            # Patch _check_overload_body to skip source validation
            _original_check_overload_body = torch._jit_internal._check_overload_body

            def _patched_check_overload_body(func):
                """Patched check that skips source validation"""
                try:
                    return _original_check_overload_body(func)
                except (RuntimeError, OSError, TypeError):
                    # Skip validation in PyInstaller - JIT won't work but inference will
                    return None

            torch._jit_internal._check_overload_body = _patched_check_overload_body
        except (ImportError, AttributeError):
            pass

    # Install an import hook to patch torch modules when they're imported
    class _TorchPatchImporter:
        """Meta path finder to patch torch modules on import"""

        def find_module(self, fullname, path=None):
            _ = path  # Unused but required by import hook interface
            if fullname == 'torch._sources':
                return self
            elif fullname == 'torch._jit_internal':
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                # Module already imported, patch it
                if fullname == 'torch._sources':
                    _patch_torch_sources()
                elif fullname == 'torch._jit_internal':
                    _patch_torch_jit_internal()
                return sys.modules[fullname]

            # Let the normal import system handle it
            import importlib
            mod = importlib.import_module(fullname)

            # Patch after import
            if fullname == 'torch._sources':
                _patch_torch_sources()
            elif fullname == 'torch._jit_internal':
                _patch_torch_jit_internal()

            return mod

    # Install the import hook
    sys.meta_path.insert(0, _TorchPatchImporter())