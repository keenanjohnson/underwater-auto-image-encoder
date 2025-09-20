"""
Fixes for PyTorch when running in PyInstaller bundle
This module patches torch to work correctly in frozen applications
"""

import sys
import os

def patch_torch_for_bundled_app():
    """Apply patches to make torch work in PyInstaller bundles"""

    # Only apply patches if we're in a PyInstaller bundle
    if not hasattr(sys, '_MEIPASS'):
        return

    # Patch the inspect module to handle missing source code gracefully
    import inspect
    original_findsource = inspect.findsource

    def patched_findsource(object):
        try:
            return original_findsource(object)
        except (OSError, IOError) as e:
            # Return dummy source for frozen modules
            if "could not get source code" in str(e) or "source code not available" in str(e):
                # Return minimal dummy source
                return (["# Source not available in bundled application\n"], 0)
            raise

    inspect.findsource = patched_findsource

    # Patch torch's config module to skip source inspection
    try:
        import torch.utils._config_module as config_module

        def patched_get_assignments(source_code, ignore_prefix="# IGNORE"):
            """Return empty dict for bundled apps"""
            return {}

        config_module.get_assignments_with_compile_ignored_comments = patched_get_assignments
    except ImportError:
        pass  # Torch not installed or different version

    # Set environment variables to reduce torch's inspection attempts
    os.environ['PYTORCH_DISABLE_LIBRARY_VALIDATION'] = '1'
    os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'

# Apply patches when module is imported
patch_torch_for_bundled_app()