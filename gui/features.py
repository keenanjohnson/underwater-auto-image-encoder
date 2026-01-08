"""
Feature flags for the GUI application.

Toggle features on/off to reduce build size or disable functionality.
"""

# GPR Support - Set to False to disable GPR file support and reduce build size
# When disabled:
# - GPR binaries are not bundled with the application
# - GPR file format is not accepted as input
# - Build size is reduced significantly
GPR_SUPPORT_ENABLED = False
