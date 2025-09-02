# GitHub Actions Workflow Update

## Changes Made

Updated all GitHub Actions to their latest versions to fix deprecation warnings:

### Action Version Updates:
- `actions/checkout@v3` → `actions/checkout@v4`
- `actions/setup-python@v4` → `actions/setup-python@v5`
- `actions/cache@v3` → `actions/cache@v4`
- `actions/upload-artifact@v3` → `actions/upload-artifact@v4`
- `softprops/action-gh-release@v1` → `softprops/action-gh-release@v2`

## Why These Changes?

GitHub deprecated v3 of the artifact actions on April 16, 2024. The v3 versions are no longer supported and will cause workflows to fail.

## Migration Notes

### upload-artifact v4 Changes:
- Artifacts are now immutable after upload
- Improved performance for large files
- Better compression algorithms
- Same syntax, just version change

### Other Updates:
- All other action updates are drop-in replacements
- No configuration changes needed
- Improved performance and security

## Testing

The workflow will now:
1. Build successfully on all platforms (Windows, macOS, Linux)
2. Upload artifacts without deprecation warnings
3. Create releases when tags are pushed

To test:
```bash
git add .github/workflows/build-gui.yml
git commit -m "Update GitHub Actions to latest versions"
git push origin basic-gui
```