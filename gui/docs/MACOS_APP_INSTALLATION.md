# macOS App Installation Guide

## Opening the Downloaded App

When you download `UnderwaterEnhancer.app` from GitHub Actions, macOS will block it by default because it's not from a recognized developer. Here's how to open it:

### Method 1: Remove Quarantine (Recommended)
Open Terminal and run:
```bash
# Remove the quarantine flag
xattr -cr /path/to/UnderwaterEnhancer.app

# For example, if it's in Downloads:
xattr -cr ~/Downloads/UnderwaterEnhancer.app

# Then open the app
open ~/Downloads/UnderwaterEnhancer.app
```

### Method 2: Right-Click to Open
1. Locate `UnderwaterEnhancer.app` in Finder
2. Right-click (or Control-click) on the app
3. Select "Open" from the context menu
4. In the dialog that appears, click "Open"
5. The app will now be saved as an exception to your security settings

### Method 3: System Settings Approval
1. Double-click the app (it will be blocked)
2. Open System Settings → Privacy & Security
3. Scroll down to see: "UnderwaterEnhancer was blocked from use because it is not from an identified developer"
4. Click "Open Anyway"
5. Enter your password if prompted
6. Click "Open" in the final dialog

## Why Does This Happen?

macOS has built-in security features called Gatekeeper that prevent unsigned apps from running. Our CI builds are "ad-hoc" signed, which means they can run but require user approval.

## Permanent Solution for Your Mac

Once you've opened the app using any method above, macOS will remember your choice and won't block it again on your machine.

## For Developers: Building Locally

If you build the app locally on your Mac, it won't have these restrictions:
```bash
# Build locally without restrictions
python build_scripts/build_app.py
open dist/UnderwaterEnhancer.app
```

## Troubleshooting

### App Won't Open at All
Check the executable permissions:
```bash
chmod +x /path/to/UnderwaterEnhancer.app/Contents/MacOS/UnderwaterEnhancer
```

### "App is Damaged" Error
This usually means the download was corrupted. Try:
1. Re-download the app
2. Make sure to unzip/extract properly
3. Use the xattr command above

### Console Shows "killed" Message
This is the quarantine in action. Use Method 1 above to fix it.

## Distribution Note

For proper distribution without these issues, the app would need:
1. An Apple Developer account ($99/year)
2. Proper code signing with a Developer ID certificate
3. Notarization by Apple

The current CI builds are suitable for testing and internal use.