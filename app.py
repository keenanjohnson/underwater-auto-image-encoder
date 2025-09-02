#!/usr/bin/env python3
"""
Underwater Image Enhancer GUI Application
Main entry point for the desktop application
"""

import sys
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.gui.main_window import UnderwaterEnhancerApp

def main():
    """Main application entry point"""
    app = UnderwaterEnhancerApp()
    app.mainloop()

if __name__ == "__main__":
    main()