#!/usr/bin/env python3
"""
Test script to verify GUI window sizing shows all controls
"""

import customtkinter as ctk
from src.gui.main_window import UnderwaterEnhancerApp

def test_window_size():
    """Test that the window is properly sized"""
    print("Testing GUI Window Sizing...")
    print("-" * 40)
    
    app = UnderwaterEnhancerApp()
    
    # Update to calculate actual sizes
    app.update_idletasks()
    
    # Get window info
    width = app.winfo_width()
    height = app.winfo_height()
    
    print(f"Window Size: {width}x{height}")
    print(f"Configured: 950x750")
    print(f"Minimum: 900x700")
    
    # Check if main controls exist and are visible
    controls = [
        ("Model Browse Button", app.model_browse_btn),
        ("Input Browse Button", app.input_browse_btn),
        ("Output Browse Button", app.output_browse_btn),
        ("Process Button", app.process_btn),
        ("Cancel Button", app.cancel_btn),
        ("Theme Button", app.theme_btn),
        ("Progress Bar", app.progress_bar),
        ("Log Text", app.log_text)
    ]
    
    print("\nControl Visibility:")
    all_visible = True
    for name, widget in controls:
        try:
            # Check if widget is mapped (visible)
            is_visible = widget.winfo_ismapped()
            status = "✓" if is_visible else "✗"
            print(f"  {status} {name}")
            if not is_visible:
                all_visible = False
        except:
            print(f"  ✗ {name} - Error checking visibility")
            all_visible = False
    
    print("-" * 40)
    if all_visible:
        print("✓ All controls are visible!")
    else:
        print("✗ Some controls are not visible")
    
    # Keep window open for manual inspection
    print("\nWindow will stay open for 3 seconds for visual inspection...")
    app.after(3000, app.destroy)
    app.mainloop()

if __name__ == "__main__":
    test_window_size()