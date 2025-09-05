#!/usr/bin/env python3
"""
Create application icons for Underwater Enhancer
Creates a wave-based icon with underwater theme
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import os
from pathlib import Path

def create_underwater_icon(size=512):
    """Create an underwater-themed icon with enhancement symbolism"""
    # Create a new image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    
    # Create circular background
    center = size // 2
    radius = int(size * 0.45)
    
    # Create base circle with ocean gradient
    base = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    base_draw = ImageDraw.Draw(base)
    
    # Ocean blue gradient background
    for i in range(radius, 0, -2):
        progress = (radius - i) / radius
        # Gradient from deep ocean to bright aqua
        r = int(0 + 80 * progress)
        g = int(120 + 80 * progress)
        b = int(180 + 50 * progress)
        base_draw.ellipse(
            [center - i, center - i, center + i, center + i],
            fill=(r, g, b, 255)
        )
    
    # Add wave pattern overlay
    wave_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    wave_draw = ImageDraw.Draw(wave_layer)
    
    # Create three wave bands
    num_waves = 3
    for wave_idx in range(num_waves):
        wave_points = []
        wave_offset = radius * 0.2 * (wave_idx - 1)
        base_y = center + wave_offset
        
        # Generate smooth wave using sine function
        for x in range(0, size + 1, 2):
            # Calculate wave amplitude based on distance from center
            dist_from_center = abs(x - center)
            amplitude = 25 * (1 - dist_from_center / center) * 0.7
            
            # Multiple sine waves for more natural look
            y = base_y + amplitude * np.sin(x * 0.015 + wave_idx * 1.5)
            wave_points.append((x, y))
        
        # Complete the polygon to fill below the wave
        wave_points.extend([(size, size * 2), (0, size * 2)])
        
        # Wave colors - lighter on top
        alpha = 120 - wave_idx * 20
        if wave_idx == 0:
            color = (180, 230, 250, alpha)  # Light cyan
        elif wave_idx == 1:
            color = (120, 200, 240, alpha)  # Medium blue
        else:
            color = (80, 170, 220, alpha)   # Deeper blue
        
        wave_draw.polygon(wave_points, fill=color)
    
    # Create circular mask
    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        fill=255
    )
    
    # Apply mask to wave layer
    wave_layer.putalpha(mask)
    
    # Composite base and waves
    img = Image.alpha_composite(base, wave_layer)
    
    # Add enhancement symbol - an upward arrow with sparkles
    symbol_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    symbol_draw = ImageDraw.Draw(symbol_layer)
    
    # Draw main upward arrow
    arrow_size = int(radius * 0.5)
    arrow_x = center
    arrow_y = center
    
    # Arrow shaft
    shaft_width = int(arrow_size * 0.25)
    shaft_height = int(arrow_size * 0.6)
    symbol_draw.rectangle(
        [arrow_x - shaft_width//2, arrow_y,
         arrow_x + shaft_width//2, arrow_y + shaft_height],
        fill=(255, 255, 255, 220)
    )
    
    # Arrow head (triangle)
    head_width = int(arrow_size * 0.5)
    head_height = int(arrow_size * 0.4)
    arrow_points = [
        (arrow_x, arrow_y - head_height),  # Top point
        (arrow_x - head_width//2, arrow_y),  # Bottom left
        (arrow_x + head_width//2, arrow_y),  # Bottom right
    ]
    symbol_draw.polygon(arrow_points, fill=(255, 255, 255, 220))
    
    # Add sparkle effects around arrow
    sparkle_positions = [
        (arrow_x - arrow_size//3, arrow_y - arrow_size//4),
        (arrow_x + arrow_size//3, arrow_y - arrow_size//3),
        (arrow_x, arrow_y - arrow_size//2 - 10),
    ]
    
    for sx, sy in sparkle_positions:
        # Draw 4-point star sparkle
        sparkle_size = 8
        symbol_draw.line(
            [(sx - sparkle_size, sy), (sx + sparkle_size, sy)],
            fill=(255, 255, 255, 180), width=2
        )
        symbol_draw.line(
            [(sx, sy - sparkle_size), (sx, sy + sparkle_size)],
            fill=(255, 255, 255, 180), width=2
        )
    
    # Apply mask to symbol layer
    symbol_layer.putalpha(mask)
    img = Image.alpha_composite(img, symbol_layer)
    
    # Add subtle top highlight for depth
    highlight_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    highlight_draw = ImageDraw.Draw(highlight_layer)
    
    # Elliptical highlight at top
    highlight_ellipse = [
        center - int(radius * 0.8), center - radius,
        center + int(radius * 0.8), center - int(radius * 0.3)
    ]
    highlight_draw.ellipse(highlight_ellipse, fill=(255, 255, 255, 50))
    
    # Blur the highlight
    highlight_layer = highlight_layer.filter(ImageFilter.GaussianBlur(radius=15))
    highlight_layer.putalpha(mask)
    img = Image.alpha_composite(img, highlight_layer)
    
    # Add subtle vignette/border
    border_layer = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    border_draw = ImageDraw.Draw(border_layer)
    
    # Draw border
    border_width = max(4, int(size * 0.02))
    border_draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        outline=(0, 80, 140, 255), width=border_width
    )
    
    # Inner highlight border
    border_draw.ellipse(
        [center - radius + border_width, center - radius + border_width,
         center + radius - border_width, center + radius - border_width],
        outline=(100, 180, 220, 100), width=1
    )
    
    img = Image.alpha_composite(img, border_layer)
    
    return img

def create_ico_file(img, output_path):
    """Create Windows .ico file with multiple sizes"""
    sizes = [16, 32, 48, 64, 128, 256]
    icons = []
    
    for size in sizes:
        resized = img.resize((size, size), Image.Resampling.LANCZOS)
        # Enhance contrast for smaller sizes
        if size <= 32:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(resized)
            resized = enhancer.enhance(1.2)
        icons.append(resized)
    
    # Save as .ico with multiple sizes
    icons[0].save(output_path, format='ICO', sizes=[(s, s) for s in sizes], 
                  append_images=icons[1:])
    print(f"Created {output_path}")

def create_icns_file(img, output_path):
    """Create macOS .icns file"""
    # For .icns, we need specific sizes
    sizes = {
        'icon_16x16': 16,
        'icon_16x16@2x': 32,
        'icon_32x32': 32,
        'icon_32x32@2x': 64,
        'icon_128x128': 128,
        'icon_128x128@2x': 256,
        'icon_256x256': 256,
        'icon_256x256@2x': 512,
        'icon_512x512': 512,
        'icon_512x512@2x': 1024,
    }
    
    # Create temporary directory for icon files
    temp_dir = Path('temp_icns')
    temp_dir.mkdir(exist_ok=True)
    
    iconset_dir = temp_dir / 'icon.iconset'
    iconset_dir.mkdir(exist_ok=True)
    
    # Create each size
    for name, size in sizes.items():
        if size <= 512:
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
        else:
            # For 1024, we need to upscale
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
        
        resized.save(iconset_dir / f'{name}.png')
    
    # Use iconutil to create .icns (macOS only)
    import subprocess
    import platform
    
    if platform.system() == 'Darwin':
        try:
            subprocess.run(['iconutil', '-c', 'icns', '-o', output_path, str(iconset_dir)], 
                          check=True)
            print(f"Created {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create .icns file: {e}")
            print("You may need to manually convert the PNG to ICNS")
    else:
        print(f"Cannot create .icns file on {platform.system()}. Please convert on macOS.")
    
    # Clean up temporary files
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

def main():
    # Ensure assets directory exists
    assets_dir = Path('assets')
    assets_dir.mkdir(exist_ok=True)
    
    # Create the main icon at high resolution
    print("Creating underwater enhancer icon...")
    icon = create_underwater_icon(512)
    
    # Save as PNG
    png_path = assets_dir / 'icon.png'
    icon.save(png_path, 'PNG')
    print(f"Created {png_path}")
    
    # Create Windows .ico file
    ico_path = assets_dir / 'icon.ico'
    create_ico_file(icon, ico_path)
    
    # Create macOS .icns file
    icns_path = assets_dir / 'icon.icns'
    create_icns_file(icon, icns_path)
    
    print("\nIcon creation complete!")
    print("Files created:")
    print(f"  - {png_path} (PNG format)")
    print(f"  - {ico_path} (Windows format)")
    print(f"  - {icns_path} (macOS format)")

if __name__ == '__main__':
    main()