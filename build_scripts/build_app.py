#!/usr/bin/env python3
"""
Build script for creating platform-specific executables
"""
import sys
import platform
import subprocess
import shutil
from pathlib import Path

def compile_gpr_tools():
    """Compile gpr_tools for current platform"""
    print("Compiling gpr_tools...")
    
    system = platform.system().lower()
    
    # Run the compile script
    if system == 'windows':
        # Windows compilation (requires Visual Studio)
        compile_script = Path('build_scripts/compile_gpr_tools.bat')
        if compile_script.exists():
            subprocess.run([str(compile_script)], check=True, shell=True)
        else:
            print("Warning: Windows compile script not found")
            return False
    else:
        # Unix compilation
        compile_script = Path('build_scripts/compile_gpr_tools.sh')
        if compile_script.exists():
            subprocess.run(['bash', str(compile_script)], check=True)
        else:
            print("Warning: Unix compile script not found")
            return False
    
    print("[OK] gpr_tools compiled successfully")
    return True

def verify_binary():
    """Verify the gpr_tools binary exists"""
    system = platform.system().lower()
    
    if system == 'windows':
        binary_path = Path('binaries/win32/gpr_tools.exe')
    elif system == 'darwin':
        binary_path = Path('binaries/darwin/gpr_tools')
    else:
        binary_path = Path('binaries/linux/gpr_tools')
    
    if not binary_path.exists():
        print(f"[X] Binary not found at {binary_path}")
        print("Please compile gpr_tools first:")
        print("  ./build_scripts/compile_gpr_tools.sh")
        return False
    
    print(f"[OK] Found binary at {binary_path}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    
    # Check if requirements file exists
    req_file = Path('requirements_gui.txt')
    if not req_file.exists():
        print("Creating requirements_gui.txt...")
        create_requirements_file()
    
    # Install dependencies
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-r', 'requirements_gui.txt'
    ], check=True)
    
    print("[OK] Dependencies installed successfully")

def create_requirements_file():
    """Create requirements_gui.txt if it doesn't exist"""
    requirements = """# GUI Framework
customtkinter==5.2.0
darkdetect==0.8.0

# Image Processing
Pillow>=10.0.0
opencv-python>=4.8.0
scikit-image>=0.21.0

# ML Dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0

# Utilities
PyYAML>=6.0
tqdm>=4.65.0

# Packaging
pyinstaller>=6.0.0
"""
    
    with open('requirements_gui.txt', 'w') as f:
        f.write(requirements)

def build_executable():
    """Build the executable using PyInstaller"""
    print("\nBuilding executable with PyInstaller...")
    
    # Clean previous builds
    for dir_name in ['build', 'dist']:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"Cleaning {dir_name}/...")
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error cleaning {dir_name}: {e}")
                return False
    
    # Check if spec file exists
    spec_file = Path('pyinstaller.spec')
    if not spec_file.exists():
        print("Error: pyinstaller.spec not found")
        return False
    
    # Run PyInstaller
    try:
        subprocess.run([
            sys.executable, '-m', 'PyInstaller',
            'pyinstaller.spec',
            '--clean',
            '--noconfirm'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"PyInstaller failed: {e}")
        return False
    
    print("[OK] Executable built successfully")
    
    # Show output location
    system = platform.system().lower()
    if system == 'darwin':
        print(f"\nApplication bundle created: dist/UnderwaterEnhancer.app")
        print("To run: open dist/UnderwaterEnhancer.app")
    elif system == 'windows':
        print(f"\nExecutable created: dist/UnderwaterEnhancer.exe")
        print("To run: dist\\UnderwaterEnhancer.exe")
    else:
        print(f"\nExecutable created: dist/UnderwaterEnhancer")
        print("To run: ./dist/UnderwaterEnhancer")
    
    return True

def create_icons():
    """Create placeholder icons if they don't exist"""
    assets_dir = Path('assets')
    assets_dir.mkdir(exist_ok=True)
    
    # Create a simple placeholder icon using PIL if available
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple blue water wave icon
        size = 256
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw water waves
        draw.ellipse([20, 80, 236, 176], fill=(30, 144, 255))
        draw.ellipse([40, 120, 216, 216], fill=(100, 180, 255))
        
        # Save as PNG
        png_path = assets_dir / 'icon.png'
        img.save(png_path)
        print(f"Created icon: {png_path}")
        
        # Convert to ICO for Windows
        if platform.system().lower() == 'windows':
            ico_path = assets_dir / 'icon.ico'
            img.save(ico_path, format='ICO')
            print(f"Created icon: {ico_path}")
        
    except ImportError:
        print("PIL not available, skipping icon creation")

def main():
    print("="*50)
    print("Underwater Enhancer Build Script")
    print("="*50)
    
    # Step 1: Create icons if needed
    create_icons()
    
    # Step 2: Verify or compile gpr_tools
    if not verify_binary():
        print("\nWarning: gpr_tools binary not found.")
        print("The application will build but GPR support will not work.")
        print("To add GPR support, compile gpr_tools first.")
        response = input("\nContinue without GPR support? (y/n): ")
        if response.lower() != 'y':
            print("Build cancelled.")
            sys.exit(1)
    
    # Step 3: Install dependencies
    try:
        install_dependencies()
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)
    
    # Step 4: Build executable
    if not build_executable():
        print("Build failed!")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("Build completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()