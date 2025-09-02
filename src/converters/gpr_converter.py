"""
GPR to TIFF converter using bundled gpr_tools binary
"""
import sys
import platform
import subprocess
import tempfile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GPRConverter:
    """Handles GPR to TIFF conversion using bundled gpr_tools"""
    
    @staticmethod
    def get_gpr_tools_path():
        """Get path to bundled gpr_tools binary for current platform"""
        if getattr(sys, 'frozen', False):
            # Running in PyInstaller bundle
            base_path = Path(sys._MEIPASS) / 'binaries'
        else:
            # Running in development mode
            base_path = Path(__file__).parent.parent.parent / 'binaries'
        
        system = platform.system().lower()
        if system == 'windows':
            binary_name = 'gpr_tools.exe'
            binary_path = base_path / 'win32' / binary_name
        elif system == 'darwin':
            binary_path = base_path / 'darwin' / 'gpr_tools'
        else:  # linux
            binary_path = base_path / 'linux' / 'gpr_tools'
        
        if not binary_path.exists():
            raise FileNotFoundError(f"gpr_tools binary not found at {binary_path}")
        
        # Ensure binary is executable on Unix systems
        if system != 'windows':
            import stat
            binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC)
        
        return binary_path
    
    @classmethod
    def convert(cls, gpr_path: Path, output_path: Path = None, 
                keep_temp: bool = False) -> Path:
        """
        Convert GPR file to TIFF
        
        Args:
            gpr_path: Path to input GPR file
            output_path: Optional output path (auto-generated if None)
            keep_temp: Keep temporary files for debugging
        
        Returns:
            Path to converted TIFF file
        """
        gpr_tools = cls.get_gpr_tools_path()
        
        if output_path is None:
            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
                output_path = Path(tmp.name)
        
        # Build command - gpr_tools uses positional arguments
        cmd = [str(gpr_tools), str(gpr_path), str(output_path)]
        
        try:
            # Run conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30  # 30 second timeout
            )
            
            logger.info(f"Successfully converted {gpr_path.name} to TIFF")
            
            if not output_path.exists():
                raise RuntimeError(f"Conversion succeeded but output file not found: {output_path}")
            
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error(f"GPR conversion timed out for {gpr_path}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"GPR conversion failed: {e.stderr}")
            raise
        finally:
            # Cleanup temp files if needed
            if not keep_temp and output_path and output_path.exists() and output_path.parent == Path(tempfile.gettempdir()):
                try:
                    output_path.unlink()
                except:
                    pass
    
    @classmethod
    def is_gpr_file(cls, file_path: Path) -> bool:
        """Check if file is a GPR file"""
        return file_path.suffix.lower() in ['.gpr']
    
    @classmethod
    def batch_convert(cls, gpr_files: list[Path], output_dir: Path = None,
                     progress_callback=None) -> list[Path]:
        """
        Convert multiple GPR files to TIFF
        
        Args:
            gpr_files: List of GPR file paths
            output_dir: Directory for output files
            progress_callback: Optional callback(current, total, filename)
        
        Returns:
            List of converted TIFF paths
        """
        converted_files = []
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, gpr_path in enumerate(gpr_files):
            if progress_callback:
                progress_callback(i, len(gpr_files), gpr_path.name)
            
            if output_dir:
                output_path = output_dir / f"{gpr_path.stem}.tiff"
            else:
                output_path = None
            
            try:
                tiff_path = cls.convert(gpr_path, output_path)
                converted_files.append(tiff_path)
            except Exception as e:
                logger.error(f"Failed to convert {gpr_path}: {e}")
                # Continue with other files
        
        return converted_files