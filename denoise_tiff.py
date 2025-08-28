#!/usr/bin/env python3
"""
Denoising script for TIFF images processed by the ML model
Applies various denoising algorithms to enhance image quality
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from skimage import restoration, img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TiffDenoiser:
    """Class for denoising TIFF images with various algorithms"""
    
    SUPPORTED_METHODS = [
        'bilateral',
        'nlmeans',
        'gaussian',
        'median',
        'tv_chambolle',
        'wavelet',
        'bm3d_approximation'
    ]
    
    def __init__(self, method: str = 'nlmeans', preserve_range: bool = True):
        """
        Initialize denoiser with specified method
        
        Args:
            method: Denoising method to use
            preserve_range: Whether to preserve the original value range
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")
        
        self.method = method
        self.preserve_range = preserve_range
        logger.info(f"Initialized denoiser with method: {method}")
    
    def denoise_bilateral(self, image: np.ndarray, d: int = 9, 
                          sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """Apply bilateral filter for edge-preserving denoising"""
        if image.dtype != np.uint8:
            image_uint8 = img_as_ubyte(image)
        else:
            image_uint8 = image
        
        if len(image_uint8.shape) == 3:
            denoised = cv2.bilateralFilter(image_uint8, d, sigma_color, sigma_space)
        else:
            denoised = cv2.bilateralFilter(image_uint8, d, sigma_color, sigma_space)
        
        if self.preserve_range and image.dtype == np.float32:
            return img_as_float(denoised)
        return denoised
    
    def denoise_nlmeans(self, image: np.ndarray, h: float = 0.1, 
                       patch_size: int = 7, patch_distance: int = 11) -> np.ndarray:
        """Apply Non-Local Means denoising"""
        image_float = img_as_float(image)
        
        if len(image_float.shape) == 3:
            # Process each channel separately for color images
            denoised = np.zeros_like(image_float)
            for i in range(image_float.shape[2]):
                denoised[:, :, i] = restoration.denoise_nl_means(
                    image_float[:, :, i], 
                    h=h,
                    patch_size=patch_size,
                    patch_distance=patch_distance,
                    preserve_range=self.preserve_range
                )
        else:
            denoised = restoration.denoise_nl_means(
                image_float,
                h=h,
                patch_size=patch_size,
                patch_distance=patch_distance,
                preserve_range=self.preserve_range
            )
        
        if image.dtype == np.uint8:
            return img_as_ubyte(denoised)
        elif image.dtype == np.uint16:
            return (denoised * 65535).astype(np.uint16)
        return denoised
    
    def denoise_gaussian(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur for basic denoising"""
        kernel_size = int(2 * np.ceil(2 * sigma) + 1)
        
        if len(image.shape) == 3:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        else:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def denoise_median(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply median filter for salt-and-pepper noise removal"""
        if image.dtype != np.uint8:
            image_uint8 = img_as_ubyte(img_as_float(image))
        else:
            image_uint8 = image
        
        denoised = cv2.medianBlur(image_uint8, kernel_size)
        
        if self.preserve_range and image.dtype == np.float32:
            return img_as_float(denoised)
        elif image.dtype == np.uint16:
            return (img_as_float(denoised) * 65535).astype(np.uint16)
        return denoised
    
    def denoise_tv_chambolle(self, image: np.ndarray, weight: float = 0.1) -> np.ndarray:
        """Apply Total Variation denoising (Chambolle algorithm)"""
        image_float = img_as_float(image)
        
        if len(image_float.shape) == 3:
            # For color images, use multichannel
            denoised = restoration.denoise_tv_chambolle(
                image_float, 
                weight=weight,
                channel_axis=-1
            )
        else:
            denoised = restoration.denoise_tv_chambolle(
                image_float,
                weight=weight
            )
        
        if image.dtype == np.uint8:
            return img_as_ubyte(denoised)
        elif image.dtype == np.uint16:
            return (denoised * 65535).astype(np.uint16)
        return denoised
    
    def denoise_wavelet(self, image: np.ndarray, wavelet: str = 'db1', 
                       sigma: Optional[float] = None) -> np.ndarray:
        """Apply wavelet denoising"""
        image_float = img_as_float(image)
        
        if len(image_float.shape) == 3:
            denoised = restoration.denoise_wavelet(
                image_float,
                channel_axis=-1,
                convert2ycbcr=True,
                wavelet=wavelet,
                sigma=sigma,
                rescale_sigma=True
            )
        else:
            denoised = restoration.denoise_wavelet(
                image_float,
                wavelet=wavelet,
                sigma=sigma,
                rescale_sigma=True
            )
        
        if image.dtype == np.uint8:
            return img_as_ubyte(denoised)
        elif image.dtype == np.uint16:
            return (denoised * 65535).astype(np.uint16)
        return denoised
    
    def denoise_bm3d_approximation(self, image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Simple BM3D approximation using wavelet and bilateral filtering
        (Full BM3D requires additional dependencies)
        """
        # First pass: wavelet denoising
        denoised = self.denoise_wavelet(image, sigma=sigma)
        
        # Second pass: bilateral filter for edge preservation
        if denoised.dtype != np.uint8:
            denoised_uint8 = img_as_ubyte(img_as_float(denoised))
        else:
            denoised_uint8 = denoised
        
        final = cv2.bilateralFilter(denoised_uint8, 5, 50, 50)
        
        if image.dtype == np.float32:
            return img_as_float(final)
        elif image.dtype == np.uint16:
            return (img_as_float(final) * 65535).astype(np.uint16)
        return final
    
    def denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply selected denoising method to image
        
        Args:
            image: Input image array
            **kwargs: Method-specific parameters
        
        Returns:
            Denoised image array
        """
        if self.method == 'bilateral':
            return self.denoise_bilateral(image, **kwargs)
        elif self.method == 'nlmeans':
            return self.denoise_nlmeans(image, **kwargs)
        elif self.method == 'gaussian':
            return self.denoise_gaussian(image, **kwargs)
        elif self.method == 'median':
            return self.denoise_median(image, **kwargs)
        elif self.method == 'tv_chambolle':
            return self.denoise_tv_chambolle(image, **kwargs)
        elif self.method == 'wavelet':
            return self.denoise_wavelet(image, **kwargs)
        elif self.method == 'bm3d_approximation':
            return self.denoise_bm3d_approximation(image, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def process_tiff(self, input_path: Path, output_path: Path, **kwargs) -> Tuple[float, float]:
        """
        Process a single TIFF file
        
        Args:
            input_path: Path to input TIFF file
            output_path: Path to save denoised TIFF
            **kwargs: Method-specific parameters
        
        Returns:
            Tuple of (PSNR, SSIM) metrics comparing input and output
        """
        # Load TIFF image
        image = Image.open(input_path)
        image_array = np.array(image)
        original_dtype = image_array.dtype
        
        logger.info(f"Processing {input_path.name} - Shape: {image_array.shape}, Dtype: {original_dtype}")
        
        # Apply denoising
        denoised_array = self.denoise(image_array, **kwargs)
        
        # Ensure output has same dtype as input
        if denoised_array.dtype != original_dtype:
            if original_dtype == np.uint16:
                if denoised_array.dtype == np.float32 or denoised_array.dtype == np.float64:
                    denoised_array = (denoised_array * 65535).astype(np.uint16)
                else:
                    denoised_array = denoised_array.astype(np.uint16)
            else:
                denoised_array = denoised_array.astype(original_dtype)
        
        # Save denoised image as TIFF
        output_path.parent.mkdir(parents=True, exist_ok=True)
        denoised_image = Image.fromarray(denoised_array)
        
        # Preserve metadata if possible
        if hasattr(image, 'info'):
            denoised_image.save(output_path, 'TIFF', **image.info)
        else:
            denoised_image.save(output_path, 'TIFF')
        
        # Calculate metrics
        psnr = peak_signal_noise_ratio(image_array, denoised_array, data_range=denoised_array.max() - denoised_array.min())
        
        # SSIM calculation
        if len(image_array.shape) == 3:
            ssim = structural_similarity(image_array, denoised_array, channel_axis=2, data_range=denoised_array.max() - denoised_array.min())
        else:
            ssim = structural_similarity(image_array, denoised_array, data_range=denoised_array.max() - denoised_array.min())
        
        logger.info(f"Saved denoised image to {output_path} - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        
        return psnr, ssim
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         pattern: str = "*.tif*", **kwargs) -> None:
        """
        Process all TIFF files in a directory
        
        Args:
            input_dir: Input directory containing TIFF files
            output_dir: Output directory for denoised files
            pattern: File pattern to match (default: *.tif*)
            **kwargs: Method-specific parameters
        """
        # Find all TIFF files
        tiff_files = list(input_dir.glob(pattern))
        tiff_files.extend(list(input_dir.glob(pattern.upper())))
        tiff_files = list(set(tiff_files))  # Remove duplicates
        
        if not tiff_files:
            logger.warning(f"No TIFF files found in {input_dir} with pattern {pattern}")
            return
        
        logger.info(f"Found {len(tiff_files)} TIFF files to process")
        
        # Process each file
        metrics = []
        for tiff_path in tqdm(tiff_files, desc="Denoising TIFF files"):
            output_path = output_dir / f"{tiff_path.stem}_denoised{tiff_path.suffix}"
            psnr, ssim = self.process_tiff(tiff_path, output_path, **kwargs)
            metrics.append((psnr, ssim))
        
        # Report average metrics
        if metrics:
            avg_psnr = np.mean([m[0] for m in metrics])
            avg_ssim = np.mean([m[1] for m in metrics])
            logger.info(f"\nAverage metrics - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Denoise TIFF images with various algorithms")
    parser.add_argument('input', type=str, help='Input TIFF file or directory')
    parser.add_argument('--output', type=str, default='denoised_output',
                        help='Output directory or file path')
    parser.add_argument('--method', type=str, default='nlmeans',
                        choices=TiffDenoiser.SUPPORTED_METHODS,
                        help='Denoising method to use')
    parser.add_argument('--preserve-range', action='store_true',
                        help='Preserve original value range')
    
    # Method-specific parameters
    parser.add_argument('--bilateral-d', type=int, default=9,
                        help='Bilateral filter diameter')
    parser.add_argument('--bilateral-sigma-color', type=float, default=75,
                        help='Bilateral filter sigma for color')
    parser.add_argument('--bilateral-sigma-space', type=float, default=75,
                        help='Bilateral filter sigma for space')
    
    parser.add_argument('--nlmeans-h', type=float, default=0.1,
                        help='NL-Means filter strength')
    parser.add_argument('--nlmeans-patch-size', type=int, default=7,
                        help='NL-Means patch size')
    parser.add_argument('--nlmeans-patch-distance', type=int, default=11,
                        help='NL-Means patch search distance')
    
    parser.add_argument('--gaussian-sigma', type=float, default=1.0,
                        help='Gaussian blur sigma')
    
    parser.add_argument('--median-kernel', type=int, default=5,
                        help='Median filter kernel size')
    
    parser.add_argument('--tv-weight', type=float, default=0.1,
                        help='Total Variation weight parameter')
    
    parser.add_argument('--wavelet-type', type=str, default='db1',
                        help='Wavelet type for denoising')
    parser.add_argument('--wavelet-sigma', type=float, default=None,
                        help='Wavelet denoising sigma (auto if not specified)')
    
    parser.add_argument('--bm3d-sigma', type=float, default=0.1,
                        help='BM3D approximation sigma')
    
    args = parser.parse_args()
    
    # Initialize denoiser
    denoiser = TiffDenoiser(method=args.method, preserve_range=args.preserve_range)
    
    # Prepare method-specific kwargs
    kwargs = {}
    if args.method == 'bilateral':
        kwargs = {
            'd': args.bilateral_d,
            'sigma_color': args.bilateral_sigma_color,
            'sigma_space': args.bilateral_sigma_space
        }
    elif args.method == 'nlmeans':
        kwargs = {
            'h': args.nlmeans_h,
            'patch_size': args.nlmeans_patch_size,
            'patch_distance': args.nlmeans_patch_distance
        }
    elif args.method == 'gaussian':
        kwargs = {'sigma': args.gaussian_sigma}
    elif args.method == 'median':
        kwargs = {'kernel_size': args.median_kernel}
    elif args.method == 'tv_chambolle':
        kwargs = {'weight': args.tv_weight}
    elif args.method == 'wavelet':
        kwargs = {
            'wavelet': args.wavelet_type,
            'sigma': args.wavelet_sigma
        }
    elif args.method == 'bm3d_approximation':
        kwargs = {'sigma': args.bm3d_sigma}
    
    # Process input
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Process single file
        if output_path.suffix not in ['.tif', '.tiff']:
            output_file = output_path / f"{input_path.stem}_denoised{input_path.suffix}"
        else:
            output_file = output_path
        
        denoiser.process_tiff(input_path, output_file, **kwargs)
    elif input_path.is_dir():
        # Process directory
        denoiser.process_directory(input_path, output_path, **kwargs)
    else:
        logger.error(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()