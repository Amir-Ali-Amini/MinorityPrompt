#!/usr/bin/env python
"""
Setup script for MinorityPrompt Demographic Evaluation.

This script:
1. Checks for required Python packages
2. Downloads required models (dlib shape predictor, FairFace)
3. Verifies the installation
4. Runs a quick test

Usage:
    python -m minority_gen.evaluation.setup
    
    # Or directly:
    python setup.py
    
    # With custom model directory:
    python setup.py --model-dir /path/to/models
    
    # Skip verification test:
    python setup.py --skip-test
"""

import os
import sys
import subprocess
import urllib.request
import bz2
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


# =============================================================================
# Configuration
# =============================================================================

REQUIRED_PACKAGES = {
    # package_name: (import_name, pip_name, required)
    'torch': ('torch', 'torch', True),
    'torchvision': ('torchvision', 'torchvision', True),
    'numpy': ('numpy', 'numpy', True),
    'pandas': ('pandas', 'pandas', True),
    'dlib': ('dlib', 'dlib', True),
    'gdown': ('gdown', 'gdown', False),  # For downloading from Google Drive
    'PIL': ('PIL', 'Pillow', False),  # Optional but useful
}

MODELS = {
    'dlib_shape_predictor': {
        'filename': 'shape_predictor_5_face_landmarks.dat',
        'url': 'http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2',
        'compressed': True,
        'description': 'dlib 5-point face landmark predictor',
    },
    'fairface': {
        'filename': 'res34_fair_align_multi_7_20190809.pt',
        'gdrive_id': '1F-4AUDXrTBTLqmRojxoeWi_VKFp6nv4v',
        'compressed': False,
        'description': 'FairFace ResNet34 demographic classifier',
    },
}


# =============================================================================
# Utility Functions
# =============================================================================

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def print_status(name: str, status: bool, message: str = ""):
    """Print status with checkmark or X."""
    icon = "✓" if status else "✗"
    color_start = "\033[92m" if status else "\033[91m"  # Green or Red
    color_end = "\033[0m"
    msg = f" - {message}" if message else ""
    print(f"  {color_start}{icon}{color_end} {name}{msg}")


def get_package_version(import_name: str) -> Optional[str]:
    """Get the version of an installed package."""
    try:
        module = __import__(import_name)
        return getattr(module, '__version__', 'unknown')
    except ImportError:
        return None


# =============================================================================
# Dependency Checking
# =============================================================================

def check_dependencies() -> Tuple[List[str], List[str]]:
    """
    Check for required Python packages.
    
    Returns:
        Tuple of (installed_packages, missing_packages)
    """
    print_header("Checking Python Dependencies")
    
    installed = []
    missing = []
    
    for name, (import_name, pip_name, required) in REQUIRED_PACKAGES.items():
        version = get_package_version(import_name)
        
        if version:
            print_status(name, True, f"v{version}")
            installed.append(name)
        else:
            req_str = "(required)" if required else "(optional)"
            print_status(name, False, f"not installed {req_str}")
            if required:
                missing.append(pip_name)
    
    return installed, missing


def install_packages(packages: List[str], auto_install: bool = False) -> bool:
    """
    Install missing packages.
    
    Args:
        packages: List of package names to install
        auto_install: If True, install without asking
        
    Returns:
        True if all packages installed successfully
    """
    if not packages:
        return True
    
    print(f"\nMissing packages: {', '.join(packages)}")
    
    if not auto_install:
        response = input("Install missing packages? [y/N]: ").strip().lower()
        if response != 'y':
            print("Skipping package installation.")
            return False
    
    print("\nInstalling packages...")
    
    for package in packages:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print_status(package, True, "installed")
        except subprocess.CalledProcessError as e:
            print_status(package, False, f"failed to install: {e}")
            
            # Special handling for dlib
            if package == 'dlib':
                print("\n  Note: dlib requires CMake and C++ compiler.")
                print("  On Ubuntu/Debian: sudo apt-get install cmake libboost-all-dev")
                print("  On macOS: brew install cmake boost")
                print("  On Windows: Install Visual Studio Build Tools")
            return False
    
    return True


# =============================================================================
# Model Downloading
# =============================================================================

def download_with_progress(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress indication."""
    
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, block_num * block_size * 100 // total_size)
            downloaded = block_num * block_size / (1024 * 1024)
            total = total_size / (1024 * 1024)
            print(f"\r  {desc}: {percent}% ({downloaded:.1f}/{total:.1f} MB)", end="", flush=True)
    
    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # Newline after progress


def download_dlib_model(model_dir: Path) -> bool:
    """Download dlib shape predictor model."""
    config = MODELS['dlib_shape_predictor']
    dest = model_dir / config['filename']
    
    if dest.exists():
        print_status(config['filename'], True, "already exists")
        return True
    
    print(f"\n  Downloading {config['description']}...")
    
    try:
        # Download compressed file
        bz2_path = model_dir / f"{config['filename']}.bz2"
        download_with_progress(config['url'], bz2_path, "Downloading")
        
        # Decompress
        print("  Decompressing...", end=" ", flush=True)
        with bz2.open(bz2_path, 'rb') as f_in:
            with open(dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove compressed file
        bz2_path.unlink()
        
        print("done")
        print_status(config['filename'], True, f"saved to {dest}")
        return True
        
    except Exception as e:
        print_status(config['filename'], False, f"failed: {e}")
        return False


def download_fairface_model(model_dir: Path) -> bool:
    """Download FairFace model from Google Drive."""
    config = MODELS['fairface']
    dest = model_dir / config['filename']
    
    if dest.exists():
        print_status(config['filename'], True, "already exists")
        return True
    
    print(f"\n  Downloading {config['description']}...")
    
    try:
        # Try using gdown
        try:
            import gdown
        except ImportError:
            print("  Installing gdown for Google Drive download...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "gdown"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import gdown
        
        url = f"https://drive.google.com/uc?id={config['gdrive_id']}"
        gdown.download(url, str(dest), quiet=False)
        
        if dest.exists():
            print_status(config['filename'], True, f"saved to {dest}")
            return True
        else:
            raise Exception("Download completed but file not found")
            
    except Exception as e:
        print_status(config['filename'], False, f"failed: {e}")
        print("\n  Manual download instructions:")
        print("  1. Go to: https://github.com/joojs/fairface")
        print("  2. Download: res34_fair_align_multi_7_20190809.pt")
        print(f"  3. Place in: {model_dir}/")
        return False


def download_models(model_dir: Path) -> Dict[str, bool]:
    """
    Download all required models.
    
    Returns:
        Dictionary mapping model name to success status
    """
    print_header("Downloading Models")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model directory: {model_dir}")
    
    results = {}
    
    # Download dlib model
    results['dlib'] = download_dlib_model(model_dir)
    
    # Download FairFace model
    results['fairface'] = download_fairface_model(model_dir)
    
    return results


# =============================================================================
# Verification
# =============================================================================

def verify_installation(model_dir: Path) -> bool:
    """
    Verify that everything is installed correctly.
    
    Returns:
        True if verification passed
    """
    print_header("Verifying Installation")
    
    all_ok = True
    
    # Check models exist
    for name, config in MODELS.items():
        path = model_dir / config['filename']
        exists = path.exists()
        size = path.stat().st_size / (1024 * 1024) if exists else 0
        
        if exists:
            print_status(config['filename'], True, f"{size:.1f} MB")
        else:
            print_status(config['filename'], False, "not found")
            all_ok = False
    
    # Try importing evaluation module
    print("\n  Testing imports...")
    
    try:
        from minority_gen.evaluation import (
            DemographicEvaluator,
            FaceDetector,
            DemographicPredictor,
            calculate_all_metrics,
        )
        print_status("minority_gen.evaluation", True, "imported successfully")
    except ImportError as e:
        print_status("minority_gen.evaluation", False, f"import error: {e}")
        all_ok = False
    
    return all_ok


def run_quick_test(model_dir: Path) -> bool:
    """
    Run a quick test to verify everything works.
    
    Returns:
        True if test passed
    """
    print_header("Running Quick Test")
    
    try:
        import numpy as np
        import pandas as pd
        from minority_gen.evaluation import (
            DemographicEvaluator,
            calculate_all_metrics,
            RACE_LABELS_7,
            GENDER_LABELS,
            AGE_LABELS,
        )
        
        # Test 1: Metrics calculation (no GPU needed)
        print("\n  Test 1: Metrics calculation...")
        
        # Create dummy data
        np.random.seed(42)
        n_samples = 50
        
        df = pd.DataFrame({
            'race': np.random.choice(RACE_LABELS_7, n_samples),
            'gender': np.random.choice(GENDER_LABELS, n_samples),
            'age': np.random.choice(AGE_LABELS, n_samples),
        })
        
        metrics = calculate_all_metrics(df)
        
        assert 'race' in metrics.bias_w
        assert 'gender' in metrics.bias_w
        assert metrics.ens['race'] > 0
        
        print_status("Metrics calculation", True)
        
        # Test 2: Model loading (if models exist)
        dlib_path = model_dir / MODELS['dlib_shape_predictor']['filename']
        fairface_path = model_dir / MODELS['fairface']['filename']
        
        if dlib_path.exists() and fairface_path.exists():
            print("\n  Test 2: Model loading...")
            
            evaluator = DemographicEvaluator(
                face_model_path=str(dlib_path),
                fairface_model_path=str(fairface_path),
            )
            
            # Check models loaded
            _ = evaluator.face_detector  # Trigger lazy loading
            _ = evaluator.demographic_predictor
            
            print_status("Model loading", True)
        else:
            print("\n  Test 2: Model loading... skipped (models not found)")
        
        print("\n  All tests passed!")
        return True
        
    except Exception as e:
        print_status("Test", False, f"failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Setup script for MinorityPrompt Demographic Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full setup with all checks and downloads:
    python setup.py
    
    # Custom model directory:
    python setup.py --model-dir /path/to/models
    
    # Auto-install missing packages:
    python setup.py --auto-install
    
    # Skip the verification test:
    python setup.py --skip-test
    
    # Only check dependencies (no downloads):
    python setup.py --check-only
        """
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to store downloaded models (default: models/)'
    )
    
    parser.add_argument(
        '--auto-install',
        action='store_true',
        help='Automatically install missing packages without asking'
    )
    
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip the verification test'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check dependencies, do not download models'
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir).absolute()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     MinorityPrompt Demographic Evaluation - Setup Script      ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check dependencies
    installed, missing = check_dependencies()
    
    if missing:
        success = install_packages(missing, auto_install=args.auto_install)
        if not success:
            print("\n⚠ Some required packages are missing. Please install them manually:")
            print(f"  pip install {' '.join(missing)}")
            if not args.check_only:
                sys.exit(1)
    
    if args.check_only:
        print("\n✓ Dependency check complete (--check-only mode)")
        return
    
    # Step 2: Download models
    model_results = download_models(model_dir)
    
    if not all(model_results.values()):
        print("\n⚠ Some models failed to download. See instructions above.")
    
    # Step 3: Verify installation
    verify_ok = verify_installation(model_dir)
    
    # Step 4: Run quick test
    if not args.skip_test and verify_ok:
        test_ok = run_quick_test(model_dir)
    else:
        test_ok = True
    
    # Summary
    print_header("Setup Summary")
    
    all_ok = len(missing) == 0 and all(model_results.values()) and verify_ok and test_ok
    
    if all_ok:
        print("""
✓ Setup completed successfully!

Usage:
    from minority_gen.evaluation import DemographicEvaluator
    
    evaluator = DemographicEvaluator(
        face_model_path="{dlib_path}",
        fairface_model_path="{fairface_path}",
    )
    
    result = evaluator.evaluate_directory("your_images/", tag="test")
    print(result.metrics.summary())
        """.format(
            dlib_path=model_dir / MODELS['dlib_shape_predictor']['filename'],
            fairface_path=model_dir / MODELS['fairface']['filename'],
        ))
    else:
        print("\n⚠ Setup completed with some issues. Please check the messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
