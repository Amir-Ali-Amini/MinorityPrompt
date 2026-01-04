"""
Main evaluator class for demographic bias evaluation.

This module provides a high-level interface for evaluating demographic
bias and diversity in generated images.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import torch
    from torchvision.utils import save_image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .constants import ATTRIBUTE_COMBINATIONS, ATTRIBUTES_DICT
from .face_detection import FaceDetector, DetectedFace
from .demographic_predictor import DemographicPredictor, DemographicPrediction
from .metrics import (
    BiasMetrics,
    calculate_all_metrics,
    calculate_bias_w,
    calculate_bias_p,
    calculate_ens,
    calculate_kl_divergence,
    compare_metrics,
)


@dataclass
class EvaluationResult:
    """Container for complete evaluation results."""
    
    # Raw predictions
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Computed metrics
    metrics: Optional[BiasMetrics] = None
    
    # Metadata
    num_images: int = 0
    num_faces: int = 0
    images_with_faces: int = 0
    images_without_faces: int = 0
    
    # Timing
    detection_time: float = 0.0
    prediction_time: float = 0.0
    metrics_time: float = 0.0
    
    # Tag for identification
    tag: str = ""
    
    def save(self, output_dir: Union[str, Path], prefix: str = ""):
        """
        Save all results to directory.
        
        Args:
            output_dir: Directory to save results
            prefix: Optional prefix for filenames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{prefix}_" if prefix else ""
        
        # Save predictions DataFrame
        self.predictions_df.to_csv(output_dir / f"{prefix}predictions.csv", index=False)
        
        # Save metrics
        if self.metrics:
            # Summary
            self.metrics.summary().to_csv(output_dir / f"{prefix}metrics_summary.csv", index=False)
            
            # Bias-W
            pd.DataFrame([self.metrics.bias_w]).T.reset_index().rename(
                columns={'index': 'Attribute_Group', 0: 'Bias-W'}
            ).to_csv(output_dir / f"{prefix}bias_w.csv", index=False)
            
            # Bias-P
            if len(self.metrics.bias_p) > 0:
                self.metrics.bias_p.to_csv(output_dir / f"{prefix}bias_p.csv", index=False)
            
            # ENS
            pd.DataFrame([self.metrics.ens]).T.reset_index().rename(
                columns={'index': 'Attribute', 0: 'ENS'}
            ).to_csv(output_dir / f"{prefix}ens.csv", index=False)
            
            # KL Divergence
            pd.DataFrame([self.metrics.kl_divergence]).T.reset_index().rename(
                columns={'index': 'Attribute_Group', 0: 'KL_Divergence'}
            ).to_csv(output_dir / f"{prefix}kl_divergence.csv", index=False)
        
        # Save metadata
        metadata = {
            'tag': self.tag,
            'num_images': self.num_images,
            'num_faces': self.num_faces,
            'images_with_faces': self.images_with_faces,
            'images_without_faces': self.images_without_faces,
            'detection_time': self.detection_time,
            'prediction_time': self.prediction_time,
            'metrics_time': self.metrics_time,
            'timestamp': datetime.now().isoformat(),
        }
        with open(output_dir / f"{prefix}metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Results saved to: {output_dir}")
    
    def summary_dict(self) -> Dict[str, Any]:
        """Get summary as dictionary."""
        result = {
            'tag': self.tag,
            'num_images': self.num_images,
            'num_faces': self.num_faces,
            'face_detection_rate': self.images_with_faces / max(self.num_images, 1),
        }
        
        if self.metrics:
            result.update(self.metrics.to_dict())
        
        return result


class DemographicEvaluator:
    """
    High-level evaluator for demographic bias in generated images.
    
    This class provides a complete pipeline for:
    1. Detecting faces in images
    2. Predicting demographic attributes
    3. Computing bias and diversity metrics
    
    Example:
        evaluator = DemographicEvaluator(
            face_model_path="path/to/shape_predictor_5_face_landmarks.dat",
            fairface_model_path="path/to/res34_fair_align_multi_7_20190809.pt",
        )
        
        # Evaluate image directories
        result_baseline = evaluator.evaluate_directory("baseline_images/", tag="baseline")
        result_minority = evaluator.evaluate_directory("minority_images/", tag="minority")
        
        # Compare results
        comparison = evaluator.compare(result_baseline, result_minority)
        print(comparison)
    """
    
    def __init__(
        self,
        face_model_path: Optional[str] = None,
        fairface_model_path: Optional[str] = None,
        device: Optional[str] = None,
        save_detected_faces: bool = False,
    ):
        """
        Initialize the evaluator.
        
        Args:
            face_model_path: Path to dlib shape predictor model
            fairface_model_path: Path to FairFace model weights
            device: Device for inference ('cuda' or 'cpu')
            save_detected_faces: Whether to save detected face crops
        """
        self.save_detected_faces = save_detected_faces
        self.device = device
        
        # Lazy initialization of models
        self._face_detector = None
        self._demographic_predictor = None
        self._face_model_path = face_model_path
        self._fairface_model_path = fairface_model_path
    
    @property
    def face_detector(self) -> FaceDetector:
        """Lazy-load face detector."""
        if self._face_detector is None:
            self._face_detector = FaceDetector(
                landmark_model_path=self._face_model_path
            )
        return self._face_detector
    
    @property
    def demographic_predictor(self) -> DemographicPredictor:
        """Lazy-load demographic predictor."""
        if self._demographic_predictor is None:
            self._demographic_predictor = DemographicPredictor(
                model_path=self._fairface_model_path,
                device=self.device,
            )
        return self._demographic_predictor
    
    def evaluate_images(
        self,
        images: List[Union[str, Path, np.ndarray]],
        tag: str = "",
        output_dir: Optional[Union[str, Path]] = None,
        progress: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a list of images.
        
        Args:
            images: List of image paths or numpy arrays
            tag: Label for this evaluation
            output_dir: Optional directory to save intermediate results
            progress: Whether to show progress
            
        Returns:
            EvaluationResult containing predictions and metrics
        """
        import time
        
        result = EvaluationResult(tag=tag, num_images=len(images))
        
        # Step 1: Face Detection
        if progress:
            print(f"[{tag}] Detecting faces in {len(images)} images...")
        
        t0 = time.time()
        all_faces: List[DetectedFace] = []
        images_with_faces = 0
        images_without_faces = 0
        
        for i, image in enumerate(images):
            if progress and i % 100 == 0:
                print(f"  Processing image: {i}/{len(images)}")
            
            try:
                faces = self.face_detector.detect(image)
                all_faces.extend(faces)
                
                if len(faces) > 0:
                    images_with_faces += 1
                else:
                    images_without_faces += 1
                    if progress:
                        img_name = image if isinstance(image, str) else f"image_{i}"
                        print(f"  No face detected in: {img_name}")
            except Exception as e:
                images_without_faces += 1
                if progress:
                    print(f"  Error processing image {i}: {e}")
        
        result.detection_time = time.time() - t0
        result.num_faces = len(all_faces)
        result.images_with_faces = images_with_faces
        result.images_without_faces = images_without_faces
        
        if progress:
            print(f"  Detected {len(all_faces)} faces in {images_with_faces} images")
        
        # Save detected faces if requested
        if self.save_detected_faces and output_dir:
            faces_dir = Path(output_dir) / f"detected_faces_{tag}"
            self.face_detector.save_faces(all_faces, faces_dir)
        
        if len(all_faces) == 0:
            print(f"Warning: No faces detected. Cannot compute metrics.")
            return result
        
        # Step 2: Demographic Prediction
        if progress:
            print(f"[{tag}] Predicting demographics for {len(all_faces)} faces...")
        
        t0 = time.time()
        predictions = self.demographic_predictor.predict_faces(all_faces, progress=progress)
        result.prediction_time = time.time() - t0
        
        # Convert to DataFrame
        result.predictions_df = self.demographic_predictor.predictions_to_dataframe(predictions)
        
        # Add original image column for Bias-P calculation
        result.predictions_df['original_image'] = result.predictions_df['face_name_align'].apply(
            lambda x: os.path.basename(x).split('_face')[0] if isinstance(x, str) else str(x)
        )
        
        # Step 3: Compute Metrics
        if progress:
            print(f"[{tag}] Computing bias metrics...")
        
        t0 = time.time()
        result.metrics = calculate_all_metrics(result.predictions_df)
        result.metrics_time = time.time() - t0
        
        # Save results if output_dir provided
        if output_dir:
            result.save(output_dir, prefix=tag)
        
        return result
    
    def evaluate_directory(
        self,
        directory: Union[str, Path],
        tag: str = "",
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp'),
        output_dir: Optional[Union[str, Path]] = None,
        progress: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate all images in a directory.
        
        Args:
            directory: Path to directory containing images
            tag: Label for this evaluation
            extensions: Image file extensions to include
            output_dir: Optional directory to save results
            progress: Whether to show progress
            
        Returns:
            EvaluationResult containing predictions and metrics
        """
        directory = Path(directory)
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(set(image_paths))
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {directory}")
        
        if progress:
            print(f"Found {len(image_paths)} images in {directory}")
        
        return self.evaluate_images(
            images=[str(p) for p in image_paths],
            tag=tag,
            output_dir=output_dir,
            progress=progress,
        )
    
    def evaluate_tensors(
        self,
        tensors: List["torch.Tensor"],
        tag: str = "",
        output_dir: Optional[Union[str, Path]] = None,
        progress: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a list of image tensors.
        
        Args:
            tensors: List of image tensors [C, H, W] or [1, C, H, W] in range [0, 1]
            tag: Label for this evaluation
            output_dir: Optional directory to save results
            progress: Whether to show progress
            
        Returns:
            EvaluationResult containing predictions and metrics
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for tensor evaluation")
        
        # Convert tensors to numpy arrays
        images = []
        for tensor in tensors:
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            
            # Convert from [C, H, W] to [H, W, C] and to uint8
            img = tensor.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            images.append(img)
        
        return self.evaluate_images(
            images=images,
            tag=tag,
            output_dir=output_dir,
            progress=progress,
        )
    
    def compare(
        self,
        result_a: EvaluationResult,
        result_b: EvaluationResult,
    ) -> pd.DataFrame:
        """
        Compare metrics between two evaluation results.
        
        Args:
            result_a: First evaluation result
            result_b: Second evaluation result
            
        Returns:
            DataFrame comparing all metrics
        """
        if result_a.metrics is None or result_b.metrics is None:
            raise ValueError("Both results must have computed metrics")
        
        return compare_metrics(
            result_a.metrics,
            result_b.metrics,
            name_a=result_a.tag or "A",
            name_b=result_b.tag or "B",
        )
    
    def print_comparison(
        self,
        result_a: EvaluationResult,
        result_b: EvaluationResult,
    ):
        """Print a formatted comparison of two results."""
        comparison = self.compare(result_a, result_b)
        
        print("\n" + "="*80)
        print(f"Comparison: {result_a.tag} vs {result_b.tag}")
        print("="*80)
        
        print(f"\nFaces detected:")
        print(f"  {result_a.tag}: {result_a.num_faces} faces from {result_a.images_with_faces}/{result_a.num_images} images")
        print(f"  {result_b.tag}: {result_b.num_faces} faces from {result_b.images_with_faces}/{result_b.num_images} images")
        
        print(f"\nBias-W (lower = more uniform):")
        for _, row in comparison.iterrows():
            attr = row['attribute']
            a_val = row[f'bias_w_{result_a.tag}']
            b_val = row[f'bias_w_{result_b.tag}']
            diff = row['bias_w_diff']
            direction = "↓" if diff < 0 else "↑" if diff > 0 else "="
            print(f"  {attr:25s}: {a_val:.4f} → {b_val:.4f} ({direction} {abs(diff):.4f})")
        
        print(f"\nENS (higher = more diverse, max = # categories):")
        for _, row in comparison.iterrows():
            attr = row['attribute']
            a_val = row[f'ens_{result_a.tag}']
            b_val = row[f'ens_{result_b.tag}']
            diff = row['ens_diff']
            direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  {attr:25s}: {a_val:.2f} → {b_val:.2f} ({direction} {abs(diff):.2f})")
        
        print(f"\nKL Divergence (lower = closer to uniform):")
        for _, row in comparison.iterrows():
            attr = row['attribute']
            a_val = row[f'kl_divergence_{result_a.tag}']
            b_val = row[f'kl_divergence_{result_b.tag}']
            diff = row['kl_divergence_diff']
            direction = "↓" if diff < 0 else "↑" if diff > 0 else "="
            print(f"  {attr:25s}: {a_val:.4f} → {b_val:.4f} ({direction} {abs(diff):.4f})")
        
        print("="*80)


def quick_evaluate(
    images: List[Union[str, Path, np.ndarray]],
    face_model_path: Optional[str] = None,
    fairface_model_path: Optional[str] = None,
    tag: str = "evaluation",
    output_dir: Optional[Union[str, Path]] = None,
) -> EvaluationResult:
    """
    Quick evaluation function for simple usage.
    
    Args:
        images: List of image paths or arrays
        face_model_path: Path to dlib shape predictor
        fairface_model_path: Path to FairFace model
        tag: Label for evaluation
        output_dir: Optional directory to save results
        
    Returns:
        EvaluationResult with all metrics
        
    Example:
        result = quick_evaluate(
            images=["img1.png", "img2.png"],
            output_dir="results/"
        )
        print(result.metrics.summary())
    """
    evaluator = DemographicEvaluator(
        face_model_path=face_model_path,
        fairface_model_path=fairface_model_path,
    )
    
    return evaluator.evaluate_images(
        images=images,
        tag=tag,
        output_dir=output_dir,
    )
