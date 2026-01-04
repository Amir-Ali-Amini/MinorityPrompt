"""
Demographic Bias Evaluation Module for MinorityPrompt.

This module provides tools for evaluating demographic bias and diversity
in generated images, specifically for text-to-image models.

Pipeline:
    1. Face Detection: Uses dlib to detect and align faces
    2. Demographic Prediction: Uses FairFace model to predict race, gender, age
    3. Metrics Computation: Calculates Bias-W, Bias-P, ENS, KL Divergence

Quick Start:
    from minority_gen.evaluation import DemographicEvaluator
    
    # Initialize evaluator
    evaluator = DemographicEvaluator(
        face_model_path="path/to/shape_predictor_5_face_landmarks.dat",
        fairface_model_path="path/to/res34_fair_align_multi_7_20190809.pt",
    )
    
    # Evaluate image directories
    result_baseline = evaluator.evaluate_directory("baseline_images/", tag="baseline")
    result_minority = evaluator.evaluate_directory("minority_images/", tag="minority")
    
    # Compare results
    evaluator.print_comparison(result_baseline, result_minority)
    
    # Save results
    result_baseline.save("results/baseline/")
    result_minority.save("results/minority/")

Available Metrics:
    - Bias-W: Population-level bias (deviation from uniform distribution)
    - Bias-P: Per-image bias for multi-face images
    - ENS: Effective Number of Species (diversity metric based on Shannon entropy)
    - KL Divergence: Kullback-Leibler divergence from reference distribution

Model Requirements:
    - dlib shape predictor: shape_predictor_5_face_landmarks.dat
      Download: http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
      
    - FairFace model: res34_fair_align_multi_7_20190809.pt
      Download: https://github.com/joojs/fairface

Dependencies:
    - dlib (pip install dlib)
    - torch, torchvision
    - numpy, pandas
"""

from .constants import (
    RACE_LABELS_7,
    GENDER_LABELS,
    AGE_LABELS,
    ATTRIBUTES_DICT,
    ATTRIBUTE_COMBINATIONS,
)

from .face_detection import (
    FaceDetector,
    DetectedFace,
    rect_to_bbox,
)

from .demographic_predictor import (
    DemographicPredictor,
    DemographicPrediction,
)

from .metrics import (
    BiasMetrics,
    calculate_bias_w,
    calculate_bias_p,
    calculate_ens,
    calculate_kl_divergence,
    calculate_all_metrics,
    compare_metrics,
)

from .evaluator import (
    DemographicEvaluator,
    EvaluationResult,
    quick_evaluate,
)


__all__ = [
    # Constants
    'RACE_LABELS_7',
    'GENDER_LABELS',
    'AGE_LABELS',
    'ATTRIBUTES_DICT',
    'ATTRIBUTE_COMBINATIONS',
    
    # Face Detection
    'FaceDetector',
    'DetectedFace',
    'rect_to_bbox',
    
    # Demographic Prediction
    'DemographicPredictor',
    'DemographicPrediction',
    
    # Metrics
    'BiasMetrics',
    'calculate_bias_w',
    'calculate_bias_p',
    'calculate_ens',
    'calculate_kl_divergence',
    'calculate_all_metrics',
    'compare_metrics',
    
    # High-level API
    'DemographicEvaluator',
    'EvaluationResult',
    'quick_evaluate',
]
