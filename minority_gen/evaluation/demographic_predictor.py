"""
Demographic prediction using FairFace model.

This module provides functionality to predict race, gender, and age
from face images using the pretrained FairFace ResNet34 model.

Reference: https://github.com/joojs/fairface
Paper: "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age"
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Demographic prediction will not be available.")

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

from .constants import (
    RACE_LABELS_7,
    GENDER_LABELS,
    AGE_LABELS,
    RACE_OUTPUT_SLICE,
    GENDER_OUTPUT_SLICE,
    AGE_OUTPUT_SLICE,
    FAIRFACE_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .face_detection import DetectedFace


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for array x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


@dataclass
class DemographicPrediction:
    """Container for demographic prediction results."""
    
    # Predicted labels
    race: str = ""
    gender: str = ""
    age: str = ""
    
    # Prediction scores (probabilities)
    race_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    gender_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    age_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Source information
    source_path: str = ""
    face_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'race': self.race,
            'gender': self.gender,
            'age': self.age,
            'race_scores': self.race_scores.tolist() if len(self.race_scores) > 0 else [],
            'gender_scores': self.gender_scores.tolist() if len(self.gender_scores) > 0 else [],
            'age_scores': self.age_scores.tolist() if len(self.age_scores) > 0 else [],
            'source_path': self.source_path,
            'face_index': self.face_index,
        }
    
    def get_confidence(self, attribute: str) -> float:
        """Get confidence score for predicted label."""
        if attribute == 'race':
            return float(np.max(self.race_scores)) if len(self.race_scores) > 0 else 0.0
        elif attribute == 'gender':
            return float(np.max(self.gender_scores)) if len(self.gender_scores) > 0 else 0.0
        elif attribute == 'age':
            return float(np.max(self.age_scores)) if len(self.age_scores) > 0 else 0.0
        else:
            raise ValueError(f"Unknown attribute: {attribute}")


class DemographicPredictor:
    """
    Demographic predictor using FairFace model.
    
    Predicts race (7 classes), gender (2 classes), and age (9 classes)
    from aligned face images.
    
    Example:
        predictor = DemographicPredictor(model_path="fair_face_model/res34_fair_align_multi_7_20190809.pt")
        
        # From face detector output
        predictions = predictor.predict_faces(detected_faces)
        
        # From image paths
        predictions = predictor.predict_from_paths(["face1.png", "face2.png"])
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the demographic predictor.
        
        Args:
            model_path: Path to FairFace model weights (.pt file)
                       Expected: res34_fair_align_multi_7_20190809.pt
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install with: pip install torch torchvision")
        
        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Find model path
        if model_path is None:
            model_path = self._find_model()
        
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FairFace model not found at {model_path}. "
                "Download from: https://github.com/joojs/fairface"
            )
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((FAIRFACE_INPUT_SIZE, FAIRFACE_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    def _find_model(self) -> Optional[str]:
        """Search for FairFace model in common locations."""
        common_paths = [
            "res34_fair_align_multi_7_20190809.pt",
            "models/res34_fair_align_multi_7_20190809.pt",
            "fair_face_model/res34_fair_align_multi_7_20190809.pt",
            os.path.expanduser("~/.fairface/res34_fair_align_multi_7_20190809.pt"),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the FairFace ResNet34 model."""
        # Create ResNet34 with modified final layer
        # FairFace outputs: 7 (race) + 2 (gender) + 9 (age) = 18
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 18)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _predict_single(self, image: np.ndarray) -> DemographicPrediction:
        """
        Predict demographics for a single face image.
        
        Args:
            image: Face image as numpy array (RGB, aligned)
            
        Returns:
            DemographicPrediction object
        """
        # Transform and add batch dimension
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor).cpu().numpy().squeeze()
        
        # Parse outputs
        race_logits = outputs[RACE_OUTPUT_SLICE]
        gender_logits = outputs[GENDER_OUTPUT_SLICE]
        age_logits = outputs[AGE_OUTPUT_SLICE]
        
        # Convert to probabilities
        race_scores = softmax(race_logits)
        gender_scores = softmax(gender_logits)
        age_scores = softmax(age_logits)
        
        # Get predicted labels
        race_idx = np.argmax(race_scores)
        gender_idx = np.argmax(gender_scores)
        age_idx = np.argmax(age_scores)
        
        return DemographicPrediction(
            race=RACE_LABELS_7[race_idx],
            gender=GENDER_LABELS[gender_idx],
            age=AGE_LABELS[age_idx],
            race_scores=race_scores,
            gender_scores=gender_scores,
            age_scores=age_scores,
        )
    
    def predict_faces(
        self,
        faces: List[DetectedFace],
        progress: bool = True,
    ) -> List[DemographicPrediction]:
        """
        Predict demographics for detected faces.
        
        Args:
            faces: List of DetectedFace objects from FaceDetector
            progress: Whether to show progress
            
        Returns:
            List of DemographicPrediction objects
        """
        predictions = []
        
        for i, face in enumerate(faces):
            if progress and i % 100 == 0:
                print(f"Predicting: {i}/{len(faces)}")
            
            try:
                pred = self._predict_single(face.image)
                pred.source_path = face.source_path
                pred.face_index = face.face_index
                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Failed to predict for face {i}: {e}")
        
        return predictions
    
    def predict_from_paths(
        self,
        image_paths: List[Union[str, Path]],
        progress: bool = True,
    ) -> List[DemographicPrediction]:
        """
        Predict demographics from image file paths.
        
        Assumes images are already aligned face crops.
        
        Args:
            image_paths: List of paths to face images
            progress: Whether to show progress
            
        Returns:
            List of DemographicPrediction objects
        """
        if not DLIB_AVAILABLE:
            raise RuntimeError("dlib is required to load images. Install with: pip install dlib")
        
        predictions = []
        
        for i, path in enumerate(image_paths):
            if progress and i % 100 == 0:
                print(f"Predicting: {i}/{len(image_paths)}")
            
            try:
                # Load image
                image = dlib.load_rgb_image(str(path))
                
                # Predict
                pred = self._predict_single(image)
                pred.source_path = str(path)
                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Failed to process {path}: {e}")
        
        return predictions
    
    def predictions_to_dataframe(
        self,
        predictions: List[DemographicPrediction],
    ) -> pd.DataFrame:
        """
        Convert predictions to pandas DataFrame.
        
        Args:
            predictions: List of DemographicPrediction objects
            
        Returns:
            DataFrame with columns: source_path, face_index, race, gender, age,
                                   race_scores, gender_scores, age_scores
        """
        records = []
        for pred in predictions:
            records.append({
                'face_name_align': pred.source_path,
                'face_index': pred.face_index,
                'race': pred.race,
                'gender': pred.gender,
                'age': pred.age,
                'race_scores_fair': pred.race_scores,
                'gender_scores_fair': pred.gender_scores,
                'age_scores_fair': pred.age_scores,
            })
        
        return pd.DataFrame(records)
