"""
Face detection and alignment using dlib.

This module provides functionality to detect faces in images and
extract aligned face crops for demographic classification.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("Warning: dlib not installed. Face detection will not be available.")
    print("Install with: pip install dlib")

from .constants import DEFAULT_MAX_SIZE, FACE_SIZE, FACE_PADDING


@dataclass
class DetectedFace:
    """Container for a detected face."""
    image: np.ndarray  # Aligned face image (RGB)
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    source_path: str  # Original image path
    face_index: int  # Index if multiple faces in image


def rect_to_bbox(rect) -> Tuple[int, int, int, int]:
    """
    Convert dlib rectangle to bounding box tuple.
    
    Args:
        rect: dlib rectangle object
        
    Returns:
        Tuple of (x, y, width, height)
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


class FaceDetector:
    """
    Face detector using dlib's frontal face detector and 5-point landmark predictor.
    
    This class detects faces in images and returns aligned face crops suitable
    for demographic classification with FairFace.
    
    Example:
        detector = FaceDetector(landmark_model_path="path/to/shape_predictor_5_face_landmarks.dat")
        faces = detector.detect("image.jpg")
        for face in faces:
            print(f"Face {face.face_index}: bbox={face.bbox}")
    """
    
    def __init__(
        self,
        landmark_model_path: Optional[str] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        face_size: int = FACE_SIZE,
        padding: float = FACE_PADDING,
    ):
        """
        Initialize the face detector.
        
        Args:
            landmark_model_path: Path to dlib's shape_predictor_5_face_landmarks.dat
                               If None, will look in common locations or download
            max_size: Maximum dimension for resizing input images
            face_size: Output size for aligned face crops
            padding: Padding around face for alignment
        """
        if not DLIB_AVAILABLE:
            raise RuntimeError("dlib is required for face detection. Install with: pip install dlib")
        
        self.max_size = max_size
        self.face_size = face_size
        self.padding = padding
        
        # Initialize face detector
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Initialize landmark predictor
        if landmark_model_path is None:
            landmark_model_path = self._find_landmark_model()
        
        if landmark_model_path and os.path.exists(landmark_model_path):
            self.shape_predictor = dlib.shape_predictor(landmark_model_path)
        else:
            raise FileNotFoundError(
                f"Landmark model not found at {landmark_model_path}. "
                "Download from: http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
            )
    
    def _find_landmark_model(self) -> Optional[str]:
        """Search for landmark model in common locations."""
        common_paths = [
            "shape_predictor_5_face_landmarks.dat",
            "models/shape_predictor_5_face_landmarks.dat",
            "dlib_models/shape_predictor_5_face_landmarks.dat",
            os.path.expanduser("~/.dlib/shape_predictor_5_face_landmarks.dat"),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        old_height, old_width = img.shape[:2]
        
        if max(old_height, old_width) <= self.max_size:
            return img
        
        if old_width > old_height:
            new_width = self.max_size
            new_height = int(self.max_size * old_height / old_width)
        else:
            new_height = self.max_size
            new_width = int(self.max_size * old_width / old_height)
        
        return dlib.resize_image(img, rows=new_height, cols=new_width)
    
    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        upsample_num_times: int = 1,
    ) -> List[DetectedFace]:
        """
        Detect and align faces in an image.
        
        Args:
            image: Path to image file or numpy array (RGB)
            upsample_num_times: Number of times to upsample image for detection
                               Higher values find smaller faces but are slower
                               
        Returns:
            List of DetectedFace objects, one per detected face
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            source_path = str(image)
            img = dlib.load_rgb_image(source_path)
        else:
            source_path = "<array>"
            img = image
        
        # Resize for detection
        img_resized = self._resize_image(img)
        
        # Detect faces
        detections = self.face_detector(img_resized, upsample_num_times)
        
        if len(detections) == 0:
            return []
        
        # Get landmarks for alignment
        faces_full = dlib.full_object_detections()
        for rect in detections:
            faces_full.append(self.shape_predictor(img_resized, rect))
        
        # Get aligned face chips
        face_chips = dlib.get_face_chips(
            img_resized, 
            faces_full, 
            size=self.face_size, 
            padding=self.padding
        )
        
        # Create DetectedFace objects
        results = []
        for idx, (chip, rect) in enumerate(zip(face_chips, detections)):
            results.append(DetectedFace(
                image=np.array(chip),
                bbox=rect_to_bbox(rect),
                source_path=source_path,
                face_index=idx,
            ))
        
        return results
    
    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        progress: bool = True,
    ) -> List[List[DetectedFace]]:
        """
        Detect faces in multiple images.
        
        Args:
            images: List of image paths or arrays
            progress: Whether to show progress
            
        Returns:
            List of lists, where each inner list contains faces for one image
        """
        results = []
        
        for i, image in enumerate(images):
            if progress and i % 100 == 0:
                print(f"Processing image: {i}/{len(images)}")
            
            try:
                faces = self.detect(image)
                results.append(faces)
            except Exception as e:
                print(f"Warning: Failed to process {image}: {e}")
                results.append([])
        
        return results
    
    def save_faces(
        self,
        faces: List[DetectedFace],
        output_dir: Union[str, Path],
        prefix: str = "",
    ) -> List[str]:
        """
        Save detected faces to disk.
        
        Args:
            faces: List of DetectedFace objects
            output_dir: Directory to save face images
            prefix: Optional prefix for filenames
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for face in faces:
            # Generate filename
            source_name = Path(face.source_path).stem
            filename = f"{prefix}{source_name}_face{face.face_index}.png"
            output_path = output_dir / filename
            
            # Save using dlib
            dlib.save_image(face.image, str(output_path))
            saved_paths.append(str(output_path))
        
        return saved_paths
