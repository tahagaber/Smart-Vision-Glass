import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

@dataclass
class Config:
    # Camera settings
    width: int
    height: int
    fps: int
    buffer_size: int
    
    # Face detection
    face_scale_factor: float
    face_min_neighbors: int
    face_resize_dim: Tuple[int, int]
    face_confidence_threshold: int
    
    # Color detection
    color_min_pixels_threshold: int
    color_detection_region: Tuple[float, float]
    colors: Dict[str, Dict[str, List[List[int]]]]
    
    # Object detection
    obj_confidence_threshold: float
    obj_detection_interval: float
    obj_input_size: int
    
    # Model paths
    face_recognizer_path: str
    label_map_path: str
    face_cascade_path: str

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        return cls(
            width=config_data['camera']['width'],
            height=config_data['camera']['height'],
            fps=config_data['camera']['fps'],
            buffer_size=config_data['camera']['buffer_size'],
            
            face_scale_factor=config_data['face_detection']['scale_factor'],
            face_min_neighbors=config_data['face_detection']['min_neighbors'],
            face_resize_dim=tuple(config_data['face_detection']['resize_dimensions']),
            face_confidence_threshold=config_data['face_detection']['confidence_threshold'],
            
            color_min_pixels_threshold=config_data['color_detection']['min_pixels_threshold'],
            color_detection_region=tuple(config_data['color_detection']['detection_region']),
            colors=config_data['color_detection']['colors'],
            
            obj_confidence_threshold=config_data['object_detection']['confidence_threshold'],
            obj_detection_interval=config_data['object_detection']['detection_interval'],
            obj_input_size=config_data['object_detection']['input_size'],
            
            face_recognizer_path=config_data['models']['face_recognizer'],
            label_map_path=config_data['models']['label_map'],
            face_cascade_path=config_data['models']['face_cascade']
        )
