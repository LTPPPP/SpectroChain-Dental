import numpy as np
import hashlib
import json
import time
from typing import Dict, List, Tuple
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class SpectralData:
    """Đại diện cho dữ liệu quang phổ"""
    def __init__(self, wavelengths: np.ndarray, intensities: np.ndarray, 
                 material_type: str, purity: float = 1.0):
        self.wavelengths = wavelengths
        self.intensities = intensities
        self.material_type = material_type
        self.purity = purity
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Tính SHA-256 hash của dữ liệu quang phổ"""
        # Chuẩn hóa dữ liệu để đảm bảo tính nhất quán
        normalized_data = {
            "wavelengths": self.wavelengths.round(2).tolist(),
            "intensities": self.intensities.round(4).tolist(),
            "material_type": self.material_type
        }
        data_string = json.dumps(normalized_data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def add_noise(self, noise_level: float = 0.1) -> 'SpectralData':
        """Thêm nhiễu vào quang phổ"""
        noise = np.random.normal(0, noise_level, len(self.intensities))
        noisy_intensities = self.intensities + noise
        return SpectralData(self.wavelengths, noisy_intensities, 
                          self.material_type, self.purity)
    
    def dilute(self, dilution_factor: float = 0.8) -> 'SpectralData':
        """Pha loãng quang phổ (giảm độ tinh khiết)"""
        diluted_intensities = self.intensities * dilution_factor
        return SpectralData(self.wavelengths, diluted_intensities, 
                          self.material_type, self.purity * dilution_factor)

class MaterialDatabase:
    """Cơ sở dữ liệu vật liệu nha khoa"""
    def __init__(self):
        self.materials = self.generate_sample_materials()
    
    def generate_sample_materials(self) -> Dict[str, SpectralData]:
        """Tạo dữ liệu mẫu cho các vật liệu nha khoa"""
        materials = {}
        
        # Định nghĩa các loại vật liệu và đặc trưng quang phổ
        material_specs = {
            "titanium_implant": {
                "peaks": [144, 197, 282, 395, 515, 612],
                "base_intensity": 1000,
                "description": "Titanium Grade 4 Dental Implant"
            },
            "zirconia_crown": {
                "peaks": [178, 262, 334, 476, 538, 615],
                "base_intensity": 800,
                "description": "Yttria-Stabilized Zirconia Crown"
            },
            "composite_filling": {
                "peaks": [156, 289, 398, 512, 623, 745],
                "base_intensity": 600,
                "description": "Bis-GMA Composite Resin"
            },
            "ceramic_veneer": {
                "peaks": [168, 245, 367, 445, 578, 689],
                "base_intensity": 750,
                "description": "Feldspathic Porcelain Veneer"
            },
            "gold_crown": {
                "peaks": [125, 234, 356, 567, 678, 789],
                "base_intensity": 1200,
                "description": "Gold Alloy Crown"
            }
        }
        
        # Tạo quang phổ cho mỗi loại vật liệu
        wavelengths = np.linspace(100, 800, 350)  # Raman shift range
        
        for material_id, spec in material_specs.items():
            intensities = np.zeros_like(wavelengths)
            
            # Thêm các peak đặc trưng
            for peak in spec["peaks"]:
                peak_idx = np.argmin(np.abs(wavelengths - peak))
                # Tạo peak Gaussian
                sigma = 10  # Độ rộng peak
                peak_height = spec["base_intensity"] * (0.8 + 0.4 * random.random())
                gaussian = peak_height * np.exp(-0.5 * ((wavelengths - peak) / sigma) ** 2)
                intensities += gaussian
            
            # Thêm background noise
            background = 50 + 30 * np.random.random(len(wavelengths))
            intensities += background
            
            # Tạo SpectralData object
            materials[material_id] = SpectralData(wavelengths, intensities, material_id)
        
        return materials
    
    def get_material(self, material_id: str) -> SpectralData:
        """Lấy dữ liệu quang phổ của vật liệu"""
        return self.materials.get(material_id)
    
    def list_materials(self) -> List[str]:
        """Liệt kê tất cả vật liệu có sẵn"""
        return list(self.materials.keys())

class CounterfeitGenerator:
    """Tạo dữ liệu quang phổ giả mạo để test"""
    
    @staticmethod
    def substitute_material(original: SpectralData, substitute_material: SpectralData) -> SpectralData:
        """Thay thế bằng vật liệu rẻ tiền"""
        return SpectralData(
            substitute_material.wavelengths,
            substitute_material.intensities,
            f"counterfeit_{original.material_type}",
            0.0  # Purity = 0 cho hàng giả
        )
    
    @staticmethod
    def dilute_purity(original: SpectralData, dilution_factor: float = 0.6) -> SpectralData:
        """Pha loãng độ tinh khiết"""
        return original.dilute(dilution_factor)
    
    @staticmethod
    def degrade_storage(original: SpectralData, degradation_level: float = 0.3) -> SpectralData:
        """Mô phỏng suy thoái do bảo quản sai cách"""
        degraded = original.add_noise(degradation_level)
        # Giảm cường độ peak
        degraded.intensities *= (1 - degradation_level * 0.5)
        degraded.purity = original.purity * (1 - degradation_level)
        return degraded

class VerificationEngine:
    """Engine xác minh quang phổ"""
    
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.verification_history = []
    
    def verify_spectral_match(self, reference_hash: str, sample_hash: str) -> bool:
        """Xác minh bằng so sánh hash"""
        match = reference_hash == sample_hash
        self.verification_history.append({
            "reference_hash": reference_hash[:16] + "...",
            "sample_hash": sample_hash[:16] + "...",
            "match": match,
            "timestamp": time.time()
        })
        return match
    
    def verify_spectral_similarity(self, reference: SpectralData, sample: SpectralData) -> Tuple[bool, float]:
        """Xác minh bằng độ tương đồng quang phổ"""
        # Tính correlation coefficient
        correlation = np.corrcoef(reference.intensities, sample.intensities)[0, 1]
        
        # Tính Euclidean distance (normalized)
        norm_ref = reference.intensities / np.linalg.norm(reference.intensities)
        norm_sample = sample.intensities / np.linalg.norm(sample.intensities)
        distance = np.linalg.norm(norm_ref - norm_sample)
        
        # Combine metrics
        similarity = correlation * (1 - distance / 2)
        is_authentic = similarity >= self.threshold
        
        self.verification_history.append({
            "correlation": correlation,
            "distance": distance,
            "similarity": similarity,
            "is_authentic": is_authentic,
            "timestamp": time.time()
        })
        
        return is_authentic, similarity
    
    def batch_verify(self, reference_spectra: List[SpectralData], 
                    test_spectra: List[SpectralData], 
                    labels: List[bool]) -> Dict:
        """Xác minh hàng loạt và tính toán metrics"""
        predictions = []
        similarities = []
        
        for ref, test in zip(reference_spectra, test_spectra):
            is_authentic, similarity = self.verify_spectral_similarity(ref, test)
            predictions.append(is_authentic)
            similarities.append(similarity)
        
        # Tính toán metrics
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        # ROC AUC (sử dụng similarity scores)
        auc = roc_auc_score(labels, similarities)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "predictions": predictions,
            "similarities": similarities,
            "accuracy": sum(p == l for p, l in zip(predictions, labels)) / len(labels)
        } 