#!/usr/bin/env python3
"""
Synthetic tests to validate attention improvements before real deployment.
Creates controlled scenarios where we know what the attention SHOULD be.
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple
from loguru import logger

from src.attention_maps import ACTPolicyWithAttention


class SyntheticAttentionTester:
    """Creates synthetic scenarios to test attention quality."""
    
    def __init__(self):
        self.test_scenarios = {
            'moving_object': self._create_moving_object_scenario,
            'stationary_distractor': self._create_distractor_scenario,
            'multi_object': self._create_multi_object_scenario,
            'spatial_pattern': self._create_spatial_pattern_scenario,
            'temporal_consistency': self._create_temporal_consistency_scenario
        }
    
    def run_all_tests(self, policy) -> Dict[str, float]:
        """Run all synthetic tests."""
        results = {}
        
        for test_name, test_func in self.test_scenarios.items():
            logger.info(f"Running synthetic test: {test_name}")
            score = self._run_single_test(policy, test_func)
            results[test_name] = score
            logger.info(f"Test {test_name} score: {score:.3f}")
            
        # Overall score
        results['overall_score'] = np.mean(list(results.values()))
        
        return results
    
    def _run_single_test(self, policy, test_func) -> float:
        """Run a single synthetic test."""
        try:
            # Generate test scenario
            scenario = test_func()
            
            # Get attention from policy
            attention_maps = self._get_policy_attention(policy, scenario)
            
            # Evaluate against ground truth
            score = self._evaluate_attention_quality(attention_maps, scenario)
            
            return score
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return 0.0
    
    def _create_moving_object_scenario(self) -> Dict:
        """
        Scénario: Un objet se déplace de gauche à droite.
        L'attention devrait le suivre.
        """
        scenario = {
            'name': 'moving_object',
            'frames': [],
            'ground_truth_attention': [],
            'description': 'Object moves left to right, attention should follow'
        }
        
        # Créer 10 frames avec objet qui bouge
        for frame_idx in range(10):
            # Image de base (640x480)
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Position de l'objet (bouge de gauche à droite)
            x_pos = int(50 + frame_idx * 50)  # 50 à 500
            y_pos = 240  # Centre vertical
            
            # Dessiner objet (carré rouge)
            cv2.rectangle(img, (x_pos-20, y_pos-20), (x_pos+20, y_pos+20), (0, 0, 255), -1)
            
            # Ground truth attention (gaussienne centrée sur l'objet)
            gt_attention = self._create_gaussian_attention((480, 640), (y_pos, x_pos), sigma=50)
            
            scenario['frames'].append(img)
            scenario['ground_truth_attention'].append(gt_attention)
            
        return scenario
    
    def _create_distractor_scenario(self) -> Dict:
        """
        Scénario: Objet cible + distracteurs.
        L'attention devrait ignorer les distracteurs.
        """
        scenario = {
            'name': 'stationary_distractor',
            'frames': [],
            'ground_truth_attention': [],
            'description': 'Target object with distractors, should focus on target'
        }
        
        for frame_idx in range(5):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Objet cible (carré rouge au centre)
            target_pos = (320, 240)
            cv2.rectangle(img, (300, 220), (340, 260), (0, 0, 255), -1)
            
            # Distracteurs (carrés bleus)
            distractors = [(100, 100), (500, 100), (100, 350), (500, 350)]
            for dx, dy in distractors:
                cv2.rectangle(img, (dx-15, dy-15), (dx+15, dy+15), (255, 0, 0), -1)
            
            # Ground truth: attention sur target seulement
            gt_attention = self._create_gaussian_attention((480, 640), target_pos[::-1], sigma=40)
            
            scenario['frames'].append(img)
            scenario['ground_truth_attention'].append(gt_attention)
            
        return scenario
    
    def _create_multi_object_scenario(self) -> Dict:
        """
        Scénario: Plusieurs objets importants.
        L'attention devrait être partagée.
        """
        scenario = {
            'name': 'multi_object',
            'frames': [],
            'ground_truth_attention': [],
            'description': 'Multiple important objects, attention should be distributed'
        }
        
        for frame_idx in range(5):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Deux objets importants
            obj1_pos = (200, 150)
            obj2_pos = (440, 330)
            
            cv2.rectangle(img, (180, 130), (220, 170), (0, 255, 0), -1)  # Vert
            cv2.rectangle(img, (420, 310), (460, 350), (0, 255, 255), -1)  # Cyan
            
            # Ground truth: attention partagée
            gt_attention1 = self._create_gaussian_attention((480, 640), obj1_pos[::-1], sigma=35)
            gt_attention2 = self._create_gaussian_attention((480, 640), obj2_pos[::-1], sigma=35)
            gt_attention = gt_attention1 + gt_attention2
            gt_attention = gt_attention / gt_attention.max()  # Normaliser
            
            scenario['frames'].append(img)
            scenario['ground_truth_attention'].append(gt_attention)
            
        return scenario
    
    def _create_spatial_pattern_scenario(self) -> Dict:
        """
        Scénario: Pattern spatial spécifique.
        Test la capacité à détecter des patterns géométriques.
        """
        scenario = {
            'name': 'spatial_pattern',
            'frames': [],
            'ground_truth_attention': [],
            'description': 'Geometric patterns, should detect structured arrangements'
        }
        
        for frame_idx in range(3):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Pattern en cercle
            center = (320, 240)
            radius = 100
            num_points = 8
            
            points = []
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))
                cv2.circle(img, (x, y), 10, (255, 255, 0), -1)
                points.append((y, x))  # Note: (y, x) pour ground truth
            
            # Ground truth: attention sur tous les points du pattern
            gt_attention = np.zeros((480, 640))
            for py, px in points:
                gt_attention += self._create_gaussian_attention((480, 640), (py, px), sigma=25)
            gt_attention = gt_attention / gt_attention.max()
            
            scenario['frames'].append(img)
            scenario['ground_truth_attention'].append(gt_attention)
            
        return scenario
    
    def _create_temporal_consistency_scenario(self) -> Dict:
        """
        Scénario: Test cohérence temporelle.
        L'objet bouge lentement, l'attention devrait évoluer graduellement.
        """
        scenario = {
            'name': 'temporal_consistency',
            'frames': [],
            'ground_truth_attention': [],
            'description': 'Slowly moving object, attention should change gradually'
        }
        
        # Mouvement très lent (interpolation fine)
        start_pos = (100, 240)
        end_pos = (540, 240)
        num_frames = 20
        
        for frame_idx in range(num_frames):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Position interpolée
            t = frame_idx / (num_frames - 1)
            x_pos = int(start_pos[0] * (1-t) + end_pos[0] * t)
            y_pos = int(start_pos[1] * (1-t) + end_pos[1] * t)
            
            # Dessiner objet
            cv2.circle(img, (x_pos, y_pos), 15, (255, 0, 255), -1)
            
            # Ground truth attention
            gt_attention = self._create_gaussian_attention((480, 640), (y_pos, x_pos), sigma=40)
            
            scenario['frames'].append(img)
            scenario['ground_truth_attention'].append(gt_attention)
            
        return scenario
    
    def _create_gaussian_attention(self, shape: Tuple[int, int], center: Tuple[int, int], sigma: float) -> np.ndarray:
        """Crée une carte d'attention gaussienne."""
        h, w = shape
        cy, cx = center
        
        # Grille de coordonnées
        y, x = np.ogrid[:h, :w]
        
        # Distance au centre
        dist_sq = (x - cx)**2 + (y - cy)**2
        
        # Gaussienne
        attention = np.exp(-dist_sq / (2 * sigma**2))
        
        # Normaliser
        attention = attention / attention.max()
        
        return attention.astype(np.float32)
    
    def _get_policy_attention(self, policy, scenario: Dict) -> List[np.ndarray]:
        """Obtient l'attention du policy pour un scénario."""
        attention_maps = []
        
        for frame in scenario['frames']:
            # Convertir frame en observation
            observation = self._frame_to_observation(frame)
            
            try:
                # Obtenir attention
                with torch.no_grad():
                    action, attention = policy.select_action(observation)
                
                # Prendre la première carte d'attention (première caméra)
                if attention and len(attention) > 0 and attention[0] is not None:
                    attention_maps.append(attention[0])
                else:
                    # Attention vide si échec
                    attention_maps.append(np.zeros((480, 640), dtype=np.float32))
                    
            except Exception as e:
                logger.warning(f"Failed to get attention: {e}")
                attention_maps.append(np.zeros((480, 640), dtype=np.float32))
                
        return attention_maps
    
    def _frame_to_observation(self, frame: np.ndarray) -> Dict[str, torch.Tensor]:
        """Convertit une frame en observation pour le policy."""
        # Normaliser image
        frame_norm = frame.astype(np.float32) / 255.0
        
        # Convertir en tensor (C, H, W)
        frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1)
        
        # Ajouter dimension batch
        frame_tensor = frame_tensor.unsqueeze(0)
        
        # Observation basique (adapter selon votre config)
        observation = {
            'observation.images.cam_high': frame_tensor,
            # Ajouter autres clés selon votre configuration
        }
        
        return observation
    
    def _evaluate_attention_quality(self, predicted_attention: List[np.ndarray], scenario: Dict) -> float:
        """Évalue la qualité de l'attention prédite vs ground truth."""
        if not predicted_attention or not scenario['ground_truth_attention']:
            return 0.0
        
        scores = []
        
        for pred_attn, gt_attn in zip(predicted_attention, scenario['ground_truth_attention']):
            if pred_attn is None:
                scores.append(0.0)
                continue
                
            # Redimensionner si nécessaire
            if pred_attn.shape != gt_attn.shape:
                pred_attn = cv2.resize(pred_attn, gt_attn.shape[::-1])
            
            # Normaliser les deux
            pred_norm = pred_attn / (pred_attn.max() + 1e-8)
            gt_norm = gt_attn / (gt_attn.max() + 1e-8)
            
            # Plusieurs métriques
            # 1. Corrélation
            corr = np.corrcoef(pred_norm.flatten(), gt_norm.flatten())[0, 1]
            if np.isnan(corr):
                corr = 0.0
            
            # 2. Intersection over Union (attention seuillée)
            pred_binary = pred_norm > 0.5
            gt_binary = gt_norm > 0.5
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            iou = intersection / (union + 1e-8)
            
            # 3. Divergence KL (attention comme distributions)
            pred_dist = pred_norm + 1e-8
            gt_dist = gt_norm + 1e-8
            pred_dist = pred_dist / pred_dist.sum()
            gt_dist = gt_dist / gt_dist.sum()
            kl_div = np.sum(gt_dist * np.log(gt_dist / pred_dist))
            kl_score = np.exp(-kl_div)  # Convertir en score [0,1]
            
            # Score combiné
            combined_score = 0.4 * max(0, corr) + 0.3 * iou + 0.3 * kl_score
            scores.append(combined_score)
        
        return np.mean(scores) if scores else 0.0


def run_quick_validation(policy) -> Dict[str, float]:
    """Fonction helper pour validation rapide."""
    tester = SyntheticAttentionTester()
    return tester.run_all_tests(policy)


if __name__ == "__main__":
    logger.info("Synthetic attention testing framework ready")
    
    # Example usage:
    # policy = load_your_policy()
    # results = run_quick_validation(policy)
    # print(f"Validation scores: {results}")