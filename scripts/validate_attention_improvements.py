#!/usr/bin/env python3
"""
Validation framework for attention improvements.
Tests whether proposed changes actually improve attention quality.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy, pearsonr

from src.attention_maps import ACTPolicyWithAttention


@dataclass
class AttentionMetrics:
    """Metrics for evaluating attention quality."""
    spatial_consistency: float  # Cohérence spatiale
    temporal_consistency: float  # Cohérence temporelle  
    semantic_alignment: float  # Alignement avec objets importants
    information_content: float  # Contenu informationnel
    computational_cost: float  # Coût computationnel
    human_interpretability: float  # Interprétabilité humaine (manuel)


class AttentionValidator:
    """Validates attention improvements against baseline."""
    
    def __init__(self, baseline_policy, improved_policy, validation_episodes: List[int]):
        self.baseline_policy = baseline_policy
        self.improved_policy = improved_policy  
        self.validation_episodes = validation_episodes
        self.results = {}
        
    def run_validation_suite(self) -> Dict:
        """Run complete validation suite."""
        logger.info("Starting attention validation suite...")
        
        results = {
            'baseline_metrics': {},
            'improved_metrics': {},
            'comparisons': {},
            'statistical_tests': {}
        }
        
        # Test each episode
        for episode_id in self.validation_episodes:
            logger.info(f"Validating episode {episode_id}")
            
            # Get attention from both policies
            baseline_attention = self._extract_episode_attention(
                self.baseline_policy, episode_id
            )
            improved_attention = self._extract_episode_attention(
                self.improved_policy, episode_id  
            )
            
            # Compute metrics for both
            baseline_metrics = self._compute_attention_metrics(baseline_attention)
            improved_metrics = self._compute_attention_metrics(improved_attention)
            
            # Store results
            results['baseline_metrics'][episode_id] = baseline_metrics
            results['improved_metrics'][episode_id] = improved_metrics
            
        # Aggregate and compare
        results['comparisons'] = self._compare_metrics(
            results['baseline_metrics'], 
            results['improved_metrics']
        )
        
        # Statistical significance tests
        results['statistical_tests'] = self._statistical_validation(
            results['baseline_metrics'],
            results['improved_metrics']
        )
        
        return results
    
    def _extract_episode_attention(self, policy, episode_id: int) -> List[np.ndarray]:
        """Extract attention maps for entire episode."""
        # Implementation depends on your dataset structure
        attention_sequence = []
        
        # Load episode data
        episode_data = self._load_episode(episode_id)
        
        for frame_data in episode_data:
            observation = self._prepare_observation(frame_data)
            
            # Get attention
            with torch.no_grad():
                action, attention_maps = policy.select_action(observation)
                
            attention_sequence.append(attention_maps)
            
        return attention_sequence
    
    def _compute_attention_metrics(self, attention_sequence: List[np.ndarray]) -> AttentionMetrics:
        """Compute comprehensive attention quality metrics."""
        
        # 1. Spatial Consistency - Les régions importantes restent cohérentes
        spatial_consistency = self._compute_spatial_consistency(attention_sequence)
        
        # 2. Temporal Consistency - Évolution fluide dans le temps
        temporal_consistency = self._compute_temporal_consistency(attention_sequence)
        
        # 3. Information Content - Entropie et diversité
        information_content = self._compute_information_content(attention_sequence)
        
        # 4. Computational Cost - Temps de calcul
        computational_cost = self._measure_computational_cost(attention_sequence)
        
        return AttentionMetrics(
            spatial_consistency=spatial_consistency,
            temporal_consistency=temporal_consistency,
            semantic_alignment=0.0,  # Nécessite ground truth sémantique
            information_content=information_content,
            computational_cost=computational_cost,
            human_interpretability=0.0  # Évaluation manuelle
        )
    
    def _compute_spatial_consistency(self, attention_sequence: List[np.ndarray]) -> float:
        """Mesure la cohérence spatiale de l'attention."""
        if not attention_sequence or len(attention_sequence) < 2:
            return 0.0
            
        consistencies = []
        
        for i in range(len(attention_sequence) - 1):
            current_attention = attention_sequence[i]
            next_attention = attention_sequence[i + 1]
            
            if current_attention is None or next_attention is None:
                continue
                
            # Corrélation spatiale entre frames consécutives
            correlation = self._spatial_correlation(current_attention, next_attention)
            consistencies.append(correlation)
            
        return np.mean(consistencies) if consistencies else 0.0
    
    def _compute_temporal_consistency(self, attention_sequence: List[np.ndarray]) -> float:
        """Mesure la cohérence temporelle (pas de changements brusques)."""
        if len(attention_sequence) < 3:
            return 0.0
            
        temporal_scores = []
        
        for i in range(1, len(attention_sequence) - 1):
            prev_attn = attention_sequence[i-1] 
            curr_attn = attention_sequence[i]
            next_attn = attention_sequence[i+1]
            
            if any(x is None for x in [prev_attn, curr_attn, next_attn]):
                continue
                
            # Mesure de la "fluidité" temporelle
            prev_corr = self._spatial_correlation(prev_attn, curr_attn)
            next_corr = self._spatial_correlation(curr_attn, next_attn)
            
            # Score de cohérence temporelle
            temporal_score = min(prev_corr, next_corr)
            temporal_scores.append(temporal_score)
            
        return np.mean(temporal_scores) if temporal_scores else 0.0
    
    def _compute_information_content(self, attention_sequence: List[np.ndarray]) -> float:
        """Mesure le contenu informationnel (entropie, diversité)."""
        if not attention_sequence:
            return 0.0
            
        entropies = []
        
        for attention_map in attention_sequence:
            if attention_map is None:
                continue
                
            # Normaliser pour probabilités
            attention_flat = attention_map.flatten()
            attention_prob = attention_flat / (attention_flat.sum() + 1e-8)
            
            # Calculer entropie
            attn_entropy = entropy(attention_prob + 1e-8)
            entropies.append(attn_entropy)
            
        return np.mean(entropies) if entropies else 0.0
    
    def _spatial_correlation(self, attn1: np.ndarray, attn2: np.ndarray) -> float:
        """Calcule corrélation spatiale entre deux cartes d'attention."""
        if attn1 is None or attn2 is None:
            return 0.0
            
        # Redimensionner si nécessaire
        if attn1.shape != attn2.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(attn1.shape, attn2.shape))
            attn1 = cv2.resize(attn1, min_shape[::-1])
            attn2 = cv2.resize(attn2, min_shape[::-1])
            
        # Corrélation de Pearson
        corr, _ = pearsonr(attn1.flatten(), attn2.flatten())
        return corr if not np.isnan(corr) else 0.0
    
    def _measure_computational_cost(self, attention_sequence: List[np.ndarray]) -> float:
        """Mesure le coût computationnel."""
        # Approximation basée sur la taille des cartes d'attention
        total_ops = 0
        
        for attention_map in attention_sequence:
            if attention_map is not None:
                total_ops += attention_map.size
                
        return total_ops / len(attention_sequence) if attention_sequence else 0.0
    
    def _compare_metrics(self, baseline_metrics: Dict, improved_metrics: Dict) -> Dict:
        """Compare les métriques entre baseline et amélioration."""
        comparisons = {}
        
        # Agrégation des métriques par épisode
        baseline_agg = self._aggregate_metrics(baseline_metrics)
        improved_agg = self._aggregate_metrics(improved_metrics)
        
        # Calcul des améliorations relatives
        for metric_name in baseline_agg.keys():
            baseline_val = baseline_agg[metric_name]
            improved_val = improved_agg[metric_name]
            
            if baseline_val > 0:
                improvement_pct = ((improved_val - baseline_val) / baseline_val) * 100
            else:
                improvement_pct = 0.0
                
            comparisons[metric_name] = {
                'baseline': baseline_val,
                'improved': improved_val, 
                'improvement_pct': improvement_pct,
                'better': improved_val > baseline_val
            }
            
        return comparisons
    
    def _aggregate_metrics(self, episode_metrics: Dict) -> Dict:
        """Agrège les métriques à travers les épisodes."""
        if not episode_metrics:
            return {}
            
        # Initialiser accumulateurs
        metric_sums = {}
        metric_counts = {}
        
        # Accumuler
        for episode_id, metrics in episode_metrics.items():
            for attr_name in ['spatial_consistency', 'temporal_consistency', 
                            'information_content', 'computational_cost']:
                value = getattr(metrics, attr_name)
                
                if attr_name not in metric_sums:
                    metric_sums[attr_name] = 0.0
                    metric_counts[attr_name] = 0
                    
                metric_sums[attr_name] += value
                metric_counts[attr_name] += 1
                
        # Moyenner
        aggregated = {}
        for metric_name in metric_sums.keys():
            if metric_counts[metric_name] > 0:
                aggregated[metric_name] = metric_sums[metric_name] / metric_counts[metric_name]
            else:
                aggregated[metric_name] = 0.0
                
        return aggregated
    
    def _statistical_validation(self, baseline_metrics: Dict, improved_metrics: Dict) -> Dict:
        """Tests statistiques pour valider la significativité."""
        from scipy.stats import ttest_rel, wilcoxon
        
        # Extraire les valeurs pour chaque métrique
        baseline_values = {}
        improved_values = {}
        
        for episode_id in baseline_metrics.keys():
            baseline_ep = baseline_metrics[episode_id]
            improved_ep = improved_metrics[episode_id]
            
            for metric_name in ['spatial_consistency', 'temporal_consistency', 
                              'information_content']:
                if metric_name not in baseline_values:
                    baseline_values[metric_name] = []
                    improved_values[metric_name] = []
                    
                baseline_values[metric_name].append(getattr(baseline_ep, metric_name))
                improved_values[metric_name].append(getattr(improved_ep, metric_name))
        
        # Tests statistiques
        statistical_results = {}
        
        for metric_name in baseline_values.keys():
            baseline_vals = np.array(baseline_values[metric_name])
            improved_vals = np.array(improved_values[metric_name])
            
            # T-test apparié
            try:
                t_stat, t_pval = ttest_rel(improved_vals, baseline_vals)
                
                statistical_results[metric_name] = {
                    't_statistic': float(t_stat),
                    't_pvalue': float(t_pval), 
                    'significant': t_pval < 0.05,
                    'effect_size': float(np.mean(improved_vals - baseline_vals) / np.std(improved_vals - baseline_vals))
                }
            except Exception as e:
                logger.warning(f"Statistical test failed for {metric_name}: {e}")
                statistical_results[metric_name] = {'error': str(e)}
                
        return statistical_results
    
    def _load_episode(self, episode_id: int):
        """Load episode data - à adapter selon votre format."""
        # Placeholder - remplacer par votre chargement de données
        return []
    
    def _prepare_observation(self, frame_data):
        """Prepare observation - à adapter selon votre format."""
        # Placeholder - remplacer par votre préparation d'observation
        return {}
    
    def save_results(self, results: Dict, output_path: str):
        """Sauvegarde les résultats de validation."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir en format JSON-sérialisable
        json_results = self._serialize_results(results)
        
        with output_file.open('w') as f:
            json.dump(json_results, f, indent=2)
            
        logger.info(f"Validation results saved to {output_path}")
    
    def _serialize_results(self, results: Dict) -> Dict:
        """Convertit les résultats en format JSON-sérialisable."""
        serialized = {}
        
        for key, value in results.items():
            if key in ['baseline_metrics', 'improved_metrics']:
                serialized[key] = {}
                for episode_id, metrics in value.items():
                    serialized[key][episode_id] = {
                        'spatial_consistency': float(metrics.spatial_consistency),
                        'temporal_consistency': float(metrics.temporal_consistency),
                        'information_content': float(metrics.information_content),
                        'computational_cost': float(metrics.computational_cost)
                    }
            else:
                serialized[key] = value
                
        return serialized


if __name__ == "__main__":
    # Example usage
    logger.info("Attention validation framework ready")
    
    # TODO: Adapter avec vos policies et épisodes de validation
    # baseline_policy = load_baseline_policy()
    # improved_policy = load_improved_policy() 
    # validation_episodes = [1, 5, 10, 15, 20]  # Episodes de test
    
    # validator = AttentionValidator(baseline_policy, improved_policy, validation_episodes)
    # results = validator.run_validation_suite()
    # validator.save_results(results, "./validation_results.json")