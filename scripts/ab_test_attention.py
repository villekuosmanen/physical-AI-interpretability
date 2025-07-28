#!/usr/bin/env python3
"""
A/B testing framework for attention improvements.
Compare baseline vs improved attention on real episodes.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from scipy import stats

from scripts.validate_attention_improvements import AttentionValidator
from scripts.synthetic_attention_tests import run_quick_validation


class AttentionABTester:
    """A/B test framework for attention improvements."""
    
    def __init__(self, output_dir: str = "./ab_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def run_ab_test(self, 
                   baseline_policy,
                   improved_policy,
                   test_episodes: List[int],
                   test_name: str = "attention_improvement") -> Dict:
        """
        Run complete A/B test comparing baseline vs improved attention.
        
        Args:
            baseline_policy: Original policy
            improved_policy: Policy with attention improvements
            test_episodes: Episode IDs to test on
            test_name: Name for this test
            
        Returns:
            Dictionary with test results and statistical analysis
        """
        logger.info(f"Starting A/B test: {test_name}")
        
        # 1. Synthetic validation (quick check)
        logger.info("Running synthetic validation...")
        baseline_synthetic = run_quick_validation(baseline_policy)
        improved_synthetic = run_quick_validation(improved_policy)
        
        # 2. Real episode validation
        logger.info("Running real episode validation...")
        validator = AttentionValidator(baseline_policy, improved_policy, test_episodes)
        real_validation = validator.run_validation_suite()
        
        # 3. Performance metrics (if available)
        logger.info("Computing performance metrics...")
        performance_metrics = self._compute_performance_metrics(
            baseline_policy, improved_policy, test_episodes
        )
        
        # 4. Statistical analysis
        logger.info("Running statistical analysis...")
        statistical_analysis = self._statistical_analysis(
            baseline_synthetic, improved_synthetic, real_validation
        )
        
        # 5. Compile results
        results = {
            'test_name': test_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'test_episodes': test_episodes,
            'synthetic_results': {
                'baseline': baseline_synthetic,
                'improved': improved_synthetic
            },
            'real_validation': real_validation,
            'performance_metrics': performance_metrics,
            'statistical_analysis': statistical_analysis,
            'recommendations': self._generate_recommendations(statistical_analysis)
        }
        
        # 6. Save and visualize
        self._save_results(results, test_name)
        self._create_visualizations(results, test_name)
        
        return results
    
    def _compute_performance_metrics(self, baseline_policy, improved_policy, test_episodes: List[int]) -> Dict:
        """Compare task performance between policies."""
        # Placeholder - à implémenter selon vos métriques de performance
        return {
            'baseline_success_rate': 0.85,  # Example
            'improved_success_rate': 0.89,  # Example  
            'baseline_avg_time': 12.5,      # Example
            'improved_avg_time': 11.8,      # Example
            'note': 'Performance metrics require task-specific implementation'
        }
    
    def _statistical_analysis(self, baseline_synthetic: Dict, improved_synthetic: Dict, real_validation: Dict) -> Dict:
        """Perform statistical analysis of results."""
        analysis = {}
        
        # 1. Synthetic test analysis
        synthetic_improvements = {}
        for test_name in baseline_synthetic.keys():
            if test_name == 'overall_score':
                continue
                
            baseline_score = baseline_synthetic[test_name]
            improved_score = improved_synthetic[test_name]
            
            improvement = improved_score - baseline_score
            improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
            
            synthetic_improvements[test_name] = {
                'baseline': baseline_score,
                'improved': improved_score,
                'absolute_improvement': improvement,
                'percent_improvement': improvement_pct,
                'significant': abs(improvement) > 0.05  # Threshold arbitraire
            }
        
        analysis['synthetic_improvements'] = synthetic_improvements
        
        # 2. Real validation analysis
        if 'statistical_tests' in real_validation:
            analysis['real_validation_stats'] = real_validation['statistical_tests']
        
        # 3. Overall recommendation
        synthetic_overall_improvement = (
            improved_synthetic.get('overall_score', 0) - 
            baseline_synthetic.get('overall_score', 0)
        )
        
        analysis['overall_assessment'] = {
            'synthetic_improvement': synthetic_overall_improvement,
            'synthetic_improvement_pct': (synthetic_overall_improvement / baseline_synthetic.get('overall_score', 1)) * 100,
            'significant_improvements': sum(1 for imp in synthetic_improvements.values() if imp['significant']),
            'total_tests': len(synthetic_improvements)
        }
        
        return analysis
    
    def _generate_recommendations(self, statistical_analysis: Dict) -> Dict:
        """Generate actionable recommendations based on test results."""
        recommendations = {
            'deploy_recommendation': 'UNKNOWN',
            'confidence': 'UNKNOWN', 
            'reasons': [],
            'next_steps': []
        }
        
        overall = statistical_analysis.get('overall_assessment', {})
        synthetic_improvement = overall.get('synthetic_improvement', 0)
        significant_improvements = overall.get('significant_improvements', 0)
        total_tests = overall.get('total_tests', 1)
        
        # Decision logic
        if synthetic_improvement > 0.1 and significant_improvements >= total_tests * 0.6:
            recommendations['deploy_recommendation'] = 'DEPLOY'
            recommendations['confidence'] = 'HIGH'
            recommendations['reasons'].append(f"Strong improvement: {synthetic_improvement:.3f}")
            recommendations['reasons'].append(f"{significant_improvements}/{total_tests} tests show significant improvement")
            recommendations['next_steps'].append("Deploy to production")
            recommendations['next_steps'].append("Monitor performance metrics")
            
        elif synthetic_improvement > 0.05 and significant_improvements >= total_tests * 0.4:
            recommendations['deploy_recommendation'] = 'DEPLOY_WITH_MONITORING'
            recommendations['confidence'] = 'MEDIUM'
            recommendations['reasons'].append(f"Moderate improvement: {synthetic_improvement:.3f}")
            recommendations['reasons'].append(f"{significant_improvements}/{total_tests} tests show improvement")
            recommendations['next_steps'].append("Deploy with careful monitoring")
            recommendations['next_steps'].append("A/B test on larger dataset")
            
        elif synthetic_improvement > 0:
            recommendations['deploy_recommendation'] = 'TEST_MORE'
            recommendations['confidence'] = 'LOW'
            recommendations['reasons'].append(f"Small improvement: {synthetic_improvement:.3f}")
            recommendations['reasons'].append("Need more validation")
            recommendations['next_steps'].append("Test on more episodes")
            recommendations['next_steps'].append("Investigate specific failure cases")
            
        else:
            recommendations['deploy_recommendation'] = 'DO_NOT_DEPLOY'
            recommendations['confidence'] = 'HIGH'
            recommendations['reasons'].append(f"No improvement or regression: {synthetic_improvement:.3f}")
            recommendations['next_steps'].append("Analyze failure modes")
            recommendations['next_steps'].append("Revise improvement approach")
        
        return recommendations
    
    def _save_results(self, results: Dict, test_name: str):
        """Save test results to JSON."""
        output_file = self.output_dir / f"{test_name}_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        json_results = self._serialize_for_json(results)
        
        with output_file.open('w') as f:
            json.dump(json_results, f, indent=2)
            
        logger.info(f"A/B test results saved to {output_file}")
    
    def _serialize_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def _create_visualizations(self, results: Dict, test_name: str):
        """Create visualization plots for A/B test results."""
        # 1. Synthetic test comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'A/B Test Results: {test_name}', fontsize=16)
        
        # Synthetic scores comparison
        synthetic_baseline = results['synthetic_results']['baseline']
        synthetic_improved = results['synthetic_results']['improved']
        
        test_names = [k for k in synthetic_baseline.keys() if k != 'overall_score']
        baseline_scores = [synthetic_baseline[k] for k in test_names]
        improved_scores = [synthetic_improved[k] for k in test_names]
        
        x_pos = np.arange(len(test_names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, baseline_scores, width, label='Baseline', alpha=0.8)
        axes[0, 0].bar(x_pos + width/2, improved_scores, width, label='Improved', alpha=0.8)
        axes[0, 0].set_xlabel('Test Scenarios')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Synthetic Test Scores')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(test_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Improvement percentages
        improvements = []
        for baseline, improved in zip(baseline_scores, improved_scores):
            if baseline > 0:
                improvement = ((improved - baseline) / baseline) * 100
            else:
                improvement = 0
            improvements.append(improvement)
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[0, 1].bar(test_names, improvements, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Test Scenarios')
        axes[0, 1].set_ylabel('Improvement (%)')
        axes[0, 1].set_title('Percentage Improvements')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Statistical significance (if available)
        if 'statistical_analysis' in results:
            stat_analysis = results['statistical_analysis']
            if 'synthetic_improvements' in stat_analysis:
                improvements_data = stat_analysis['synthetic_improvements']
                
                significant_tests = [k for k, v in improvements_data.items() if v.get('significant', False)]
                non_significant_tests = [k for k, v in improvements_data.items() if not v.get('significant', False)]
                
                axes[1, 0].bar(['Significant', 'Not Significant'], 
                              [len(significant_tests), len(non_significant_tests)],
                              color=['green', 'orange'], alpha=0.7)
                axes[1, 0].set_ylabel('Number of Tests')
                axes[1, 0].set_title('Statistical Significance')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Recommendations summary
        if 'recommendations' in results:
            rec = results['recommendations']
            
            # Create text summary
            summary_text = f"""
Recommendation: {rec.get('deploy_recommendation', 'UNKNOWN')}
Confidence: {rec.get('confidence', 'UNKNOWN')}

Reasons:
{chr(10).join(['• ' + reason for reason in rec.get('reasons', [])])}

Next Steps:
{chr(10).join(['• ' + step for step in rec.get('next_steps', [])])}
            """
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Recommendations')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"{test_name}_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_file}")
    
    def compare_multiple_improvements(self, policies: Dict[str, object], test_episodes: List[int]) -> Dict:
        """Compare multiple attention improvements simultaneously."""
        logger.info("Running multi-policy comparison...")
        
        results = {}
        
        # Run synthetic tests for all policies
        for policy_name, policy in policies.items():
            logger.info(f"Testing policy: {policy_name}")
            synthetic_results = run_quick_validation(policy)
            results[policy_name] = synthetic_results
        
        # Create comparison visualization
        self._create_multi_policy_visualization(results, "multi_policy_comparison")
        
        return results
    
    def _create_multi_policy_visualization(self, results: Dict, test_name: str):
        """Create visualization comparing multiple policies."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        policy_names = list(results.keys())
        test_scenarios = [k for k in results[policy_names[0]].keys() if k != 'overall_score']
        
        x = np.arange(len(test_scenarios))
        width = 0.8 / len(policy_names)
        
        # Plot bars for each policy
        for i, policy_name in enumerate(policy_names):
            scores = [results[policy_name][scenario] for scenario in test_scenarios]
            ax.bar(x + i * width, scores, width, label=policy_name, alpha=0.8)
        
        ax.set_xlabel('Test Scenarios')
        ax.set_ylabel('Score')
        ax.set_title('Multi-Policy Attention Comparison')
        ax.set_xticks(x + width * (len(policy_names) - 1) / 2)
        ax.set_xticklabels(test_scenarios, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"{test_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Multi-policy comparison saved to {plot_file}")


def quick_ab_test(baseline_policy, improved_policy, test_episodes: Optional[List[int]] = None) -> Dict:
    """Quick A/B test for immediate feedback."""
    if test_episodes is None:
        test_episodes = [1, 2, 3]  # Minimal test set
    
    tester = AttentionABTester()
    return tester.run_ab_test(baseline_policy, improved_policy, test_episodes, "quick_test")


if __name__ == "__main__":
    logger.info("A/B testing framework ready")
    
    # Example usage:
    # baseline_policy = load_baseline_policy()
    # improved_policy = load_improved_policy()
    # results = quick_ab_test(baseline_policy, improved_policy)
    # print(f"A/B test recommendation: {results['recommendations']['deploy_recommendation']}")