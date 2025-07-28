#!/usr/bin/env python3
"""
Script principal pour lancer les tests de validation d'attention.
Utilisation simple en ligne de commande.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

# Import des modules de validation
sys.path.append(str(Path(__file__).parent.parent))

from scripts.synthetic_attention_tests import SyntheticAttentionTester
from scripts.ab_test_attention import AttentionABTester
from scripts.validate_attention_improvements import AttentionValidator

# Import de votre policy wrapper
from src.attention_maps import ACTPolicyWithAttention
from lerobot.common.policies.factory import make_policy


def load_policy(policy_path: str, attention_config: dict = None) -> ACTPolicyWithAttention:
    """Charge un policy avec configuration d'attention."""
    logger.info(f"Loading policy from {policy_path}")
    
    # Charger le policy LeRobot standard
    from lerobot.configs.policies import PreTrainedConfig
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    
    # Créer le policy
    # Note: Vous devrez peut-être adapter cette partie selon votre setup
    # dataset_meta = load_dataset_metadata()  # À implémenter
    dataset_meta = {}  # Placeholder
    
    base_policy = make_policy(policy_cfg, ds_meta=dataset_meta)
    
    # Wrapper avec attention
    if attention_config:
        policy_with_attention = ACTPolicyWithAttention(
            base_policy,
            **attention_config
        )
    else:
        policy_with_attention = ACTPolicyWithAttention(base_policy)
    
    return policy_with_attention


def run_synthetic_tests(policy_path: str, output_dir: str = "./synthetic_results"):
    """Lance les tests synthétiques sur un policy."""
    logger.info("=== SYNTHETIC ATTENTION TESTS ===")
    
    # Charger le policy
    policy = load_policy(policy_path)
    
    # Créer le testeur
    tester = SyntheticAttentionTester()
    
    # Lancer les tests
    results = tester.run_all_tests(policy)
    
    # Afficher les résultats
    print("\n📊 Résultats des Tests Synthétiques:")
    print("-" * 50)
    
    for test_name, score in results.items():
        if test_name == 'overall_score':
            continue
            
        # Colorier selon le score
        if score > 0.8:
            status = "✅ EXCELLENT"
            color = "\033[92m"  # Vert
        elif score > 0.7:
            status = "✅ GOOD"
            color = "\033[92m"  # Vert
        elif score > 0.5:
            status = "⚠️  ACCEPTABLE"
            color = "\033[93m"  # Jaune
        else:
            status = "❌ POOR"
            color = "\033[91m"  # Rouge
            
        print(f"{test_name:25} | {color}{score:.3f}{'\033[0m':>10} | {status}")
    
    print("-" * 50)
    overall = results['overall_score']
    
    if overall > 0.7:
        overall_status = "✅ PASS - Ready for deployment"
        color = "\033[92m"
    elif overall > 0.5:
        overall_status = "⚠️  CAUTION - Needs improvement"
        color = "\033[93m"
    else:
        overall_status = "❌ FAIL - Major issues detected"
        color = "\033[91m"
        
    print(f"{'OVERALL SCORE':25} | {color}{overall:.3f}{'\033[0m':>10} | {overall_status}")
    print()
    
    return results


def run_ab_test(baseline_path: str, improved_path: str, 
                episodes: Optional[list] = None,
                output_dir: str = "./ab_results"):
    """Lance un A/B test entre deux policies."""
    logger.info("=== A/B ATTENTION TEST ===")
    
    # Episodes par défaut
    if episodes is None:
        episodes = [1, 3, 5, 7, 10]
    
    # Charger les policies
    logger.info("Loading baseline policy...")
    baseline_policy = load_policy(baseline_path)
    
    logger.info("Loading improved policy...")
    improved_policy = load_policy(improved_path)
    
    # Créer le testeur A/B
    ab_tester = AttentionABTester(output_dir)
    
    # Lancer le test
    results = ab_tester.run_ab_test(
        baseline_policy,
        improved_policy,
        episodes,
        test_name="attention_improvement"
    )
    
    # Afficher les recommandations
    print("\n🔬 Résultats A/B Test:")
    print("-" * 50)
    
    rec = results['recommendations']
    deploy_rec = rec['deploy_recommendation']
    
    # Colorier selon la recommandation
    if deploy_rec == 'DEPLOY':
        color = "\033[92m"  # Vert
        emoji = "✅"
    elif deploy_rec == 'DEPLOY_WITH_MONITORING':
        color = "\033[93m"  # Jaune
        emoji = "⚠️"
    elif deploy_rec == 'TEST_MORE':
        color = "\033[93m"  # Jaune  
        emoji = "🔄"
    else:  # DO_NOT_DEPLOY
        color = "\033[91m"  # Rouge
        emoji = "❌"
    
    print(f"{emoji} Recommendation: {color}{deploy_rec}\033[0m")
    print(f"Confidence: {rec['confidence']}")
    print("\nReasons:")
    for reason in rec['reasons']:
        print(f"  • {reason}")
    print("\nNext Steps:")
    for step in rec['next_steps']:
        print(f"  → {step}")
    print()
    
    return results


def quick_validate(policy_path: str):
    """Validation rapide en une commande."""
    logger.info("=== QUICK VALIDATION ===")
    
    # Tests synthétiques seulement
    results = run_synthetic_tests(policy_path)
    
    # Décision rapide
    overall = results['overall_score']
    
    if overall > 0.7:
        print("\n✅ VALIDATION PASSED - Policy attention looks good!")
        print("   You can proceed with confidence.")
        return 0
    elif overall > 0.5:
        print("\n⚠️  VALIDATION WARNING - Some attention issues detected")
        print("   Consider running full A/B test before deployment.")
        return 1
    else:
        print("\n❌ VALIDATION FAILED - Significant attention problems")
        print("   Do not deploy without fixing issues.")
        return 2


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Validate attention improvements for robotic policies"
    )
    
    # Mode de test
    parser.add_argument('--mode', choices=['synthetic', 'ab-test', 'quick'], 
                       default='quick',
                       help='Test mode: synthetic only, A/B test, or quick validation')
    
    # Chemins des policies
    parser.add_argument('--policy', type=str, 
                       help='Path to policy checkpoint')
    parser.add_argument('--baseline', type=str,
                       help='Path to baseline policy (for A/B test)')
    parser.add_argument('--improved', type=str,
                       help='Path to improved policy (for A/B test)')
    
    # Options
    parser.add_argument('--episodes', type=int, nargs='+',
                       help='Episode IDs to test on')
    parser.add_argument('--output-dir', type=str, default='./validation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Vérifier les arguments selon le mode
    if args.mode == 'synthetic':
        if not args.policy:
            parser.error("--policy required for synthetic tests")
        return run_synthetic_tests(args.policy, args.output_dir)
        
    elif args.mode == 'ab-test':
        if not args.baseline or not args.improved:
            parser.error("--baseline and --improved required for A/B test")
        return run_ab_test(args.baseline, args.improved, args.episodes, args.output_dir)
        
    elif args.mode == 'quick':
        if not args.policy:
            parser.error("--policy required for quick validation")
        return quick_validate(args.policy)


if __name__ == "__main__":
    sys.exit(main())