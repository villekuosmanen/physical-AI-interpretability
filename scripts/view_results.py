#!/usr/bin/env python3
"""
Visualiseur simple pour les résultats de validation.
Affiche les résultats de manière claire et interactive.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import print as rprint


class ValidationResultsViewer:
    """Visualise les résultats de validation de manière élégante."""
    
    def __init__(self):
        self.console = Console()
        
    def load_results(self, results_path: str) -> Dict:
        """Charge les résultats depuis un fichier JSON."""
        path = Path(results_path)
        
        if not path.exists():
            self.console.print(f"[red]Erreur: Fichier non trouvé: {results_path}[/red]")
            return None
            
        with path.open('r') as f:
            return json.load(f)
    
    def display_synthetic_results(self, results: Dict):
        """Affiche les résultats des tests synthétiques."""
        if 'synthetic_results' not in results:
            self.console.print("[yellow]Pas de résultats synthétiques trouvés[/yellow]")
            return
            
        synthetic = results['synthetic_results']
        
        # Créer un tableau
        table = Table(title="🧪 Tests Synthétiques", show_header=True, header_style="bold magenta")
        table.add_column("Test", style="cyan", width=25)
        table.add_column("Baseline", justify="right", style="yellow")
        table.add_column("Improved", justify="right", style="green")
        table.add_column("Amélioration", justify="right", style="bold")
        table.add_column("Status", justify="center")
        
        baseline_results = synthetic.get('baseline', {})
        improved_results = synthetic.get('improved', {})
        
        for test_name in baseline_results:
            if test_name == 'overall_score':
                continue
                
            baseline_score = baseline_results.get(test_name, 0)
            improved_score = improved_results.get(test_name, 0)
            
            if baseline_score > 0:
                improvement = ((improved_score - baseline_score) / baseline_score) * 100
            else:
                improvement = 0
                
            # Status avec emoji
            if improvement > 10:
                status = "✅ [green]BETTER[/green]"
            elif improvement > 0:
                status = "➕ [yellow]SLIGHT[/yellow]"
            else:
                status = "❌ [red]WORSE[/red]"
                
            table.add_row(
                test_name,
                f"{baseline_score:.3f}",
                f"{improved_score:.3f}",
                f"{improvement:+.1f}%",
                status
            )
        
        # Overall score
        baseline_overall = baseline_results.get('overall_score', 0)
        improved_overall = improved_results.get('overall_score', 0)
        overall_improvement = ((improved_overall - baseline_overall) / baseline_overall) * 100 if baseline_overall > 0 else 0
        
        table.add_section()
        table.add_row(
            "[bold]OVERALL SCORE[/bold]",
            f"[bold]{baseline_overall:.3f}[/bold]",
            f"[bold]{improved_overall:.3f}[/bold]",
            f"[bold]{overall_improvement:+.1f}%[/bold]",
            "🎯" if overall_improvement > 5 else "⚠️"
        )
        
        self.console.print(table)
    
    def display_recommendations(self, results: Dict):
        """Affiche les recommandations finales."""
        if 'recommendations' not in results:
            return
            
        rec = results['recommendations']
        
        # Couleur selon la recommandation
        deploy_rec = rec.get('deploy_recommendation', 'UNKNOWN')
        if deploy_rec == 'DEPLOY':
            color = "green"
            emoji = "✅"
        elif deploy_rec == 'DEPLOY_WITH_MONITORING':
            color = "yellow"
            emoji = "⚠️"
        elif deploy_rec == 'TEST_MORE':
            color = "yellow"
            emoji = "🔄"
        else:
            color = "red"
            emoji = "❌"
        
        # Panel principal
        content = f"""[bold {color}]{emoji} {deploy_rec}[/bold {color}]

[bold]Confiance:[/bold] {rec.get('confidence', 'UNKNOWN')}

[bold]Raisons:[/bold]
"""
        for reason in rec.get('reasons', []):
            content += f"  • {reason}\n"
            
        content += "\n[bold]Prochaines Étapes:[/bold]\n"
        for step in rec.get('next_steps', []):
            content += f"  → {step}\n"
        
        panel = Panel(content, title="📋 Recommandation Finale", border_style=color)
        self.console.print(panel)
    
    def display_statistical_tests(self, results: Dict):
        """Affiche les tests statistiques."""
        if 'real_validation' not in results:
            return
            
        stats = results['real_validation'].get('statistical_tests', {})
        if not stats:
            return
            
        table = Table(title="📊 Tests Statistiques", show_header=True)
        table.add_column("Métrique", style="cyan")
        table.add_column("t-statistic", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Significatif", justify="center")
        table.add_column("Effect Size", justify="right")
        
        for metric, test_result in stats.items():
            if 'error' in test_result:
                continue
                
            sig = "✅" if test_result.get('significant', False) else "❌"
            
            table.add_row(
                metric,
                f"{test_result.get('t_statistic', 0):.3f}",
                f"{test_result.get('t_pvalue', 1):.4f}",
                sig,
                f"{test_result.get('effect_size', 0):.3f}"
            )
        
        self.console.print(table)
    
    def create_comparison_plot(self, results: Dict, output_path: Optional[str] = None):
        """Crée un graphique de comparaison."""
        if 'synthetic_results' not in results:
            return
            
        synthetic = results['synthetic_results']
        baseline = synthetic.get('baseline', {})
        improved = synthetic.get('improved', {})
        
        # Préparer les données
        tests = [k for k in baseline.keys() if k != 'overall_score']
        baseline_scores = [baseline.get(t, 0) for t in tests]
        improved_scores = [improved.get(t, 0) for t in tests]
        
        # Créer le graphique
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique en barres
        x_pos = np.arange(len(tests))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, baseline_scores, width, label='Baseline', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, improved_scores, width, label='Improved', alpha=0.8)
        
        ax1.set_xlabel('Tests')
        ax1.set_ylabel('Score')
        ax1.set_title('Comparaison des Scores')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(tests, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique radar
        angles = np.linspace(0, 2 * np.pi, len(tests), endpoint=False).tolist()
        baseline_scores_normalized = [(s - min(baseline_scores)) / (max(baseline_scores) - min(baseline_scores)) for s in baseline_scores]
        improved_scores_normalized = [(s - min(improved_scores)) / (max(improved_scores) - min(improved_scores)) for s in improved_scores]
        
        baseline_scores_normalized += baseline_scores_normalized[:1]
        improved_scores_normalized += improved_scores_normalized[:1]
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, baseline_scores_normalized, 'o-', linewidth=2, label='Baseline', color='orange')
        ax2.fill(angles, baseline_scores_normalized, alpha=0.25, color='orange')
        ax2.plot(angles, improved_scores_normalized, 'o-', linewidth=2, label='Improved', color='green')
        ax2.fill(angles, improved_scores_normalized, alpha=0.25, color='green')
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(tests)
        ax2.set_ylim(0, 1)
        ax2.set_title('Profil d\'Attention', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.console.print(f"[green]Graphique sauvé: {output_path}[/green]")
        else:
            plt.show()
    
    def display_summary(self, results_path: str):
        """Affiche un résumé complet des résultats."""
        results = self.load_results(results_path)
        
        if not results:
            return
            
        # Titre
        self.console.rule("[bold blue]Résultats de Validation d'Attention[/bold blue]")
        
        # Info générales
        if 'test_name' in results:
            self.console.print(f"\n[bold]Test:[/bold] {results['test_name']}")
        if 'timestamp' in results:
            self.console.print(f"[bold]Date:[/bold] {results['timestamp']}")
        if 'test_episodes' in results:
            self.console.print(f"[bold]Episodes testés:[/bold] {results['test_episodes']}")
        
        self.console.print()
        
        # Résultats synthétiques
        self.display_synthetic_results(results)
        self.console.print()
        
        # Tests statistiques
        self.display_statistical_tests(results)
        self.console.print()
        
        # Recommandations
        self.display_recommendations(results)


def main():
    """Point d'entrée principal."""
    if len(sys.argv) < 2:
        print("Usage: python view_results.py <results_file.json> [--plot output.png]")
        sys.exit(1)
        
    results_file = sys.argv[1]
    
    viewer = ValidationResultsViewer()
    viewer.display_summary(results_file)
    
    # Optionnel: sauver le graphique
    if '--plot' in sys.argv:
        plot_idx = sys.argv.index('--plot')
        if plot_idx + 1 < len(sys.argv):
            output_path = sys.argv[plot_idx + 1]
            viewer.create_comparison_plot(viewer.load_results(results_file), output_path)


if __name__ == "__main__":
    # Installer rich si pas présent
    try:
        from rich import print
    except ImportError:
        print("Installing rich for better display...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
        
    main()