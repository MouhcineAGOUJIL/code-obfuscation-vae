# visualize_metrics.py
"""
Visualisation des métriques pour le VAE d'obfuscation de code.
Ce script génère des graphiques pertinents pour évaluer le modèle.
"""

import matplotlib.pyplot as plt
import numpy as np
import ast
from obfuscate import obfuscate_snippet
from model_vae import VAEWrapper
import os

# Configuration pour des graphiques plus jolis
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class VAEMetrics:
    def __init__(self):
        self.training_history = {
            'epoch': [],
            'loss': [],
            'reconstruction_loss': [],
            'kl_loss': []
        }
        
    def parse_training_log(self, log_file='training_log.txt'):
        """Parse le log d'entraînement pour extraire les métriques."""
        # Si vous avez sauvegardé l'historique, chargez-le ici
        # Pour la démo, on va créer des données synthétiques basées sur votre sortie
        epochs = 15
        self.training_history['epoch'] = list(range(1, epochs + 1))
        
        # Données approximatives basées sur votre sortie d'entraînement
        self.training_history['loss'] = [13.66, 14.21, 13.92, 14.17, 13.91, 
                                         13.17, 13.00, 13.44, 13.42, 13.11,
                                         12.77, 11.30, 10.48, 10.82, 9.60]
        self.training_history['reconstruction_loss'] = [13.66, 14.21, 13.92, 14.17, 13.91,
                                                        13.16, 12.99, 13.44, 13.42, 13.09,
                                                        12.71, 11.07, 9.86, 9.66, 8.11]
        self.training_history['kl_loss'] = [0.00003, 0.00006, 0.00014, 0.0003, 0.00057,
                                            0.0011, 0.0020, 0.0037, 0.0076, 0.0182,
                                            0.0599, 0.2292, 0.6155, 1.1552, 1.4877]
    
    def plot_training_curves(self, save_path='examples/training_curves.png'):
        """Graphique 1: Courbes d'entraînement (Loss)."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Métriques d\'Entraînement du VAE', fontsize=16, fontweight='bold')
        
        # 1. Loss totale
        axes[0, 0].plot(self.training_history['epoch'], 
                       self.training_history['loss'], 
                       'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Perte Totale (Total Loss)', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(self.training_history['epoch'], 
                                self.training_history['loss'], 
                                alpha=0.3)
        
        # 2. Reconstruction Loss
        axes[0, 1].plot(self.training_history['epoch'], 
                       self.training_history['reconstruction_loss'], 
                       'g-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Perte de Reconstruction', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Reconstruction Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].fill_between(self.training_history['epoch'], 
                                self.training_history['reconstruction_loss'], 
                                alpha=0.3, color='green')
        
        # 3. KL Divergence
        axes[1, 0].plot(self.training_history['epoch'], 
                       self.training_history['kl_loss'], 
                       'r-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Divergence KL', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].set_yscale('log')  # Échelle log car très petit au début
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].fill_between(self.training_history['epoch'], 
                                self.training_history['kl_loss'], 
                                alpha=0.3, color='red')
        
        # 4. Les deux pertes ensemble
        ax2 = axes[1, 1]
        ax2.plot(self.training_history['epoch'], 
                self.training_history['reconstruction_loss'], 
                'g-o', label='Reconstruction', linewidth=2, markersize=5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Reconstruction Loss', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.grid(True, alpha=0.3)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.training_history['epoch'], 
                     self.training_history['kl_loss'], 
                     'r-s', label='KL Divergence', linewidth=2, markersize=5)
        ax2_twin.set_ylabel('KL Loss', color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        axes[1, 1].set_title('Reconstruction vs KL Divergence', fontweight='bold')
        
        # Légendes
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        os.makedirs('examples', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
        plt.close()
    
    def evaluate_obfuscation_quality(self, test_snippets, save_path='examples/obfuscation_quality.png'):
        """Graphique 2: Qualité de l'obfuscation."""
        results = {
            'snippet': [],
            'syntactically_valid': [],
            'avg_similarity': [],
            'num_variants': []
        }
        
        for i, snippet in enumerate(test_snippets, 1):
            variants = obfuscate_snippet(snippet, n_variants=3, use_vae=True, seed=i)
            valid_count = 0
            
            for variant in variants:
                try:
                    ast.parse(variant)
                    valid_count += 1
                except:
                    pass
            
            results['snippet'].append(f'Snippet {i}')
            results['syntactically_valid'].append((valid_count / len(variants)) * 100)
            results['num_variants'].append(len(variants))
            
            # Similarité simple (ratio de longueur)
            avg_sim = np.mean([abs(len(v) - len(snippet)) / len(snippet) for v in variants])
            results['avg_similarity'].append(avg_sim * 100)
        
        # Créer le graphique
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Qualité de l\'Obfuscation', fontsize=16, fontweight='bold')
        
        x = np.arange(len(results['snippet']))
        width = 0.6
        
        # 1. Validité syntaxique
        bars1 = axes[0].bar(x, results['syntactically_valid'], width, color='green', alpha=0.7)
        axes[0].set_ylabel('% Variantes Valides')
        axes[0].set_title('Validité Syntaxique', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results['snippet'])
        axes[0].set_ylim(0, 110)
        axes[0].axhline(y=80, color='r', linestyle='--', label='Seuil acceptable (80%)')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%', ha='center', va='bottom')
        
        # 2. Nombre de variantes générées
        bars2 = axes[1].bar(x, results['num_variants'], width, color='blue', alpha=0.7)
        axes[1].set_ylabel('Nombre de Variantes')
        axes[1].set_title('Variantes Générées', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results['snippet'])
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 3. Différence de longueur (proxy pour la modification)
        bars3 = axes[2].bar(x, results['avg_similarity'], width, color='orange', alpha=0.7)
        axes[2].set_ylabel('% Différence de Longueur')
        axes[2].set_title('Niveau de Modification', fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(results['snippet'])
        axes[2].grid(axis='y', alpha=0.3)
        
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé: {save_path}")
        plt.close()
        
        return results
    
    def plot_latent_space_distribution(self, test_snippets, save_path='examples/latent_space.png'):
        """Graphique 3: Distribution de l'espace latent."""
        try:
            vae = VAEWrapper.load_default()
            
            # Encoder les snippets
            latent_means = []
            latent_stds = []
            
            for snippet in test_snippets:
                tok = vae.tokenizer
                x = np.expand_dims(tok.encode(snippet), axis=0)
                z_mean, z_logvar, _ = vae.encoder.predict(x, verbose=0)
                latent_means.append(z_mean[0])
                latent_stds.append(np.exp(0.5 * z_logvar[0]))
            
            latent_means = np.array(latent_means)
            latent_stds = np.array(latent_stds)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('Distribution de l\'Espace Latent', fontsize=16, fontweight='bold')
            
            # 1. Distribution des moyennes
            axes[0].violinplot([latent_means[:, i] for i in range(min(8, latent_means.shape[1]))],
                              showmeans=True, showmedians=True)
            axes[0].set_title('Distribution des Moyennes Latentes', fontweight='bold')
            axes[0].set_xlabel('Dimension Latente')
            axes[0].set_ylabel('Valeur')
            axes[0].grid(axis='y', alpha=0.3)
            
            # 2. Heatmap des moyennes
            im = axes[1].imshow(latent_means.T[:10], aspect='auto', cmap='coolwarm')
            axes[1].set_title('Représentations Latentes (10 premières dim)', fontweight='bold')
            axes[1].set_xlabel('Snippet')
            axes[1].set_ylabel('Dimension Latente')
            axes[1].set_xticks(range(len(test_snippets)))
            axes[1].set_xticklabels([f'S{i+1}' for i in range(len(test_snippets))])
            plt.colorbar(im, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Graphique sauvegardé: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Impossible de générer le graphique d'espace latent: {e}")
    
    def generate_summary_report(self, test_snippets):
        """Génère un rapport complet avec tous les graphiques."""
        print("\n" + "="*60)
        print("📊 GÉNÉRATION DES MÉTRIQUES ET GRAPHIQUES")
        print("="*60 + "\n")
        
        # 1. Courbes d'entraînement
        print("1️⃣ Génération des courbes d'entraînement...")
        self.parse_training_log()
        self.plot_training_curves()
        
        # 2. Qualité de l'obfuscation
        print("\n2️⃣ Évaluation de la qualité de l'obfuscation...")
        results = self.evaluate_obfuscation_quality(test_snippets)
        
        # 3. Espace latent
        print("\n3️⃣ Analyse de l'espace latent...")
        self.plot_latent_space_distribution(test_snippets)
        
        # Résumé textuel
        print("\n" + "="*60)
        print("📈 RÉSUMÉ DES RÉSULTATS")
        print("="*60)
        print(f"\n🎯 Perte finale: {self.training_history['loss'][-1]:.2f}")
        print(f"🔄 Reconstruction loss finale: {self.training_history['reconstruction_loss'][-1]:.2f}")
        print(f"📊 KL divergence finale: {self.training_history['kl_loss'][-1]:.4f}")
        
        avg_valid = np.mean(results['syntactically_valid'])
        print(f"\n✅ Validité syntaxique moyenne: {avg_valid:.1f}%")
        print(f"🔢 Variantes générées en moyenne: {np.mean(results['num_variants']):.1f}")
        
        print("\n" + "="*60)
        print("✨ Tous les graphiques ont été sauvegardés dans le dossier 'examples/'")
        print("="*60 + "\n")


def main():
    """Fonction principale."""
    # Snippets de test
    test_snippets = [
        "def add(a, b):\n    return a + b\n\nprint(add(2,3))",
        "def greet(name):\n    msg = f'Hello {name}!'\n    return msg\n\nprint(greet('Alice'))",
        "def factorial(n):\n    if n<=1:\n        return 1\n    return n*factorial(n-1)\n\nprint(factorial(5))"
    ]
    
    # Créer l'évaluateur
    metrics = VAEMetrics()
    
    # Générer tous les graphiques et le rapport
    metrics.generate_summary_report(test_snippets)


if __name__ == '__main__':
    main()