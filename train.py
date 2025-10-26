# train.py
"""
Exemple d'entraînement minimal du VAE sur des snippets fournis.
Lancer:
    python train.py
"""
from model_vae import VAEWrapper
import glob
import os

# collecte des snippets (fichiers .py dans data/)
def load_snippets(folder='data'):
    texts = []
    if not os.path.exists(folder):
        os.makedirs(folder)
    for path in glob.glob(os.path.join(folder, '*.py')):
        with open(path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts

if __name__ == '__main__':
    texts = load_snippets('data')
    if len(texts) < 5:
        print("Attention: dataset petit. Ajoute des snippets dans data/*.py pour un meilleur entraînement.")
    wrapper = VAEWrapper.train_on_texts(texts, epochs=15, batch_size=8, latent_dim=32, maxlen=256)
    print("Entraînement terminé. Poids sauvegardés (vae_encoder.weights.h5, vae_decoder.weights.h5).")