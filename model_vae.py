# model_vae.py
"""
VAE minimal pour séquences de tokens (demo pédagogique).
Ce wrapper implémente : encode -> latent -> sample -> decode.
Il utilise un tokenizer très simple (char-level) pour garder le code court.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# --- tokenizer minimal char-level ---
class CharTokenizer:
    def __init__(self, chars=None, maxlen=256):
        if chars is None:
            chars = sorted(list(set([chr(i) for i in range(32, 127)])))
        self.chars = chars
        self.char2idx = {c:i+1 for i,c in enumerate(self.chars)}  # 0 reserved for padding
        self.idx2char = {i+1:c for i,c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 1
        self.maxlen = maxlen

    def encode(self, s):
        arr = np.zeros(self.maxlen, dtype=np.int32)
        for i,ch in enumerate(s[:self.maxlen]):
            arr[i] = self.char2idx.get(ch, 0)
        return arr

    def decode(self, arr):
        s = ''.join(self.idx2char.get(int(i), '') for i in arr if i!=0)
        return s


# --- VAE model custom ---
class VAEModel(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.sparse_categorical_crossentropy(data, reconstruction),
                    axis=0
                )
            )
            
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# --- VAE wrapper ---
class VAEWrapper:
    model_path = 'vae_demo.h5'
    tok_path = 'vae_tokenizer.json'

    def __init__(self, tokenizer: CharTokenizer, encoder: Model, decoder: Model, vae: Model):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae

    @staticmethod
    def build(tokenizer: CharTokenizer, latent_dim=32):
        maxlen = tokenizer.maxlen
        vocab = tokenizer.vocab_size

        # Encoder
        inputs = layers.Input(shape=(maxlen,), dtype='int32')
        x = layers.Embedding(vocab, 32, mask_zero=True)(inputs)
        x = layers.Bidirectional(layers.LSTM(128))(x)
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_logvar = layers.Dense(latent_dim, name='z_logvar')(x)

        def sample_z(args):
            mean, logvar = args
            eps = tf.random.normal(shape=tf.shape(mean))
            return mean + tf.exp(0.5 * logvar) * eps

        z = layers.Lambda(sample_z, name='z')([z_mean, z_logvar])
        encoder = Model(inputs, [z_mean, z_logvar, z], name='encoder')

        # Decoder
        latent_inputs = layers.Input(shape=(latent_dim,))
        x = layers.RepeatVector(maxlen)(latent_inputs)
        x = layers.LSTM(128, return_sequences=True)(x)
        outputs = layers.TimeDistributed(layers.Dense(vocab, activation='softmax'))(x)
        decoder = Model(latent_inputs, outputs, name='decoder')

        # VAE model complet
        vae = VAEModel(encoder, decoder)
        vae.compile(optimizer='adam')

        wrapper = VAEWrapper(tokenizer, encoder, decoder, vae)
        return wrapper, vae

    # convenience training wrapper
    @staticmethod
    def train_on_texts(texts, epochs=10, batch_size=32, latent_dim=32, maxlen=256):
        tok = CharTokenizer(maxlen=maxlen)
        # build model
        wrapper, vae = VAEWrapper.build(tok, latent_dim=latent_dim)
        # prepare data
        X = np.stack([tok.encode(t) for t in texts])
        vae.fit(X, epochs=epochs, batch_size=batch_size)
        # save encoder and decoder weights separately
        wrapper.encoder.save_weights('vae_encoder.weights.h5')
        wrapper.decoder.save_weights('vae_decoder.weights.h5')
        # save tokenizer config
        import json
        with open('vae_tokenizer.json', 'w') as f:
            json.dump({'chars': wrapper.tokenizer.chars, 'maxlen': wrapper.tokenizer.maxlen}, f)
        # return wrapper
        return wrapper

    # generation: encode, add noise, decode
    def generate_variant(self, text: str, temperature: float = 1.0):
        enc = self.encoder
        dec = self.decoder
        tok = self.tokenizer
        x = np.expand_dims(tok.encode(text), axis=0)
        z_mean, z_logvar, z = enc.predict(x, verbose=0)
        # sample new z by adding gaussian noise
        z_new = z_mean + np.random.normal(scale=0.5, size=z_mean.shape)
        # decode
        preds = dec.predict(z_new, verbose=0)[0]  # shape (maxlen, vocab)
        # greedy/sample with temperature
        if temperature <= 0:
            idxs = preds.argmax(axis=-1)
        else:
            # sample per timestep
            idxs = []
            for p in preds:
                p = np.log(np.clip(p,1e-9,1.0)) / temperature
                p = np.exp(p) / np.sum(np.exp(p))
                idx = np.random.choice(len(p), p=p)
                idxs.append(idx)
            idxs = np.array(idxs)
        return tok.decode(idxs)

    @classmethod
    def load_default(cls):
        # For the TP, we try to load pre-trained weights; if not present we raise so the caller can handle
        # Here we create a small tokenizer and an untrained model to avoid failing completely.
        tok = CharTokenizer(maxlen=256)
        wrapper, vae = VAEWrapper.build(tok, latent_dim=32)
        # Attempt to load weights if exist
        import os
        if os.path.exists('vae_demo_weights.weights.h5'):
            vae.load_weights('vae_demo_weights.weights.h5')
        return wrapper