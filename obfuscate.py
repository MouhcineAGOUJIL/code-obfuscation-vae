# obfuscate.py
"""
Pipeline d'obfuscation combinant méthodes simples et VAE.
Usage:
    from obfuscate import obfuscate_snippet
    obfuscated_variants = obfuscate_snippet(code_str, n_variants=3, use_vae=True)
"""

import base64
import re
import random
from typing import List

# simple tokenizer using Python tokenize could be used, but for clarity we do a light-weight token split
IDENT_RE = re.compile(r'\b[_a-zA-Z]\w*\b')

def quick_rename(code: str, seed: int = None) -> str:
    """Renomme les identifiants simples par des noms courts (a1, a2...)."""
    if seed is not None:
        random.seed(seed)
    names = list(dict.fromkeys(IDENT_RE.findall(code)))  # preserve order
    # we will avoid renaming keywords (a minimal list)
    keywords = set([
        'def','return','if','else','for','while','in','import','from','as','class',
        'try','except','finally','with','pass','break','continue','lambda','True','False','None'
    ])
    mapping = {}
    counter = 0
    for n in names:
        if n in keywords or n.isdigit():
            continue
        new = f'_{counter}'
        counter += 1
        mapping[n] = new
    # replace whole words
    def repl(m):
        tok = m.group(0)
        return mapping.get(tok, tok)
    return IDENT_RE.sub(repl, code)

def add_dummy_code(code: str) -> str:
    """Ajoute du code mort (dead code) pour obscurcir."""
    lines = code.split('\n')
    # Ajouter quelques variables inutiles au début
    dummy = [
        "# Auto-generated code",
        "_unused_var_1 = lambda x: x * 2",
        "_unused_var_2 = [i for i in range(10)]",
    ]
    return '\n'.join(dummy + [''] + lines)

def space_obfuscation(code: str) -> str:
    """Modifie les espaces et ajoute des commentaires."""
    # Ajouter des commentaires aléatoires
    lines = code.split('\n')
    result = []
    for line in lines:
        result.append(line)
        if random.random() < 0.3 and line.strip() and not line.strip().startswith('#'):
            result.append("    # ...")
    return '\n'.join(result)

# VAE interface (light wrapper) – expects model_vae.py to implement encode/decode
try:
    from model_vae import VAEWrapper
    _vae_available = True
except Exception:
    VAEWrapper = None
    _vae_available = False

def obfuscate_snippet(code: str, n_variants: int = 3, use_vae: bool = True, seed: int = None) -> List[str]:
    """Retourne une liste de variantes obfusquées."""
    variants = []
    
    # 1) simple rule-based variants
    variants.append(quick_rename(code, seed=seed))
    variants.append(add_dummy_code(code))
    
    if seed is not None:
        random.seed(seed + 1)
    variants.append(space_obfuscation(code))

    # 2) optionally apply VAE
    if use_vae and _vae_available:
        try:
            vae = VAEWrapper.load_default()
            # tokenize simply by splitting on whitespace/punct for the demo
            for i in range(n_variants):
                v = vae.generate_variant(code, temperature=0.6 + 0.2 * (i / max(1, n_variants - 1)))
                variants.append(v)
        except Exception as e:
            print(f"VAE generation failed: {e}")

    # deduplicate and return
    seen = set()
    unique = []
    for v in variants:
        if v not in seen:
            unique.append(v)
            seen.add(v)
    return unique