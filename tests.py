# tests.py
"""
Tests d'obfuscation: montre original -> variantes.
Lancer: python tests.py
"""
from obfuscate import obfuscate_snippet
import ast
import sys

snippets = [
    # snippet 1
    "def add(a, b):\n    return a + b\n\nprint(add(2,3))",
    # snippet 2
    "def greet(name):\n    msg = f'Hello {name}!'\n    return msg\n\nprint(greet('Alice'))",
    # snippet 3
    "def factorial(n):\n    if n<=1:\n        return 1\n    return n*factorial(n-1)\n\nprint(factorial(5))"
]

def safe_exec(code_str):
    # limited sandbox: we only parse AST to ensure syntax validity; *no* exec for safety in classroom
    try:
        ast.parse(code_str)
        return True, "syntax_ok"
    except Exception as e:
        return False, str(e)

def main():
    for i, s in enumerate(snippets, 1):
        print("="*40)
        print(f"SNIPPET {i} (original):\n{s}\n")
        variants = obfuscate_snippet(s, n_variants=2, use_vae=False, seed=i)
        for j, v in enumerate(variants, 1):
            ok, msg = safe_exec(v)
            print(f"--- variante {j} (validité syntaxique: {ok}) ---\n{v}\n")
        print("\n")
    print("Fin des tests. Pour tester VAE, entraînez d'abord avec train.py puis relancez tests.py avec use_vae=True.")
if __name__ == '__main__':
    main()
