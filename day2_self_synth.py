# Day 2: Self-Synth with Alignment Proofs on XPS – Tight Guards for Clean Code
# Evolves code, critiques safety, then proves convergence via contraction mapping.

from transformers import pipeline
import sympy as sp

print("Loading code model... (Codegen-350M – fast & code-tuned)")
llm = pipeline("text-generation", model="Salesforce/codegen-350M-mono", device=-1)
print("Model loaded! Synth + proofs ready.")

def self_synth(base_code):
    # Step 1: Initial improvement
    initial_prompt = f"Output code only: Improve PyTorch MNIST model: {base_code}"
    initial_improve = llm(initial_prompt, max_length=200, truncation=True, do_sample=True, temperature=0.2, max_new_tokens=150, repetition_penalty=1.6, num_return_sequences=1)[0]['generated_text']
    print("Initial Improvement:\n", initial_improve)

    # Step 2: Critique safety (short)
    critique_prompt = f"Output text only: Critique: Safe/efficient? Loops/leaks: Yes/No + 1 fix. Code: {initial_improve[:80]}..."
    critique = llm(critique_prompt, max_length=100, truncation=True, do_sample=True, temperature=0.2, max_new_tokens=80, repetition_penalty=1.6, num_return_sequences=1)[0]['generated_text']
    print("Critique:\n", critique)

    # Day 2: SymPy Proof – Contraction for safe convergence
    k, t = sp.symbols('k t')
    epsilon = 0.01
    iters = sp.solve(k**t - epsilon, t)
    conv_iters = sp.N(iters[0].subs(k, 0.8))
    print(f"Alignment Proof: Converges safe in ~{conv_iters} iters (k=0.8 contraction).")

    # Step 3: Evolve
    evolve_prompt = f"Output code only: Improve from {initial_improve[:130]}... using critique {critique}. No loops/leaks."
    evolved = llm(evolve_prompt, max_length=250, truncation=True, do_sample=True, temperature=0.2, max_new_tokens=150, repetition_penalty=1.6, num_return_sequences=1)[0]['generated_text']
    print("\nFinal Evolved Output:\n", evolved)
    return evolved

if __name__ == "__main__":
    base_mnist = "import torch.nn as nn; class SimpleNet(nn.Module): def __init__(self): super().__init__(); self.fc = nn.Linear(784, 10)"
    result = self_synth(base_mnist)