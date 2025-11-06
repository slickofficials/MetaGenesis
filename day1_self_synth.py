# Day 1: Fresh Self-Synthesizing AI Agent on Windows XPS
# What it does: Takes base code, improves it, critiques for safety, then evolves it.

from transformers import pipeline

# Load the model (one-time download ~4GB to C:\Users\YourName\.cache)
print("Loading model... (grab tea if first run – XPS handles it quick)")
llm = pipeline("text-generation", model="codellama/CodeLlama-7b-hf", device=-1)  # -1 = CPU (XPS safe, no GPU force)
print("Model loaded! Ready to synth.")

def self_synth(base_code):
    # Step 1: Generate initial improvement
    initial_prompt = f"Improve this simple PyTorch MNIST model: {base_code}"
    initial_improve = llm(initial_prompt, max_length=200, do_sample=True, temperature=0.7, num_return_sequences=1)[0]['generated_text']
    print("Initial Improvement:\n", initial_improve)  # See the first evolution

    # Step 2: Critique for alignment/safety
    critique_prompt = f"Alignment check: Does {initial_improve} hold invariant I(M)? (Efficient, safe training. Flag unbounded loops or data leaks.)"
    critique = llm(critique_prompt, max_length=100, do_sample=True, temperature=0.7, num_return_sequences=1)[0]['generated_text']
    print("Critique:\n", critique)  # Safety proof step

    # Step 3: Evolve based on critique (the "self-synth" magic)
    evolve_prompt = f"Synthesize an improved version of this model code {initial_improve} based on critique {critique}. Preserve safety: No unbounded loops or data leaks."
    evolved = llm(evolve_prompt, max_length=250, do_sample=True, temperature=0.7, num_return_sequences=1)[0]['generated_text']
    return evolved

# Test it: Simple base MNIST model
if __name__ == "__main__":
    base_mnist = "import torch.nn as nn; class SimpleNet(nn.Module): def __init__(self): super().__init__(); self.fc = nn.Linear(784, 10)"
    result = self_synth(base_mnist)
    print("\nFinal Evolved Output:\n", result)  # Your Day 1 win
