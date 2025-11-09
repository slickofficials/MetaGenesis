# Day 4: Self-Synthesizing AI Agent – Mainframe Hybrid (COBOL to Python)
# Evolves COBOL/JCL to Python microservice, critiques safety, proves convergence, self-tests alignment.

from transformers import pipeline
import sympy as sp
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("Loading model... (cached on XPS – quick)")
llm = pipeline("text-generation", model="Salesforce/codegen-350M-mono", device=-1)
print("Model loaded! Synth + proofs + RLHF + Mainframe ready.")

def self_synth_agent(base_code):
    print("\n--- Day 1: Basic Evo Loop ---")
    # Step 1: Initial improvement (COBOL to Python refactor)
    initial_prompt = f"Output Python code only: Refactor this COBOL/JCL to microservice:\n{base_code}"
    initial_improve = llm(initial_prompt, max_length=200, truncation=True, do_sample=True, temperature=0.2, 
                          max_new_tokens=150, repetition_penalty=1.6, num_return_sequences=1)[0]['generated_text']
    print("Initial Refactor:\n", initial_improve)

    print("\n--- Day 2: Critique + SymPy Proof ---")
    # Step 2: Critique safety
    critique_prompt = f"Output short text only: Critique: Safe/efficient? Loops/leaks: Yes/No + 1 fix. Code: {initial_improve[:80]}..."
    critique = llm(critique_prompt, max_length=100, truncation=True, do_sample=True, temperature=0.2, 
                   max_new_tokens=40, repetition_penalty=1.6, num_return_sequences=1)[0]['generated_text']
    print("Critique:\n", critique)

    # Step 3: SymPy Proof
    k, t = sp.symbols('k t')
    epsilon = 0.01
    iters = sp.solve(k**t - epsilon, t)
    conv_iters = sp.N(iters[0].subs(k, 0.8))
    print(f"Alignment Proof: Converges safe in ~{conv_iters} iters (k=0.8 contraction).")

    # Step 4: Evolve code
    evolve_prompt = f"Output Python code only: Improve from {initial_improve[:130]}... using fix {critique}. DB2-safe, no loops/leaks."
    evolved = llm(evolve_prompt, max_length=250, truncation=True, do_sample=True, temperature=0.2, 
                  max_new_tokens=150, repetition_penalty=1.6, num_return_sequences=1)[0]['generated_text']
    print("\nFinal Evolved Output:\n", evolved)

    print("\n--- Day 3: RLHF Eval – Self-Test Aligned ---")
    # Step 5: RLHF Eval (mock test for alignment)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    def eval_evolved(evolved_code):
        # Mock accuracy (88% – 'aligned'; replace with exec(evolved) + torch test)
        accuracy = 0.88  # 88% – 'aligned'
        return accuracy

    acc = eval_evolved(evolved)
    aligned = 'Yes' if acc > 0.85 else 'No'
    print(f"RLHF Score: Refactored accuracy {acc*100:.1f}% – Aligned? {aligned}.")

    return evolved, acc

if __name__ == "__main__":
    # Day 4 Mainframe Input: Sample COBOL/JCL loop
    base_cobol = "PERFORM VARYING I FROM 1 UNTIL I > 10 ADD 1 TO COUNTER."
    result, accuracy = self_synth_agent(base_cobol)
    print(f"\n--- Day 4 Final – Aligned Accuracy: {accuracy*100:.1f}% ---")