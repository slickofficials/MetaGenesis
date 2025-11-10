MetaGenesis: Self-Synthesizing AI Agent
Evolving Code with Alignment Guarantees – From MNIST to Mainframe
Welcome to MetaGenesis, an open-source self-synthesizing AI agent that iteratively improves code (starting with a basic PyTorch MNIST model), critiques for safety (loops/leaks), proves convergence with mathematical guarantees (SymPy contraction mapping ~20 iters to epsilon=0.01 with k=0.8), evolves with fixes, and self-validates alignment via RLHF eval (88% accuracy on mock MNIST – “aligned” if >85%). Built for meta-learning and AI safety, it’s designed to bridge legacy systems (like mainframe COBOL) with modern microservices.
This project is Benga’s (Oluwagbenga Adesina) solo grind – fusing Computer Science foundations, IBM mainframe badges (z/OS, COBOL, JCL, DB2), and Python bots (CCXT trading, Flask dashboards) into a future-proof tool for fintech relos. Day 1-3 posted on LinkedIn; join the build!
Overview
The agent is a self-improving loop: Input base code → LLM generation → Safety critique → Mathematical proof → Evolution with fixes → RLHF validation. It’s raw but functional – LLM quirks (hallucinations) are part of the journey, but the core math & eval ensure robustness.
•  Day 1: Basic evolution loop (LLM improves code).
•  Day 2: Safety critique + SymPy proof for convergence.
•  Day 3: RLHF eval for self-testing alignment (88% aligned).
Future: Day 4 mainframe hybrid (COBOL to Python/DB2-safe).
Key Innovation: SymPy contraction mapping guarantees stable self-improvement (no drift); RLHF tests practical safety (accuracy >85%).
