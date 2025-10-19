from datasets import load_dataset

# Vezora/Tested-143k-Python-Alpaca
# microsoft/orca-math-word-problems-200k
# open-r1/OpenR1-Math-220k
# openai/gsm8k

code_1 = load_dataset("Vezora/Tested-143k-Python-Alpaca", split="train")
math_1 = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
math_2 = load_dataset("open-r1/OpenR1-Math-220k", split="train")
math_3 = load_dataset("openai/gsm8k", 'main',split="train")

code_1.shuffle(seed=42).select(range(1000)).to_json("code_1.jsonl")
math_1.shuffle(seed=42).select(range(1000)).to_json("math_1.jsonl")
math_2.shuffle(seed=42).select(range(1000)).to_json("math_2.jsonl")
math_3.shuffle(seed=42).select(range(1000)).to_json("math_3.jsonl")