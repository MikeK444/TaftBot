import json
import random

INPUT_FILE = "taftInstruct.jsonl"
TRAIN_FILE = "taft_train.jsonl"
EVAL_FILE = "taft_eval.jsonl"

EVAL_SIZE = 50  

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.seed(42)
random.shuffle(lines)

eval_lines = lines[:EVAL_SIZE]
train_lines = lines[EVAL_SIZE:]

with open(EVAL_FILE, "w", encoding="utf-8") as f:
    for line in eval_lines:
        f.write(line)

with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for line in train_lines:
        f.write(line)

print(f"Eval examples: {len(eval_lines)}")
print(f"Train examples: {len(train_lines)}")
