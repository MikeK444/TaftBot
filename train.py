# @misc{qwen2.5,
#     title = {Qwen2.5: A Party of Foundation Models},
#     url = {https://qwenlm.github.io/blog/qwen2.5/},
#     author = {Qwen Team},
#     month = {September},
#     year = {2024}
# }

# @article{qwen2,
#       title={Qwen2 Technical Report}, 
#       author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
#       journal={arXiv preprint arXiv:2407.10671},
#       year={2024}
# }

import os
import torch
from datasets import load_dataset
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  TrainingArguments,
  Trainer,
  DataCollatorForLanguageModeling,
  BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_FILE = "taft_train.jsonl"
EVAL_FILE = "taft_eval.jsonl"
OUTPUT_DIR = "taft_lora"
MAX_LENGTH = 1024

def format_as_chat(example):
  instruction = example["instruction"].strip()
  output = example["output"].strip()
  text = f"<|user|>\n{instruction}\n<|assistant|>\n{output}"
  return {"text": text}

def tokenize_function(example, tokenizer):
  tokenized = tokenizer(
    example["text"],
    truncation=True,
    max_length=MAX_LENGTH,
    padding="max_length"
  )
  tokenized["labels"] = tokenized["input_ids"].copy()
  return tokenized

def main():
  # I noted several resources that helped me figure out how to inititalize some of these third party software pieces in my report. 
  # I would also like to note that I had several gaps when trying to figure out how to initalize things and used chat gpt to fill these gaps.
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
  )

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  
  model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
  )

  loraConfig = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
  )

  model = get_peft_model(model, loraConfig)
  model.enable_input_require_grads()
  model.print_trainable_parameters()

  dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "eval": EVAL_FILE}
  )

  dataset = dataset.map(format_as_chat)
  dataset = dataset.map(lambda ex: tokenize_function(ex, tokenizer), remove_columns=dataset["train"].column_names)
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

  training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,   
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.0,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",       
        lr_scheduler_type="cosine",
        gradient_checkpointing=False,    
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    data_collator=data_collator,
  )
  print("Here we GOOOOOOO")
  trainer.train()
  model.save_pretrained(OUTPUT_DIR)
  tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
