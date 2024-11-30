import os
os.environ["WANDB_DISABLED"]="true"
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import DPOTrainer,DPOConfig,SFTTrainer
import sys, os
import re
import random
from multiprocessing import cpu_count
base_model_name = ""
dataset_file = '../../data/wildjailbreak/mini_train.json'
output_dir = '' 



# load the training dataset
dataset = load_dataset("json", data_files={'train': dataset_file})
dataset = dataset['train'].shuffle(seed=42)

device_map = "auto"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=device_map,
    trust_remote_code=True,
)
base_model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if tokenizer.model_max_length > 100_000:
  tokenizer.model_max_length = 2048

def apply_chat_template(example, tokenizer):
    messages = []
    if 'vanilla' in example['data_type']:
        instruction = example['vanilla']
    else:
        instruction = example['adversarial']
    response = example["completion"]
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": response})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    # print(new_example["text"])
   
    return example

dataset = dataset.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                # remove_columns=column_names,
                                desc="Applying chat template",)


lora_dropout=0.05
lora_alpha=16
lora_r=16
learning_rate=5e-5

batch_size = 4

def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        lora_dropout=lora_dropout,
        lora_alpha=lora_alpha,
        r=lora_r,
        bias="none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )

    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config

model, lora_config = create_peft_config(base_model)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    warmup_steps=10,
    logging_steps=1,
    num_train_epochs=1,
    save_steps=10,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    # report_to="None"
)

trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        # packing=True,
        peft_config=lora_config,
        max_seq_length=tokenizer.model_max_length,
)

trainer.train()
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

