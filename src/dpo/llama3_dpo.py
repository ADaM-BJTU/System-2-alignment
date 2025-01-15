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
from trl import DPOTrainer,DPOConfig
from peft import PeftModel
import sys, os


from fire import Fire

def main(base_model_name = "sft-model", output_dir = ""):
  
    dataset_file = '../data/dpo_train_llama3.json'
    # load the training dataset
    dataset = load_dataset("json", data_files={'train': dataset_file})
    dataset = dataset['train'].shuffle(seed=42)
    device_map = "auto"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        trust_remote_code=True,
    )
    # sft_lora_weights = '/mnt/sdc1/yuhang/proj/culture_aware_alignment/finetune/trl/opinion/checkpoints/opinion-sft-self-instruct-principle-gpt35/final_checkpoint'
    # base_model = PeftModel.from_pretrained(
    #             base_model,
    #             sft_lora_weights,
    #             torch_dtype=torch.float16,
    #         )
    base_model.config.use_cache = False

    # when running with ref model whis error comes:
    # ValueError: You passed both a ref_model and a peft_config.
    # For training PEFT adapters with DPO there is no need to pass a reference model. Please pass `ref_model=None` in case you want to train PEFT adapters.
    # ref_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_name,
    #     quantization_config=bnb_config,
    #     device_map=device_map,
    #     trust_remote_code=True,
    # )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def get_prompt(example):
        prompt_sample = [
            {"role": "user", "content": example['prompt']}
        ]
        
        prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
        example['prompt'] = prompt_for_model

        example['chosen'] = example['chosen'] + tokenizer.eos_token
        example['rejected'] = example['rejected']+ tokenizer.eos_token

        # print(example)

        return example

    dataset = dataset.map(get_prompt)

    # from https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb
    lora_dropout=0.05
    lora_alpha=16
    lora_r=16
    learning_rate=5e-5

    batch_size = 1

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


    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        warmup_steps=10,
        logging_steps=10,
        num_train_epochs=1,
        save_steps=20,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
    )

    trainer = DPOTrainer(
        base_model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        beta=0.1,
        max_prompt_length=1024,
        max_length=2048,
    )
    trainer.train()
    # todo: during training getting these warning:

    # i guess this is on the base model, need to check. in that case this is fine
    # UserWarning: None of the inputs have requires_grad=True. Gradients will be None

    # seems that this can be ignored:
    # Could not estimate the number of tokens of the input, floating-point operations will not be computed
    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

Fire(main)