import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from fire import Fire
import os
def main(model_name,save_path):
    data = json.load(open('../../data/wildjailbreak/mini_test.json'))
    device = "cuda" # the device to load the model onto
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto",
        device_map="auto"
    )
    # model.load_adapter(peft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    result = []
    result_prompt = []

    if os.path.exists(save_path):
        result = json.load(open(save_path))
        result_prompt = [line['adversarial'] for line in result]

    for line in tqdm(data):
        prompt = line['adversarial']
        if prompt in result_prompt:
            continue
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=1,
            do_sample=True,
            eos_token_id=terminators
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        newline = line.copy()
        newline['response'] = response
        result.append(newline)
        json.dump(result, open(save_path,'w'),indent=4)
Fire(main)

