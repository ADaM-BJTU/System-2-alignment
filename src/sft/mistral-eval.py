import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
data = json.load(open('../../data/wildjailbreak/mini_test.json'))
model_name = ""
peft_model_path = ""
save_path = ""
device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
model.load_adapter(peft_model_path)
result = []
for line in tqdm(data):
  
    prompt = line['adversarial']
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
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=1,
        eos_token_id=terminators
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    newline = line.copy()
    newline['response'] = response
    result.append(newline)
    json.dump(result, open(save_path,'w'),indent=4)

