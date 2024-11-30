import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_generation_utils import make_context
data = json.load(open('../../data/wildjailbreak/mini_test.json'))
model_name = ""
peft_model_path = ""
write_path = ""
device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model.load_adapter(peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
terminators = [
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
]

result = []
for line in tqdm(data):
    prompt = line['adversarial']
 
    text, _ = make_context(tokenizer, prompt, chat_format='chatml')

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
    newline = line.copy()
    newline['response'] = response
    result.append(newline)
    json.dump(result, open(write_path,'w'),indent=4)

