import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
 
def compare_model_weights(model1, model2):
    """
    Compare the weights of two models and return True as soon as any layer's weights are different (early exit).
    Return False if all weights are the same.
    """
    for name1, param1 in model1.named_parameters():
        if name1 in model2.state_dict():
            param2 = model2.state_dict()[name1]
            # Early exit if any weights are different
            if not torch.allclose(param1, param2):
                print(f"Layer '{name1}': Weights are DIFFERENT.")
                return True
        else:
            print(f"Layer '{name1}' not found in the second model.")
            return True
    
    # Return False if no differences were found
    return False
 
 
# Define the paths to your base model and LoRA directories
# base_model_dir = '/home/yuhang/proj/llms/culture/Mistral-7B-Instruct-v0.1/'
# lora_model_dir = './mistral_all_sft/final_checkpoint/'
# merged_model_dir ='./mistral_all_sft/merged/'

base_model_dir = ''
lora_model_dir = ''
merged_model_dir =''


# Step 1: Load the base model and tokenizer
# !!!! check torch_dtype in the config.json is same as below
# !!!! otherwise the size will change
print("Loading base model and tokenizer...")
model_base = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
    )
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
 
# Optional: check model params before and after merging
import copy
model_base_original = copy.deepcopy(model_base)
 
# Step 2: Load the LoRA configuration
print("Loading LoRA configuration...")
peft_config = PeftConfig.from_pretrained(lora_model_dir)
 
# Step 3: Load the LoRA weights into the base model
print("Loading LoRA model and applying weights...")
model_lora = PeftModel.from_pretrained(
    model_base, 
    lora_model_dir,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
    )
 
# Step 4: Merge the LoRA weights with the base model and unload LoRA
print("Merging LoRA weights into base model...")
model_merged = model_lora.merge_and_unload()
# Now `merged_model` contains the base model with LoRA weights merged
 
# Optional: check model params before and after merging
isdifferent = compare_model_weights(model_base_original, model_merged)
if isdifferent:
    print("Merging is valid.")
else:
    print("Merging changes no params. Merging may be invalid.")
 
# Save the merged model
print(f"Saving merged model to {merged_model_dir}...")
model_merged.save_pretrained(merged_model_dir, max_shard_size="4GB")
tokenizer.save_pretrained(merged_model_dir)
 
print("Model merging complete.")