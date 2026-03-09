import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
# 导入本地 Agent 和工具
from src.core import LocalAgent
from src.tools import (
    get_current_datetime,
    search_wikipedia,
    get_current_temperature,
    rag_search,
)
# ================== 加载本地模型 ==================
base_model_path = "../Qwen2.5-3B-Instruct"
lora_path = "../3Boutput_new_2"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("Loading LoRA weights...")
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
print("Model loaded successfully!")


agent = LocalAgent(
    model=model,
    tokenizer=tokenizer,
    tools=[
            get_current_datetime,
            search_wikipedia,
            rag_search,
            get_current_temperature,
    ],
    verbose=True
)

# ================== 对话循环 ==================
print("开始对话（输入 exit 退出）")

while True:
    prompt = input("\033[94mUser: \033[0m")  # 蓝色
    if prompt.lower() == "exit":
        break
    response = agent.get_completion(prompt)
    print("\033[92mAssistant: \033[0m", response)
