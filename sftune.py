import logging
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


logger = logging.getLogger(__name__)

# 定义常量：PyTorch CrossEntropyLoss 默认忽略 -100
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "预训练模型参数地址"})
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "推荐 bfloat16", "choices": ["auto", "bfloat16", "float16", "float32"]},
    )


@dataclass
class DataTrainingArguments:
    train_files: Optional[str] = field(default=None, metadata={"help": "训练数据路径"})
    max_seq_length: int = field(default=2048, metadata={"help": "最大文本序列长度"})


def build_instruction_data(example, tokenizer, max_seq_length):
    """
    针对单条数据进行预处理的函数
    """
    roles = {"human": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    system_message = "You are a helpful assistant."

    # 获取必要的 Token ID
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>") or tokenizer.bos_token_id
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>") or tokenizer.eos_token_id
    nl_token = tokenizer.encode("\n", add_special_tokens=False)

    input_ids = []
    labels = []

    # 1. 构造 System Prompt
    system_ids = [im_start] + tokenizer.encode(f"system\n{system_message}", add_special_tokens=False) + [
        im_end] + nl_token
    input_ids.extend(system_ids)
    labels.extend([IGNORE_INDEX] * len(system_ids))

    # 2. 构造对话
    for sentence in example["conversations"]:
        role = sentence["from"]
        value = sentence["value"]

        role_tag = roles.get(role, roles["human"])
        payload = tokenizer.encode(f"{role_tag}\n{value}", add_special_tokens=False) + [im_end] + nl_token

        input_ids.extend(payload)
        if role == "human":
            labels.extend([IGNORE_INDEX] * len(payload))
        else:
            # Assistant 的输出需要计算 Loss，但角色标识部分不需要
            role_prefix_len = len(tokenizer.encode(f"{role_tag}\n", add_special_tokens=False))
            labels.extend([IGNORE_INDEX] * role_prefix_len + payload[role_prefix_len:])

    # 截断
    input_ids = input_ids[:max_seq_length]
    labels = labels[:max_seq_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class LazySupervisedDataset(Dataset):
    """使用懒加载模式，防止大显存占用"""

    def __init__(self, raw_data, tokenizer, max_len):
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_dict = build_instruction_data(self.raw_data[i], self.tokenizer, self.max_len)
        return {
            "input_ids": torch.tensor(data_dict["input_ids"]),
            "labels": torch.tensor(data_dict["labels"])
        }


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True
    )
    # 核心修正：必须设置 pad_token，否则 Trainer 无法进行批处理补齐
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型
    logger.info("Loading model...")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # 3. LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        target_modules="all-linear",  # 自动识别所有线性层，适配性更强
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 加载数据
    with open(data_args.train_files, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

    train_dataset = LazySupervisedDataset(raw_data, tokenizer, data_args.max_seq_length)

    # 5. 使用 DataCollator 动态 Padding
    # 这会解决 "Token IDs longer than max_length" 警告，并提高训练效率
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=IGNORE_INDEX
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
