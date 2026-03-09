from typing import List
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

RAG_PROMPT_TEMPLATE = """
使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
有用的回答:
"""


class BaseModel:
    def __init__(self, model) -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class LocalChat:

    def __init__(self, model, tokenizer):
        # 本地推理除了需要加载实际的模型，还需要对应的分词器对象
        self.model = model
        self.tokenizer = tokenizer

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        # 构建输入文本
        text = ""

        # system 消息
        text += "<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n"

        # 历史对话（假设 history 中每条消息的 role 是 "human" 或 "assistant"）
        for msg in history:
            role = msg["role"]  # 必须是 "human" 或 "assistant"
            text += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"

        # 当前用户问题
        user_content = RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)
        text += f"<|im_start|>human\n{user_content}<|im_end|>\n"

        # 添加助手起始标记，让模型开始生成
        text += "<|im_start|>assistant\n"

        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)

        # 通常返回一个张量(batch_size, seq_len)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # 使用贪婪搜索
            repetition_penalty=1.1,  # 轻微惩罚重复
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # 从生成的token序列中提取新生成的部分
        # 首先由decode将生成的tokenID序列转换为原始文本字符串
        # outputs[0]即去除批次中第一个样本的tokenId序列
        # input_ids : (batch_size, seq_len)
        # input_ids.shape[1]即seq_len， 因为需要知道输入部分有多少个token，以便从新生成的总序列中切掉输入部分
        # 利用切片操作，即[inputs.input_ids.shape[1]:]从新生成部分的开始到最后
        # 跳过特殊字符后，在解码的时候特殊的token不会出现在输出的字符串中
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return response
