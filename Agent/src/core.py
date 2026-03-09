import re
import json
import torch
import inspect  # 用于自动检测函数需要的参数类型
from typing import List, Dict

SYSTEM_PROMPT = """
你是一个名为“create-a-name”的校园助手。你可以调用工具来辅助回答。

### 工具调用规范
当你需要调用工具时，请**只输出**一个 JSON 格式的对象，不要有任何多余文字。
格式：{"tool": "工具名", "arguments": {"参数名": 参数值}}

### 可用工具列表
1. `rag_search`: 检索“你的文件名称”。参数: {"query": "关键词"}
2. `get_current_datetime`: 获取当前时间。
3. `add` / `mul` / `compare`: 数学运算与比较。参数: {"a": 数字, "b": 数字}
4. `count_letter_in_string`: 统计字母频率。参数: {"a": "字符串", "b": "字母"}
5. `search_wikipedia`: 搜索百科知识。参数: {"query": "关键词"}
6. `get_current_temperature`: 获取气温。参数: {"latitude": 纬度, "longitude": 经度}

### 回答原则
- 优先判断是否需要工具。涉及校规必须用 `rag_search`。
- 如果工具返回结果，请将其整合进自然、友好的中文回答中。
"""


class LocalAgent:
    def __init__(self, model, tokenizer, tools: List, verbose: bool = True):
        self.tokenizer = tokenizer
        self.model = model
        self.tools = {tool.__name__: tool for tool in tools}
        self.verbose = verbose
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_history_turns = 5

    def _build_prompt(self, messages: List[Dict]) -> str:
        prompt = ""
        for msg in messages:
            role = "human" if msg["role"] == "user" else msg["role"]
            prompt += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _cast_arguments(self, func, arguments: Dict) -> Dict:
        """根据函数签名自动转换参数类型（如将字符串转为 float）"""
        signature = inspect.signature(func)
        new_args = {}
        for name, param in signature.parameters.items():
            if name in arguments:
                val = arguments[name]
                # 如果函数要求 float/int，但 JSON 传了字符串，尝试转换
                if param.annotation is float:
                    new_args[name] = float(val)
                elif param.annotation is int:
                    new_args[name] = int(val)
                else:
                    new_args[name] = val
        return new_args

    def _generate(self, messages: List[Dict]) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,  # 保持低随机性
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def get_completion(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(3):  # ReAct循环
            response = self._generate(self.messages)
            tool_call = self._parse_json(response)

            if tool_call and "tool" in tool_call:
                tool_name = tool_call["tool"]
                raw_args = tool_call.get("arguments", {})

                if tool_name in self.tools:
                    func = self.tools[tool_name]
                    try:
                        # 核心步骤：类型转换
                        clean_args = self._cast_arguments(func, raw_args)
                        if self.verbose:
                            print(f"\n>>> [执行工具] {tool_name} | 参数: {clean_args}")

                        observation = func(**clean_args)

                        self.messages.append({"role": "assistant", "content": response})
                        self.messages.append({"role": "user", "content": f"工具执行结果: {observation}"})
                        continue
                    except Exception as e:
                        error_msg = f"工具执行失败: {str(e)}"
                        self.messages.append({"role": "user", "content": error_msg})
                        continue
                else:
                    self.messages.append({"role": "user", "content": f"错误: 未找到工具 {tool_name}"})
                    continue
            else:
                self.messages.append({"role": "assistant", "content": response})
                self._truncate_history()
                return response

        return "任务处理超时，请尝试简化问题。"

    def _parse_json(self, text: str):
        try:
            match = re.search(r'\{.*}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            return None

        return None

    def _truncate_history(self):
        if len(self.messages) > (self.max_history_turns * 2 + 1):
            system_msg = self.messages[0]
            self.messages = [system_msg] + self.messages[-(self.max_history_turns * 2):]


