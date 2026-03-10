import re
import json
import torch
import inspect  # 用于自动检测函数需要的参数类型
from typing import List, Dict

# 假设之前的工具定义和导入已经在你的环境中
# ... (此处省略你提供的 tools 代码) ...

SYSTEM_PROMPT = """
你是一个名为“太理通”的校园助手。请根据工具返回的事实回答问题。

### 1. 工具使用 (必须只输出 JSON)
- 涉及校规、地点、政策：调用 `rag_search`。
- 涉及当前时间：调用 `get_current_datetime`。
格式：{"tool": "工具名", "arguments": {"参数名": "值"}}

### 2. 回答准则
- **有据可查**：只根据 `rag_search` 返回的内容回答。手册没写的直接说不知道。
- **直击要点**：先说结论（是/否/时间），再说具体规定。
- **拒绝幻觉**：严禁编造手册中没有的细节（如夜宵、具体赔偿金额等）。

### 3. 可用工具
`rag_search`(query), `get_current_datetime`(), `search_wikipedia`(query)
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

        CAMPUS_KEYWORDS = ["学分", "毕业", "挂科", "学生手册", "奖学金", "宿舍", "校规", "不及格", "太理", "助学金"]      ## 根据你的学生手册修改

        for _ in range(3):  # ReAct 循环
            response = self._generate(self.messages)
            tool_call = self._parse_json(response)

            if tool_call and "tool" in tool_call:
                tool_name = tool_call["tool"]
                raw_args = tool_call.get("arguments", {})
                query_content = str(raw_args.get("query", ""))

                # --- 硬核拦截逻辑 ---   
                if tool_name == "search_wikipedia":  # 由于3B模型和训练数据的限制，模型可能会将校园问题使用维基百科查询，设置拦截
                    # 检查查询词是否包含校园敏感词
                    if any(kw in query_content for kw in CAMPUS_KEYWORDS):
                        observation = "【系统拦截】检测到您正试图通过维基百科查询校园内部事务。维基百科数据不准确，请必须改用 query_student_handbook 进行查询。"
                        self.messages.append({"role": "assistant", "content": response})
                        self.messages.append({"role": "user", "content": observation})
                        continue

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
