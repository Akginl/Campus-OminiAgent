import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.core import LocalAgent
from src.tools import (
    get_current_datetime,
    search_wikipedia,
    get_current_temperature,
    rag_search
)

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

# --- 页面配置与样式 ---
st.set_page_config(page_title="太理通 Agent", page_icon="🤖", layout="wide")
st.title("🤖 太理通：高校规章制度智能专家")
st.caption("基于 Qwen2.5-3B + LoRA 微调 | 支持 ReAct 角色推理与混合检索")

# --- 统一工具列表 ---
# 3B 模型对工具出现的先后顺序极其敏感
SHARED_TOOLS = [
    get_current_datetime,
    search_wikipedia,
    rag_search,
    get_current_temperature
]


# --- 模型加载逻辑 ---
# --- 只缓存模型和分词器 ---
@st.cache_resource
def load_model_and_tokenizer():
    base_model_path = "../Qwen2.5-3B-Instruct"
    lora_path = "../3Boutput_new_2"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # 建议让 transformers 自动分配
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()

# --- 初始化 Session State ---
# 用于 UI 显示的历史记录
if "display_history" not in st.session_state:
    st.session_state.display_history = []

# 用于 Agent 推理的真实状态
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# --- 渲染历史对话 ---
# 这一步是关键！脚本每次重跑，都会先把之前的聊天记录画出来
for message in st.session_state.display_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 处理当前用户输入 ---
if prompt := st.chat_input("请输入您关于太原理工大学的问题..."):
    # A. 展示用户输入并存入 UI 历史
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.display_history.append({"role": "user", "content": prompt})

    # B. 准备 Agent 实例
    # 每次都创建新实例，但同步之前的 agent_messages
    agent = LocalAgent(
        model=model,
        tokenizer=tokenizer,
        tools=SHARED_TOOLS,
        verbose=True
    )
    agent.messages = st.session_state.agent_messages

    # C. 生成并展示助手回答
    with st.chat_message("assistant"):
        with st.spinner("正在思考并检索手册..."):
            response = agent.get_completion(prompt)
            st.markdown(response)

    # D. 更新状态：同步 UI 历史和 Agent 真实状态
    st.session_state.display_history.append({"role": "assistant", "content": response})
    st.session_state.agent_messages = agent.messages
