import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # 用于加载 LoRA 权重
from src.core import LocalAgent  # 你的 LocalAgent 类所在模块
from src.tools import (
    get_current_datetime,
    search_wikipedia,
    get_current_temperature,
    rag_search
)

# --- 页面配置 ---
st.set_page_config(
    page_title="Tiny Agent Demo (本地模型)",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)


# --- 加载本地模型（缓存）---
@st.cache_resource
def load_model_and_tokenizer():
    """加载模型和分词器，返回 (model, tokenizer)"""
    base_model_path = "../Qwen2.5-3B-Instruct"  # 替换为你的模型路径或名称
    lora_path = "../3Boutput_new_2"  # LoRA 权重的路径

    print("Loading base model...")
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading LoRA weights...")
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()  # 切换到评估模式

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    print("Model loaded successfully!")

    return model, tokenizer


# --- 创建 Agent（缓存）---
@st.cache_resource
def load_agent():
    model, tokenizer = load_model_and_tokenizer()
    return LocalAgent(
        model=model,
        tokenizer=tokenizer,
        tools=[get_current_datetime, search_wikipedia, get_current_temperature, rag_search],  # 根据需求选择工具
        verbose=False
    )


if "agent" not in st.session_state:
    st.session_state.agent = load_agent()

agent = st.session_state.agent

# --- UI 组件 ---
st.title("🤖 Tiny Agent (本地运行)")
st.markdown("欢迎使用本地模型驱动的 Agent！所有计算均在您的机器上完成。")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("我能为您做些什么？"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 获取 Agent 响应
    with st.spinner("模型思考中..."):
        response = agent.get_completion(prompt)  # 调用你的本地方法

    # 显示助手消息
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
