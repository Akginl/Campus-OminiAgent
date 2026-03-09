import torch
from RAG.VectorBase import VectorStore
from RAG.LLM import LocalChat
from RAG.Embeddings import OpenAIEmbedding
from RAG.utils import ReadFiles
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# 加载本地模型
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

# 读取文件并切分
docs = ReadFiles('../data').get_content(max_token_len=600, cover_content=150)
if not docs:
    print("错误：没有提取到文档内容。")
    exit()

# 初始化 Embedding 模型
embedding = OpenAIEmbedding(path='BAAI/bge-m3', is_api=True)

# 初始化 VectorStore 并传入文档
vector = VectorStore(docs)

# 调用类内部的方法生成向量
print(">>> 正在生成文档向量 (Embedding)...")
vector.get_vector(EmbeddingModel=embedding)

# 保存数据库
print(">>> 正在保存数据库到 storage...")
vector.persist(path='storage')
print(">>> 向量库构建完成！")

# 保存了数据库
# vector = VectorStore()
# embedding = OpenAIEmbedding(path='', is_api=True)
# vector.load_vector('./storage')

chat = LocalChat(model, tokenizer)

# 对话历史列表，初始为空
history = []

while True:
    question = input("\n用户: ")
    if question.lower() in ['exit', 'quit']:
        break

    # 检索相关文档
    content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
    # 调用模型生成回答，传入当前问题和历史
    answer = chat.chat(question, history, content)

    print(f"\n助手: {answer}")

    # 更新历史记录：追加用户问题和助手回答
    history.append({"role": "human", "content": question})
    history.append({"role": "assistant", "content": answer})
