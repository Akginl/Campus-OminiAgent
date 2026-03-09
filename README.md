# Campus-OminiAgent
Campus-OminiAgent是一个针对高校《学生手册》等复杂条文类文档设计，集成动态分块、混合检索与 Agent的行动推理的语言模型

## 项目简介
本系统旨在解决传统 RAG 在处理长篇幅、高密度、严谨性强的规章制度文档时存在的语义断裂与指令偏移问题。
通过在 8GB 显存单卡环境下对 Qwen2.5-3B 进行 LoRA 指令微调，结合 React，系统能够精准解析《太原理工大学学生手册》。
对于未能成功调用 RAG 的情况也能根据自身的微调回答。

## 核心特征

### 1. 标题感知型动态分块
针对规章制度文档层级分明的特点，自主设计并实现了 HACE 算法。
语义锚定：利用正则表达式动态捕获章节标题（如“第三章”、“第十二条”），并将其作为元数据前缀注入每一个文本分块。
消除孤岛：有效解决了传统 RAG 在切分长文档时导致的“语义丢失”问题，确保每个分块都携带完整的政策背景信息。

### 2. 双路协同混合检索架构 (Hybrid Retrieval with RRF)
为了平衡语义理解与专有名词的精确匹配，构建了 HybridRetriever 模块：

稠密检索：基于 BGE-M3 向量模型，捕捉深层语义关联。

稀疏检索：利用 BM25Okapi 算法，对特定术语进行硬匹配。

RRF 排名融合：采用倒数排名融合算法，将两路结果动态归一化。实测在《学生手册》测试集的五十个测试案例上，Hit Rate@3 从 55.6% 提升至稳定 80.0%以上。

### 3. 稳健型 ReAct Agent 任务编排
在 Qwen2.5-3B 微调模型上实现了推理与行动的闭环：

类型自适配器：针对小模型容易出现的参数类型误输出问题，利用 Python inspect 库开发了自动类型转换器，确保模型生成的字符串能精准映射为工具函数所需的 int 或 float，提升系统运行的稳定性。

## 项目架构
```
Campus-OminiAgent/
├── Agent/                  # Agent 核心逻辑与演示
│   ├── src/                # Agent 引擎
│   │   ├── core.py         # Agent 核心：实现 ReAct 循环、工具调用分发与类型对齐逻辑
│   │   └── tools.py        # 工具注册中心
│   ├── RAG/                # RAG 检索增强模块
│   │   ├── Embeddings.py   # BGE-M3 向量化实现
│   │   ├── VectorBase.py   # 向量数据库操作与 RRF 混合检索
│   │   ├── LLM.py          # 模型适配层：封装本地 Qwen 推理，处理 ChatML 模板与 RAG 上下文注入
│   │   └── utils.py        # 文本切分与清洗工具
│   ├── storage/            # 向量库存储文件
│   ├── RAG_demo.py             # 终端交互 Demo
│   ├── demo.py             # 终端交互 Demo
│   ├── web_demo.py         # Gradio/Streamlit Web 界面
│   └── eval_rag.py         # 评估引擎：自动化生成测试用例，量化对比 Vector 与 Hybrid 检索效能
├── data/                   # 数据集，存放切块文本数据和sft微调训练数据
│   ├── ...
├── 3Boutput/               # LoRA微调权重文件
│   ├── ...
├── Qwen2.5-3B-Instruct/    # 底座模型
│   ├── ...
├── sftune.py               # LoRA 微调训练脚本
├── requirements.txt        # 项目依赖
├── .env                    # 环境配置文件
└── README.md
```

