import os
import PyPDF2
import markdown
import json
import tiktoken
from bs4 import BeautifulSoup
import re

# 获取指定编码器对象，能够将文本转换为token ID列表，以及将token ID列表还原为文本的功能
# cl100k_base适用于gpt-4等模型
enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    读取文档的类
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.file_list = self.get_files()

    def get_files(self):
        # 目标文件夹路径
        file_list = []
        # os.walk会递归访问所有目录下的子目录
        # 这里访问self.path 即传入的目录路径
        # 返回三元组，分别是 当前遍历目录路径，当前目录下所有子目录名称，当前目录下所有非目录文件名称
        for filepath, dirnames, filenames in os.walk(self.path):
            # os.walk函数将递归遍历到指定的文件夹
            for filename in filenames:
                if filename.endswith(".md"):
                    # 将md结尾文件的绝对路径加入到目标文件夹列表内
                    # 绝对目录由os.path.join拼接形成，filepath是当前目录路径，filename是当前文件
                    # 可拼接成此文件的绝对目录，再由append加入到列表中
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    @classmethod
    def read_pdf(cls, file_path: str):
        # 只读二进制方式打开可读取非文本文件
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # 对于pdf的每一页
            for page_num in range(len(reader.pages)):
                # reader.pages返回一个包含PDF所有页面的列表，其中每个元素是一个PageObject实例，代表PDF的一页
                # extract_text()是PageObject的一个方法，用于从PDF页面中提取文本内容，解析文本元素并返回一个字符串
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            # 将markdown格式的文本转换为html
            html_text = markdown.markdown(md_text)
            # 用BeautifulSoup从HTML中提取文本
            # 第一个参数为要解析的HTML字符串或文件对象
            # 第二个参数为指定解析器
            # 返回一个BeautifulSoup对象，代表整个解析树
            soup = BeautifulSoup(html_text, 'html.parser')
            # soup的方法，get_text()可提取文档中所有的文本内容，去除HTML标签，返回一个字符串，包含所有可见文本
            plain_text = soup.get_text()
            # 使用正则表达式移除网址连接
            text = re.sub(r'http\S+', '', plain_text)
            return text

    @classmethod
    def read_txt(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith(".pdf"):
            # 类方法允许在不依赖实例的情况下操作类级别数据或与类相关的功能
            return cls.read_pdf(file_path)
        elif file_path.endswith(".md"):
            return cls.read_markdown(file_path)
        elif file_path.endswith(".txt"):
            return cls.read_txt(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def get_chunk(cls, file_path: str,  max_token_len: int, cover_content: int):
        """
        context_prefix: 注入的上下文信息，例如 "[来源: 学生手册 - 宿舍管理规定]"
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        file_name = os.path.basename(file_path).replace(".txt", "")
        lines = content.splitlines()

        chunk_text = []
        current_main_title = "学校概况"
        current_sub_section = ""

        # 手册优化正则
        main_title_pattern = re.compile(r'.*(办法|规定|要求|简则|简介)$|.*（修订）$')
        sub_section_pattern = re.compile(r'^第[一二三四五六七八九十百]+条.*')

        curr_chunk = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 更新当前章节信息
            if main_title_pattern.match(line) and len(line) < 40:
                current_main_title = line
                continue
            if sub_section_pattern.match(line):
                current_sub_section = line[:15]

            # 构造动态前缀
            context_prefix = f"[{file_name} | {current_main_title} | {current_sub_section}] "

            # 计算可用空间：最大长度 - 前缀长度 - 重叠长度
            prefix_len = len(enc.encode(context_prefix))
            effective_len = max_token_len - prefix_len - cover_content

            # 试探性合并
            test_chunk = curr_chunk + "\n" + line if curr_chunk else line

            if len(enc.encode(test_chunk)) <= effective_len:
                curr_chunk = test_chunk
            else:
                if curr_chunk:
                    chunk_text.append(context_prefix + curr_chunk)

                # 开启新块并处理 Overlap
                prev_body = curr_chunk
                cover_part = prev_body[-cover_content:] if len(prev_body) > cover_content else prev_body
                curr_chunk = cover_part + "\n" + line

        if curr_chunk:
            chunk_text.append(f"[{file_name} | {current_main_title} | {current_sub_section}] " + curr_chunk)

        return chunk_text

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        for file_path in self.file_list:
            # 每一个文件调用一次内部处理逻辑
            content = self.get_chunk(file_path, max_token_len, cover_content)
            docs.extend(content)
        return docs


class Documents:
    """
    获取已分好类的json格式文档
    """

    def __init__(self, path: str = ''):
        self.path = path

    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            # 读取json数据
            content = json.load(f)
            return content
