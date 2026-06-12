"""
统一目录配置
所有流水线共享相同的目录结构
"""
import os

# 基础目录（相对于 run_dataflow）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 统一目录结构
DATA_DIR = os.path.join(BASE_DIR, "data")          # 输入文件
OUTPUT_DIR = os.path.join(BASE_DIR, "output")      # 最终输出
CACHE_DIR = os.path.join(BASE_DIR, "cache")        # 处理缓存
MINERU_CACHE_DIR = os.path.join(BASE_DIR, ".cache", "mineru")  # MinerU中间文件

# 书名对应的子目录名
BOOK_CONFIG = {
    "中医望诊彩色图谱": {
        "input_name": "wangzhen",
        "cache_subdir": "wangzhen",
        "mineru_subdir": "wangzhen",
    },
    "望面诊病图解_赵理明": {
        "input_name": "zhaoliming",
        "cache_subdir": "zhaoliming",
        "mineru_subdir": "zhaoliming",
    },
    "中医望诊与舌诊彩色图解_刘文兰": {
        "input_name": "liuwenlan",
        "cache_subdir": "liuwenlan",
        "mineru_subdir": "liuwenlan",
    },
}


def get_book_paths(book_name: str):
    """获取指定书的所有路径"""
    config = BOOK_CONFIG.get(book_name)
    if not config:
        raise ValueError(f"未知的书名: {book_name}")

    return {
        "input_jsonl": os.path.join(DATA_DIR, f"input_{config['input_name']}.jsonl"),
        "cache_dir": os.path.join(CACHE_DIR, config["cache_subdir"]),
        "mineru_dir": os.path.join(MINUERU_CACHE_DIR, config["mineru_subdir"]),
        "output_dir": os.path.join(OUTPUT_DIR, book_name),
    }


def ensure_dirs():
    """确保所有基础目录存在"""
    for d in [DATA_DIR, OUTPUT_DIR, CACHE_DIR, MINERU_CACHE_DIR]:
        os.makedirs(d, exist_ok=True)
