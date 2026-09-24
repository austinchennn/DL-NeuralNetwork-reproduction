"""每个网络目录都是独立脚本集合，模块名互相重复（model / config / utils ...）。
这里临时把目标目录放到 sys.path 最前面导入，并清掉上一个目录留下的同名模块缓存。"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODULES = ("config", "dataset", "engine", "model", "utils", "tokenizer", "layers",
                 "train", "inference", "diffusion")


def load(folder: str, *names: str):
    path = str(ROOT / folder)
    for name in LOCAL_MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, path)
    try:
        modules = [importlib.import_module(n) for n in names]
    finally:
        sys.path.remove(path)
    return modules[0] if len(modules) == 1 else modules
