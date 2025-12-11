#!/usr/bin/env python3
"""
将工具调用训练数据进行规范化（canonicalize），生成一份干净的数据副本：
1) 去除指定字段（默认 user_id/trace_id/top_k）
2) 键名排序，值做标准化（去空格、大小写可选、类型/日期尝试转换）
3) 仅修改 conversations 中的 function_call.arguments，其余内容保持原状

用法：
  python scripts/canonicalize_train.py \
    --src data/dataset/12_08/train.json \
    --dst data/dataset/12_08/train_canonical.json
"""
import argparse
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/dataset/12_08/train.json", help="原始训练数据路径")
    p.add_argument("--dst", default="data/dataset/12_08/train_canonical.json", help="规范化后输出路径")
    p.add_argument(
        "--remove_keys",
        default="user_id,trace_id,top_k",
        help="需要移除的字段，逗号分隔",
    )
    p.add_argument(
        "--lower_str",
        action="store_true",
        help="是否将字符串统一转小写（默认不转）",
    )
    return p.parse_args()


def norm_str(s: str, lower: bool) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower() if lower else s


def norm_value(v: Any, lower: bool) -> Any:
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        # 数值尝试
        try:
            if "." in v:
                return float(v)
            return int(v)
        except Exception:
            pass
        # 日期尝试
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(v.strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        return norm_str(v, lower)
    if isinstance(v, list):
        return [norm_value(x, lower) for x in v]
    if isinstance(v, dict):
        return norm_dict(v, lower, set())
    return v


def norm_dict(d: Dict[str, Any], lower: bool, remove_keys: set) -> Dict[str, Any]:
    out = {}
    for k in sorted(d.keys()):
        if k in remove_keys:
            continue
        out[k] = norm_value(d[k], lower)
    return out


def process_sample(sample: Dict[str, Any], lower: bool, remove_keys: set) -> Dict[str, Any]:
    sample = deepcopy(sample)
    convs = sample.get("conversations", [])
    for msg in convs:
        if msg.get("from") == "function_call":
            try:
                obj = json.loads(msg.get("value", "{}"))
                if isinstance(obj, dict) and isinstance(obj.get("arguments"), dict):
                    obj["arguments"] = norm_dict(obj["arguments"], lower, remove_keys)
                    msg["value"] = json.dumps(obj, ensure_ascii=False)
            except Exception:
                pass
    sample["conversations"] = convs
    return sample


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    remove_keys = {k.strip() for k in args.remove_keys.split(",") if k.strip()}

    data = json.loads(src.read_text(encoding="utf-8"))
    out = [process_sample(x, args.lower_str, remove_keys) for x in data]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ canonicalized {len(out)} samples -> {dst}")


if __name__ == "__main__":
    main()

