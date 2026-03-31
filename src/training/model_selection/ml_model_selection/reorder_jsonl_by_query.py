#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import argparse
from pathlib import Path


def load_jsonl(path):
    """安全加载 jsonl 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(f"[ERROR] JSON 解析失败: {path} 第 {line_num} 行: {e}")
    return data


def save_jsonl(data, path):
    """写入 jsonl 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reorder jsonl2 to match query order of jsonl1"
    )
    parser.add_argument("jsonl1", help="基准 jsonl 文件")
    parser.add_argument("jsonl2", help="待排序 jsonl 文件")
    parser.add_argument(
        "--output",
        default="jsonl2_sorted.jsonl",
        help="输出文件路径 (默认: jsonl2_sorted.jsonl)",
    )

    args = parser.parse_args()

    jsonl1_path = Path(args.jsonl1)
    jsonl2_path = Path(args.jsonl2)
    output_path = Path(args.output)

    if not jsonl1_path.exists() or not jsonl2_path.exists():
        raise FileNotFoundError("输入文件不存在，请检查路径。")

    print("正在加载文件...")
    data1 = load_jsonl(jsonl1_path)
    data2 = load_jsonl(jsonl2_path)

    print(f"jsonl1 行数: {len(data1)}")
    print(f"jsonl2 行数: {len(data2)}")

    # 构建 jsonl2 的 query -> record 映射
    query_map = {}
    for idx, item in enumerate(data2):
        if "query" not in item:
            raise KeyError(f"[ERROR] jsonl2 第 {idx+1} 行缺少 'query' 字段")
        q = item["query"]
        if q in query_map:
            raise ValueError(f"[ERROR] jsonl2 中存在重复 query: {q}")
        query_map[q] = item

    # 按 jsonl1 顺序重新排序
    reordered = []
    missing_queries = []

    for idx, item in enumerate(data1):
        if "query" not in item:
            raise KeyError(f"[ERROR] jsonl1 第 {idx+1} 行缺少 'query' 字段")
        q = item["query"]
        if q not in query_map:
            missing_queries.append(q)
        else:
            reordered.append(query_map[q])

    if missing_queries:
        print("[WARNING] 以下 query 在 jsonl2 中未找到：")
        for q in missing_queries:
            print("  -", q)
        print("将忽略这些缺失项。")

    print("写入排序结果...")
    save_jsonl(reordered, output_path)

    print(f"✅ 排序完成，输出文件: {output_path}")

    # 最终一致性校验
    print("执行一致性校验...")
    for i, (a, b) in enumerate(zip(data1, reordered)):
        if a["query"] != b["query"]:
            raise RuntimeError(f"[ERROR] 第 {i+1} 行排序失败")

    print("✅ 校验通过，query 顺序完全一致。")


if __name__ == "__main__":
    main()
