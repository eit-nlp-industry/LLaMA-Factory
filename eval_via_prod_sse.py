#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于生产端 SSE 接口的对齐评估脚本

功能：
- 读取标注数据（包含 conversations 与 pair2 目标工具）
- 直接请求生产接口(默认为 http://125.122.38.32:8085/mcp_end2end/stream)获取检索Top5与最终工具调用
- 计算 recall@5 与 precision@1，并输出报告

使用示例：
python eval_via_prod_sse.py \
  --input_file /home/qiyang_shi/testing/LLaMA-Factory/data/dataset/12_08/train_enhanced.json \
  --output_file /home/qiyang_shi/testing/LLaMA-Factory/data/dataset/result/data_evaluation_prod.json \
  --start_idx 0 --end_idx 50
"""

import asyncio
import aiohttp
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


PROD_SSE_URL = os.getenv("PROD_SSE_URL", "http://125.122.38.32:8085/mcp_end2end/stream")
#RETRIEVAL_ENDPOINT = os.getenv("RETRIEVAL_ENDPOINT", "http://125.122.38.32:6227/v1/mcp/tools/call")
RETRIEVAL_ENDPOINT = os.getenv("RETRIEVAL_ENDPOINT", "http://125.122.38.32:9527/v1/databoard/tools/call")

def _extract_user_query_from_conversation(item: Dict[str, Any]) -> str:
    """从标注数据的一条对话中提取原始用户问题（第一条 human）。"""
    conversations = item.get("conversations", [])
    for msg in conversations:
        if msg.get("from") == "human":
            return str(msg.get("value", ""))
    return ""


def _extract_pair2_target_tool(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """从标注数据中提取 pair2 的目标工具名和参数（查找 human->observation 后、下一条为 function_call 的目标）。"""
    conversations = item.get("conversations", [])
    # 逻辑：遇到 observation 后紧邻的下一条若是 function_call，则视为 pair2 目标
    for i, msg in enumerate(conversations):
        if msg.get("from") == "observation":
            if i + 1 < len(conversations) and conversations[i + 1].get("from") == "function_call":
                target_raw = conversations[i + 1].get("value", "")
                try:
                    target_obj = json.loads(target_raw)
                    return target_obj.get("name"), target_obj.get("arguments", {})
                except Exception:
                    # 非标准JSON则返回原串
                    return None, None
    return None, None


async def call_prod_sse_for_case(session: aiohttp.ClientSession, query: str, user_id: str = "1") -> Dict[str, Any]:
    """调用生产 SSE 接口，抓取第一跳检索Top5与最终选择的工具。"""
    payload = {
        "query": query,
        "prompt_template": "standard",
        "user_id": user_id,
        "role_code": 1,  # 必填字段
        "user_history": [],
        "save_method": 0,
        "is_vector": False,
        "is_probabilistic": False,
        "use_retrieval": True,
        "tool_category": None,
        "mcp_data": None,  # 应为字符串或 None，不是对象
        "front_data": {},
        "message_id": f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }

    retrieved_top5: List[str] = []
    predicted_tool: Optional[Dict[str, Any]] = None
    retrieval_call_params: Optional[Dict[str, Any]] = None
    # 作为回退的数据缓存
    _delta_toolcall_buffer: List[str] = []
    _last_result_delta: Optional[Dict[str, Any]] = None

    headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "application/json"
    }

    event_count = 0
    http_status = None
    error_msg = None

    # 最长等待时长与最短驻留，避免瞬时退出
    max_duration_sec = 30
    min_duration_sec = 3
    start_ts = datetime.now().timestamp()

    async with session.post(PROD_SSE_URL, json=payload, headers=headers) as resp:
        http_status = resp.status
        if http_status != 200:
            try:
                text = await resp.text()
            except Exception:
                text = ""
            return {
                "retrieved_top5": [],
                "predicted_tool": None,
                "retrieval_call_params": None,
                "http_status": http_status,
                "error": f"non-200 response: {http_status}",
                "response_preview": text[:1000]
            }

        current_event = None
        try:
            async for raw in resp.content:
                try:
                    line = raw.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if not line:
                    continue
                # 忽略 SSE id 行
                if line.startswith("id:"):
                    continue
                if line.startswith("event:"):
                    current_event = line.split("event:", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line.split("data:", 1)[1].strip()
                if data_str == "[DONE]":
                    # 若流很快结束但仍未达到最短驻留，则继续等待片刻以容错代理缓冲
                    if datetime.now().timestamp() - start_ts < min_duration_sec:
                        await asyncio.sleep(min_duration_sec - (datetime.now().timestamp() - start_ts))
                    break
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue

                event_count += 1
                if current_event == "tool_call.create.delta":
                    # 增量工具调用内容，通常是字符串形式，累积以便必要时解析
                    content = data.get("content")
                    if isinstance(content, str) and content:
                        _delta_toolcall_buffer.append(content)

                elif current_event == "tool_call.created":
                    tool_call = data.get("tool_call")
                    if isinstance(tool_call, dict):
                        name = tool_call.get("name")
                        if name == "retrieval_tool" and retrieval_call_params is None:
                            retrieval_call_params = tool_call
                        elif name and name != "retrieval_tool":
                            predicted_tool = tool_call

                elif current_event == "tool_response.completed":
                    result = data.get("result_delta")
                    if isinstance(result, dict):
                        _last_result_delta = result
                    names: List[str] = []
                    if isinstance(result, list):
                        for item in result[:5]:
                            if isinstance(item, dict) and isinstance(item.get("name"), str):
                                names.append(item["name"])
                    elif isinstance(result, dict):
                        tools = result.get("tools") or []
                        for item in tools[:5]:
                            if isinstance(item, dict) and isinstance(item.get("name"), str):
                                names.append(item["name"])
                        # 回退：从 tool_calling_chain 中解析候选工具
                        if not names:
                            chain = result.get("tool_calling_chain") or []
                            if isinstance(chain, list) and chain:
                                first = chain[0]
                                t_resp = first.get("tool_response") if isinstance(first, dict) else None
                                if isinstance(t_resp, list):
                                    for item in t_resp[:5]:
                                        if isinstance(item, dict) and isinstance(item.get("name"), str):
                                            names.append(item["name"])
                    if names:
                        retrieved_top5 = names

                elif current_event in ("answer.completed", "response.completed"):
                    # 进一步回退：某些实现会把完整链路放到 round_data 或 usage 区域
                    # 这里尝试从 data 中再次提取 tool_calling_chain 作为候选
                    chain = data.get("tool_calling_chain") or {}
                    if not chain:
                        round_data = data.get("round_data") or {}
                        chain = round_data.get("tool_calling_chain") if isinstance(round_data, dict) else {}
                    names: List[str] = []
                    if isinstance(chain, list) and chain:
                        first = chain[0]
                        t_resp = first.get("tool_response") if isinstance(first, dict) else None
                        if isinstance(t_resp, list):
                            for item in t_resp[:5]:
                                if isinstance(item, dict) and isinstance(item.get("name"), str):
                                    names.append(item["name"])
                    if names and not retrieved_top5:
                        retrieved_top5 = names

                # 超时保护：若超过最长时长则跳出
                if datetime.now().timestamp() - start_ts > max_duration_sec:
                    error_msg = f"timeout after {max_duration_sec}s"
                    break
        except Exception as e:
            error_msg = str(e)

    return {
        "retrieved_top5": retrieved_top5,
        "predicted_tool": predicted_tool,
        "retrieval_call_params": retrieval_call_params,
        "http_status": http_status,
        "event_count": event_count,
        **({"error": error_msg} if error_msg else {})
    }


def _extract_tool_names_from_response(resp_obj: Dict[str, Any], top_k: int = 5) -> List[str]:
    names: List[str] = []
    try:
        if isinstance(resp_obj.get("result"), list):
            for item in resp_obj["result"][:top_k]:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    names.append(item["name"])
        elif isinstance(resp_obj.get("result"), dict):
            tools = resp_obj["result"].get("tools") or []
            for item in tools[:top_k]:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    names.append(item["name"])
        elif isinstance(resp_obj.get("data"), list):
            for item in resp_obj["data"][:top_k]:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    names.append(item["name"])
        if not names:
            # 兜底在全文中提取 "name": "..."
            text = json.dumps(resp_obj, ensure_ascii=False)
            import re as _re
            names = _re.findall(r'"name"\s*:\s*"([^"]+)"', text)[:top_k]
    except Exception:
        pass
    return names[:top_k]


async def call_retrieval_endpoint(session: aiohttp.ClientSession, retrieval_call_params: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "jsonrpc": "2.0",
        "id": "eval_req_001",
        "method": "tools/call",
        "params": {
            "name": "retrieval_tool",
            "arguments": retrieval_call_params.get("arguments", retrieval_call_params) or {}
        }
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    status = None
    try:
        async with session.post(RETRIEVAL_ENDPOINT, json=payload, headers=headers) as resp:
            status = resp.status
            try:
                data = await resp.json()
            except Exception:
                data = {"raw": (await resp.text())[:2000]}
            names = _extract_tool_names_from_response(data, top_k=5) if status == 200 else []
            return {
                "retrieval_http_status": status,
                "retrieval_response_preview": json.dumps(data, ensure_ascii=False)[:2000],
                "retrieved_top5": names
            }
    except Exception as e:
        return {
            "retrieval_http_status": status,
            "retrieval_error": str(e),
            "retrieved_top5": []
        }


def calc_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"total": 0, "recall@5": 0.0, "precision@1": 0.0, "arg_accuracy": 0.0, "arg_denominator": 0}
    recall_hits = 0
    precision_hits = 0
    precision_denominator = 0  # precision@1的分母：recall@5成功的样本数
    arg_hits = 0
    arg_total = 0  # 仅统计 recall 成功的样本
    for r in rows:
        tgt_name = r.get("target_tool_name")
        retrieved = r.get("retrieved_top5", []) or []
        pred = (r.get("predicted_tool") or {}).get("name")
        recall_success = tgt_name and tgt_name in retrieved
        if recall_success:
            recall_hits += 1
            precision_denominator += 1  # 只有在recall成功时才计入precision分母
            # precision@1只在recall@5成功的样本中计算
            if tgt_name and pred and tgt_name == pred:
                precision_hits += 1
        # 仅在 arg_match 有效（非 None）时计入参数准确率
        if r.get("arg_match") is not None:
            arg_total += 1
            if r.get("arg_match") == 1:
                arg_hits += 1
    return {
        "total": total,
        "recall@5": recall_hits / total,
        "precision@1": (precision_hits / precision_denominator) if precision_denominator > 0 else 0.0,
        "precision_denominator": precision_denominator,  # 记录precision@1的分母
        "arg_accuracy": (arg_hits / arg_total) if arg_total > 0 else 0.0,
        "arg_denominator": arg_total
    }


async def main():
    parser = argparse.ArgumentParser(description="基于生产SSE接口的评估")
    parser.add_argument("--input_file", "-i", type=str, required=True)
    parser.add_argument("--output_file", "-o", type=str, required=True)
    parser.add_argument("--start_idx", "-s", type=int, default=0)
    parser.add_argument("--end_idx", "-e", type=int, default=50)
    parser.add_argument("--user_id", type=str, default=os.getenv("EVAL_USER_ID", "1"))
    parser.add_argument("--checkpoint_file", "-c", type=str, default=None,
                        help="断点续跑文件路径；提供则每个case完成后都会写入断点进度")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    start = max(0, args.start_idx)
    end = min(len(data), args.end_idx if args.end_idx is not None else len(data))
    if start >= end:
        print("无有效评估范围")
        return

    results: List[Dict[str, Any]] = []
    processed_indices: set = set()

    # 读取断点文件
    if args.checkpoint_file and os.path.exists(args.checkpoint_file):
        try:
            with open(args.checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            results = ckpt.get("results", [])
            processed_indices = set(ckpt.get("processed_indices", []))
            print(f"🔁 从断点恢复：已处理 {len(processed_indices)} 个cases；继续评估...")
        except Exception as e:
            print(f"⚠️ 读取断点失败，将从头开始: {e}")
            results = []
            processed_indices = set()

    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=5, limit_per_host=5)
    total_cases = end - start
    print(f"\n开始评估，共 {total_cases} 个cases ({start} 到 {end-1})")
    print("=" * 60)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for idx in range(start, end):
            if idx in processed_indices:
                # 跳过已处理
                current_num = (idx - start + 1)
                total_cases = end - start
                print(f"[{current_num}/{total_cases}] 跳过已完成的 case {idx}")
                continue
            item = data[idx]
            user_query = _extract_user_query_from_conversation(item)
            target_tool_name, target_args = _extract_pair2_target_tool(item)
            
            current_num = idx - start + 1
            print(f"[{current_num}/{total_cases}] 处理 case {idx}...", end=" ", flush=True)

            try:
                r = await call_prod_sse_for_case(session, user_query, user_id=args.user_id)
            except Exception as e:
                r = {"error": str(e), "retrieved_top5": [], "predicted_tool": None, "retrieval_call_params": None}
                print(f"❌ SSE调用失败: {str(e)[:50]}")

            # 若前端SSE未直接给出 Top5，则根据第一跳 retrieval 的 query 再请求 6227 网关获取Top5
            if (not r.get("retrieved_top5")) and r.get("retrieval_call_params"):
                try:
                    retrieval_summary = await call_retrieval_endpoint(session, r["retrieval_call_params"])
                    r.update(retrieval_summary)
                except Exception as e:
                    r.update({"retrieval_error": str(e)})

            # 计算该case的recall@5（1表示成功，0表示失败）
            retrieved_top5 = r.get("retrieved_top5", []) or []
            recall_value = 1 if (target_tool_name and target_tool_name in retrieved_top5) else 0
            
            # 计算第二轮 function_call 的参数匹配（与标注 target_arguments 比较）
            # 仅在 recall 成功（recall_value==1）且工具名称也匹配时才计算参数准确率
            pred_tool_name = (r.get("predicted_tool") or {}).get("name")
            pred_args = (r.get("predicted_tool") or {}).get("arguments") or {}
            tgt_args = target_args or {}
            arg_details = {}
            # 只有工具名称匹配时，参数匹配才有意义
            if recall_value == 1 and pred_tool_name == target_tool_name and isinstance(tgt_args, dict) and tgt_args:
                all_match = True
                for k, v in tgt_args.items():
                    pv = pred_args.get(k)
                    is_match = (pv == v)
                    arg_details[k] = {"target": v, "predict": pv, "match": is_match}
                    if not is_match:
                        all_match = False
                arg_match_value = 1 if all_match else 0
            else:
                arg_match_value = None  # 跳过参数评估（工具未召回、工具名不匹配或无目标参数）

            row = {
                "index": idx,
                "user_query": user_query,
                "target_tool_name": target_tool_name,
                "target_arguments": target_args,
                "recall@5": recall_value,  # 添加recall字段：1表示成功，0表示失败
                "arg_match": arg_match_value,  # 第二轮参数是否与标注完全一致（1/0）
                "arg_match_details": arg_details,
                **r
            }
            results.append(row)
            processed_indices.add(idx)
            
            # 打印进度信息
            retrieved_str = f"检索到{len(retrieved_top5)}个工具" if retrieved_top5 else "未检索到工具"
            recall_str = "✅ recall成功" if recall_value == 1 else "❌ recall失败"
            pred_name = (r.get("predicted_tool") or {}).get("name") or "无"
            if arg_match_value is None:
                if recall_value == 0:
                    arg_str = "⏭️ 参数评估跳过(未召回)"
                elif pred_tool_name != target_tool_name:
                    arg_str = f"⏭️ 参数评估跳过(工具不匹配: {pred_name} ≠ {target_tool_name})"
                else:
                    arg_str = "⏭️ 参数评估跳过(无目标参数)"
            else:
                arg_str = "✅ 参数完全匹配" if arg_match_value == 1 else "❌ 参数不匹配"
            print(f"{retrieved_str} | {recall_str} | 预测工具: {pred_name} | {arg_str}")

            # 实时保存断点
            if args.checkpoint_file:
                try:
                    ckpt_data = {
                        "results": results,
                        "processed_indices": sorted(list(processed_indices)),
                        "meta": {
                            "input_file": args.input_file,
                            "output_file": args.output_file,
                            "user_id": args.user_id,
                            "range": [start, end]
                        }
                    }
                    os.makedirs(os.path.dirname(args.checkpoint_file), exist_ok=True)
                    with open(args.checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(ckpt_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"⚠️ 写入断点失败: {e}")

    print("=" * 60)
    print(f"评估完成，共处理 {len(results)} 个cases\n")

    metrics = calc_metrics(results)
    report = {
        "summary": {
            "api": PROD_SSE_URL,
            "user_id": args.user_id,
            "start_idx": start,
            "end_idx": end,
            "metrics": metrics
        },
        "cases": results
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ 评估结果已保存: {args.output_file}")
    print(f"📊 总体指标: recall@5={metrics['recall@5']:.3f}, precision@1={metrics['precision@1']:.3f}\n")

    # 评估完成后删除断点（可选）
    if args.checkpoint_file and os.path.exists(args.checkpoint_file):
        try:
            os.remove(args.checkpoint_file)
            print(f"🗑️ 已删除断点文件: {args.checkpoint_file}")
        except Exception as e:
            print(f"⚠️ 删除断点文件失败: {e}")
    
    # 筛选recall=0的cases，单独保存
    recall_failed_cases = [case for case in results if case.get("recall@5", 0) == 0]
    if recall_failed_cases:
        failed_output_file = args.output_file.replace(".json", "_recall_failed.json")
        failed_report = {
            "summary": {
                "api": PROD_SSE_URL,
                "user_id": args.user_id,
                "start_idx": start,
                "end_idx": end,
                "total_failed": len(recall_failed_cases),
                "total_cases": len(results),
                "failure_rate": len(recall_failed_cases) / len(results) if results else 0.0
            },
            "cases": recall_failed_cases
        }
        with open(failed_output_file, "w", encoding="utf-8") as f:
            json.dump(failed_report, f, ensure_ascii=False, indent=2)
        print(f"❌ Recall失败cases已保存: {failed_output_file} (共 {len(recall_failed_cases)} 条)")
        print(f"   失败率: {len(recall_failed_cases) / len(results) * 100:.1f}%")
    else:
        print("✅ 所有cases的recall@5都成功！")

    # 筛选precision@1失败的cases（仅在recall@5成功的样本中，预测工具名不等于目标工具名）
    precision_failed_cases = []
    precision_eligible = 0  # recall@5成功的样本数（precision@1的分母）
    for case in results:
        recall_success = case.get("recall@5", 0) == 1
        if recall_success:
            precision_eligible += 1
        tgt = case.get("target_tool_name")
        pred = (case.get("predicted_tool") or {}).get("name")
        # 只有recall@5成功，但预测工具名不等于目标工具名时，才算precision@1失败
        if recall_success and tgt and (pred != tgt):
            precision_failed_cases.append(case)
    if precision_failed_cases:
        precision_failed_output = args.output_file.replace(".json", "_precision_failed.json")
        precision_failed_report = {
            "summary": {
                "api": PROD_SSE_URL,
                "user_id": args.user_id,
                "start_idx": start,
                "end_idx": end,
                "total_failed": len(precision_failed_cases),
                "total_cases": len(results),
                "precision_eligible": precision_eligible,  # recall@5成功的样本数
                "failure_rate": (len(precision_failed_cases) / precision_eligible) if precision_eligible > 0 else 0.0
            },
            "cases": precision_failed_cases
        }
        with open(precision_failed_output, "w", encoding="utf-8") as f:
            json.dump(precision_failed_report, f, ensure_ascii=False, indent=2)
        print(f"❌ Precision@1失败cases已保存: {precision_failed_output} (共 {len(precision_failed_cases)} 条)")
        if precision_eligible > 0:
            print(f"   基于recall@5成功的样本失败率: {len(precision_failed_cases) / precision_eligible * 100:.1f}%  (可评估 {precision_eligible} 条)")
    else:
        if precision_eligible > 0:
            print("✅ 所有recall@5成功的cases在precision@1上均命中！")
        else:
            print("ℹ️ 本次无recall@5成功的样本，无法评估precision@1")

    # 筛选参数匹配失败（arg_accuracy失败）的cases（仅统计被评估样本 arg_match==0）
    arg_failed_cases = [case for case in results if case.get("arg_match") == 0]
    arg_eligible = sum(1 for case in results if case.get("arg_match") is not None)
    if arg_failed_cases:
        arg_failed_output = args.output_file.replace(".json", "_arg_failed.json")
        arg_failed_report = {
            "summary": {
                "api": PROD_SSE_URL,
                "user_id": args.user_id,
                "start_idx": start,
                "end_idx": end,
                "total_failed": len(arg_failed_cases),
                "total_eligible": arg_eligible,
                "eligible_failure_rate": (len(arg_failed_cases) / arg_eligible) if arg_eligible else 0.0
            },
            "cases": arg_failed_cases
        }
        with open(arg_failed_output, "w", encoding="utf-8") as f:
            json.dump(arg_failed_report, f, ensure_ascii=False, indent=2)
        print(f"❌ 参数匹配失败cases已保存: {arg_failed_output} (共 {len(arg_failed_cases)} 条)")
        if arg_eligible:
            print(f"   基于可评估样本失败率: {len(arg_failed_cases) / arg_eligible * 100:.1f}%  (可评估 {arg_eligible} 条)")
    else:
        if arg_eligible:
            print("✅ 所有可评估的cases参数完全匹配！")
        else:
            print("ℹ️ 本次无可评估的参数匹配样本（arg_match 皆为 None）")

    # 筛选完全成功的cases（同时满足recall@5、precision@1和arg_accuracy成功）
    success_cases = []
    for case in results:
        recall_success = case.get("recall@5", 0) == 1
        tgt_name = case.get("target_tool_name")
        pred_tool = case.get("predicted_tool") or {}
        pred_name = pred_tool.get("name")
        precision_success = recall_success and tgt_name and (pred_name == tgt_name)
        arg_success = case.get("arg_match") == 1
        
        # 同时满足三个条件才算成功
        if recall_success and precision_success and arg_success:
            success_cases.append(case)
    
    if success_cases:
        success_output = args.output_file.replace(".json", "_success.json")
        success_report = {
            "summary": {
                "api": PROD_SSE_URL,
                "user_id": args.user_id,
                "start_idx": start,
                "end_idx": end,
                "total_success": len(success_cases),
                "total_cases": len(results),
                "success_rate": len(success_cases) / len(results) if results else 0.0
            },
            "cases": success_cases
        }
        with open(success_output, "w", encoding="utf-8") as f:
            json.dump(success_report, f, ensure_ascii=False, indent=2)
        print(f"✅ 完全成功cases已保存: {success_output} (共 {len(success_cases)} 条)")
        print(f"   成功率: {len(success_cases) / len(results) * 100:.1f}%")
    else:
        print("ℹ️ 本次无完全成功的cases（需同时满足recall@5、precision@1和arg_accuracy）")


if __name__ == "__main__":
    asyncio.run(main())


