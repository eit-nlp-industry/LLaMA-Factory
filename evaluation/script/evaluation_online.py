#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本
功能：
1. 读取JSON文件中的对话数据
2. 提取human的value作为query调用server:8020
3. 处理流式返回结果
4. 对比和存储结果
"""

import json
import httpx
import asyncio
import time
import re
import os
from typing import Dict, List, Any
from utils.custom_logging import setup_logging
from utils.extraction import extract_json_from_string
from loguru import logger
from collections import Counter
setup_logging()


class ModelEvaluator:
    def __init__(self, server_url: str = "http://localhost:8020/mcp_end2end/stream"):
        self.server_url = server_url
        self.results = []
        self.client = None
        self.start_time = None
        self.error_count = 0
        self.success_count = 0
        
    def load_data(self, file_path: str) -> List[Dict]:
        """加载JSON数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载数据文件，共{len(data)}条记录")
            return data
        except Exception as e:
            logger.error(f"加载数据文件失败: {e}")
            return []
    
    def extract_human_queries(self, data: List[Dict]) -> List[Dict]:
        """提取所有human的value作为query"""
        queries = []
        for i, item in enumerate(data):
            if 'conversations' in item:
                for conv in item['conversations']:
                    if conv.get('from') == 'human':
                        query_data = {
                            'index': i,
                            'query': conv.get('value', ''),
                            'original_data': item
                        }
                        queries.append(query_data)
                        break  # 只取第一个human的value
        logger.info(f"提取到{len(queries)}个查询")
        return queries
    
    def parse_sse_events(self, sse_content: str, filter_events: List[str] = None) -> List[Dict]:
        """
        解析SSE格式的内容，提取指定类型的事件
        
        Args:
            sse_content: SSE格式的文本内容（可以是多行）
            filter_events: 需要过滤的事件类型列表，如果为None则解析所有事件
        
        Returns:
            解析成功的事件列表
        """
        events = []
        current_event = {}
        parsed_count = 0
        failed_count = 0
        
        # 如果没有指定过滤事件，设置默认过滤
        if filter_events is None:
            filter_events = ['tool_call.created', 'tool_response.completed']
        
        for line in sse_content.split('\n'):
            line = line.strip()
            if not line:
                # 空行表示一个完整的事件结束
                if current_event and 'event' in current_event and 'data' in current_event:
                    event_type = current_event['event']
                    
                    # 只处理我们关心的事件类型
                    if event_type in filter_events:
                        # 使用extract_json_from_string解析data字段中的JSON
                        data_content = extract_json_from_string(current_event['data'])
                        if data_content is not None:
                            event_obj = {
                                'id': current_event.get('id'),
                                'event': current_event['event'],
                                'data': data_content
                            }
                            events.append(event_obj)
                            parsed_count += 1
                            logger.debug(f"✅ 成功解析事件: {current_event['event']} (ID: {current_event.get('id', 'N/A')})")
                        else:
                            failed_count += 1
                            logger.warning(f"❌ 无法解析事件数据: {current_event['event']} - {current_event['data'][:100]}...")
                current_event = {}
                continue
            
            # 解析SSE格式的字段
            if line.startswith('id: '):
                current_event['id'] = line[4:]
            elif line.startswith('event: '):
                current_event['event'] = line[7:]
            elif line.startswith('data: '):
                current_event['data'] = line[6:]
                # 检查是否是结束标记
                if current_event['data'].strip() == '[DONE]':
                    logger.debug("收到结束标记 [DONE]")
                    break
            else:
                logger.debug(f"未知格式的行: {line}")
        
        # 处理最后一个事件（如果没有空行结尾）
        if current_event and 'event' in current_event and 'data' in current_event:
            event_type = current_event['event']
            
            # 只处理我们关心的事件类型
            if event_type in filter_events:
                data_content = extract_json_from_string(current_event['data'])
                if data_content is not None:
                    event_obj = {
                        'id': current_event.get('id'),
                        'event': current_event['event'],
                        'data': data_content
                    }
                    events.append(event_obj)
                    parsed_count += 1
                    logger.debug(f"✅ 成功解析最后一个事件: {current_event['event']} (ID: {current_event.get('id', 'N/A')})")
                else:
                    failed_count += 1
                    logger.warning(f"❌ 无法解析最后一个事件数据: {current_event['event']} - {current_event['data'][:100]}...")
        
        # 统计和日志输出
        logger.info(f"=== SSE解析结果统计 ===")
        logger.info(f"成功解析事件数: {parsed_count}")
        logger.info(f"解析失败事件数: {failed_count}")
        logger.info(f"总事件数: {len(events)}")
        
        if events:
            event_types = [event.get('event', 'unknown') for event in events]
            event_counts = Counter(event_types)
            logger.info(f"事件类型分布: {dict(event_counts)}")
        else:
            logger.warning("⚠️ 未解析到任何目标事件")
        
        return events

    async def call_server(self, query: str, max_retries: int = 3, retry_delay: float = 2.0) -> List[Dict]:
        """异步调用server:8020端口，处理流式返回，支持重试机制"""
        payload = {
            "user_id": "166",
            "role_code": 1,
            "query": query,
            "save_method": 0
        }
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    async with client.stream(
                        'POST',
                        self.server_url,
                        json=payload,
                        headers={'Accept': 'text/event-stream'}
                    ) as response:
                        response.raise_for_status()
                        
                        # 收集所有SSE文本内容
                        sse_content = ""
                        async for line in response.aiter_text():
                            logger.debug(f"Received data: {line}")
                            sse_content += line
                            
                            # 检查是否收到结束标记
                            if '[DONE]' in line:
                                logger.debug("收到结束标记 [DONE]")
                                break
                        
                        # 使用封装的方法解析SSE内容，只解析我们关心的事件
                        events = self.parse_sse_events(
                            sse_content, 
                            filter_events=['tool_call.created', 'tool_response.completed']
                        )
                        
                        # 验证关键事件类型
                        has_tool_call = any(event.get('event') == 'tool_call.created' for event in events)
                        has_tool_response = any(event.get('event') == 'tool_response.completed' for event in events)
                        logger.info(f"包含工具调用事件: {'✅' if has_tool_call else '❌'}")
                        logger.info(f"包含工具响应事件: {'✅' if has_tool_response else '❌'}")
                        
                        return events
                        
            except httpx.RequestError as e:
                logger.warning(f"Call server failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    logger.error(f"All retry attempts failed for query: {query[:50]}...")
                    raise Exception(f"Server connection failed after {max_retries} attempts: {e}")
            except httpx.TimeoutException as e:
                logger.warning(f"Server timeout (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Timeout after all retry attempts for query: {query[:50]}...")
                    raise Exception(f"Server timeout after {max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing response (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Unexpected error after all retry attempts for query: {query[:50]}...")
                    raise Exception(f"Unexpected error after {max_retries} attempts: {e}")
        
        raise Exception("All retry attempts exhausted")
    
    def extract_tool_calls_and_observations(self, events: List[Dict]) -> Dict[str, List]:
        """Extract tool_call.created and tool_response.completed content from events"""
        tool_calls = []
        tool_responses = []
        
        logger.debug(f"开始提取工具调用和响应，共 {len(events)} 个事件")
        
        for event in events:
            event_type = event.get('event')
            event_data = event.get('data', {})
            
            if event_type == 'tool_call.created':
                logger.debug(f"Extract tool_call.created content: {event}")
                # Extract tool_call information
                tool_call_info = event_data.get('tool_call', {})
                if tool_call_info:
                    tool_calls.append(tool_call_info)  # 直接存储JSON对象
                    logger.debug(f"✅ 提取工具调用: {tool_call_info.get('name', 'unknown')}")
                else:
                    logger.warning(f"❌ tool_call.created 事件中缺少 tool_call 信息")
                    
            elif event_type == 'tool_response.completed':
                logger.debug(f"Extract tool_response.completed content: {event}")
                # Extract tool_response information

                if 'result_delta' in event_data:
                    tool_response = event_data['result_delta'].get('result', [])
                    tool_responses.append(tool_response)  # 直接存储JSON对象
                    logger.debug(f"✅ 提取工具响应: {len(str(tool_response))} 字符")
                else:
                    tool_response = []
                    tool_responses.append(tool_response)  # 直接存储JSON对象
        
        logger.info(f"Extract {len(tool_calls)} tool calls, {len(tool_responses)} tool responses")
        return {
            'tool_calls': tool_calls,
            'tool_responses': tool_responses
        }
    
    def extract_original_data(self, original_data: Dict) -> Dict[str, List]:
        """Extract function_call and observation content from original data"""
        function_calls = []
        observations = []
        
        if 'conversations' in original_data:
            for conv in original_data['conversations']:
                if conv.get('from') == 'function_call':
                    # 解析JSON字符串为对象
                    try:
                        function_call_obj = json.loads(conv.get('value', '{}'))
                        function_calls.append(function_call_obj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析function_call JSON时出错: {e}")
                        function_calls.append({})
                elif conv.get('from') == 'observation':
                    # 解析JSON字符串为对象
                    try:
                        observation_obj = json.loads(conv.get('value', '[]'))
                        observations.append(observation_obj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析observation JSON时出错: {e}")
                        observations.append([])
        
        return {
            'function_calls': function_calls,
            'observations': observations
        }
    
    def compare_tool_call(self, server_call: Dict, original_call: Dict) -> Dict:
        """比较单个工具调用，检查name和arguments的匹配度"""
        try:
            # 检查name是否一致
            name_match = server_call.get('name') == original_call.get('name')
            name_score = 1.0 if name_match else 0.0
            
            # 检查arguments是否一致
            server_args = server_call.get('arguments', {})
            original_args = original_call.get('arguments', {})
            arguments_match = server_args == original_args
            arguments_score = 1.0 if arguments_match else 0.0
            
            return {
                'name_match': name_match,
                'name_score': name_score,
                'arguments_match': arguments_match,
                'arguments_score': arguments_score,
                'server_name': server_call.get('name', ''),
                'original_name': original_call.get('name', ''),
                'server_arguments': server_args,
                'original_arguments': original_args
            }
        except (KeyError, TypeError) as e:
            logger.warning(f"比较工具调用时出错: {e}")
            return {
                'name_match': False,
                'name_score': 0.0,
                'arguments_match': False,
                'arguments_score': 0.0,
                'server_name': '',
                'original_name': '',
                'server_arguments': {},
                'original_arguments': {},
                'error': str(e)
            }

    def compare_results(self, server_data: Dict[str, List], original_data: Dict[str, List]) -> Dict:
        """详细比较服务器返回结果和原始数据"""
        
        # 初始化比较结果结构
        comparison = {
            'tool_calls_comparison': {
                'server_count': len(server_data['tool_calls']),
                'original_count': len(original_data['function_calls']),
                'detailed_scores': [],
                'name_average_score': 0.0,
                'arguments_average_score': 0.0,
                'non_retrieval_name_average_score': 0.0,
                'non_retrieval_arguments_average_score': 0.0
            },
            'tool_responses_comparison': {
                'server_count': len(server_data['tool_responses']),
                'original_count': len(original_data['observations']),
                'detailed_scores': [],
                'average_score': 0.0
            },
            'overall_scores': {
                'tool_responses_avg': 0.0
            }
        }
        
        # 1. 比较工具调用 (tool_calls)
        tool_call_name_scores = []
        tool_call_arguments_scores = []
        non_retrieval_name_scores = []
        non_retrieval_arguments_scores = []
        max_tool_calls = max(len(server_data['tool_calls']), len(original_data['function_calls']))
        
        for i in range(max_tool_calls):
            server_call = server_data['tool_calls'][i] if i < len(server_data['tool_calls']) else None
            original_call = original_data['function_calls'][i] if i < len(original_data['function_calls']) else None
            
            if server_call is None:
                # 服务器缺少该调用
                score_detail = {
                    'index': i,
                    'server_present': False,
                    'original_present': True,
                    'name_score': 0.0,
                    'arguments_score': 0.0,
                    'original_call': original_call
                }
            elif original_call is None:
                # 原始数据缺少该调用
                score_detail = {
                    'index': i,
                    'server_present': True,
                    'original_present': False,
                    'name_score': 0.0,
                    'arguments_score': 0.0,
                    'server_call': server_call
                }
            else:
                # 两者都存在，进行详细比较
                # 直接使用JSON对象
                call_comparison = self.compare_tool_call(server_call, original_call)
                score_detail = {
                    'index': i,
                    'server_present': True,
                    'original_present': True,
                    'name_score': call_comparison['name_score'],
                    'arguments_score': call_comparison['arguments_score'],
                    'name_match': call_comparison['name_match'],
                    'arguments_match': call_comparison['arguments_match'],
                    'server_name': call_comparison['server_name'],
                    'original_name': call_comparison['original_name'],
                    'server_call': server_call,  # 直接存储JSON对象
                    'original_call': original_call  # 直接存储JSON对象
                }
                if 'error' in call_comparison:
                    score_detail['error'] = call_comparison['error']
            
            comparison['tool_calls_comparison']['detailed_scores'].append(score_detail)
            tool_call_name_scores.append(score_detail['name_score'])
            tool_call_arguments_scores.append(score_detail['arguments_score'])
            
            # 收集非retrieval_tool的评分
            if server_call and original_call:
                server_name = server_call.get('name', '')
                original_name = original_call.get('name', '')
                # 只有当两个都不是retrieval_tool时才计入非retrieval评分
                if server_name != 'retrieval_tool' and original_name != 'retrieval_tool':
                    non_retrieval_name_scores.append(score_detail['name_score'])
                    non_retrieval_arguments_scores.append(score_detail['arguments_score'])
        
        # 计算工具调用name和arguments分别的平均分
        comparison['tool_calls_comparison']['name_average_score'] = (
            sum(tool_call_name_scores) / len(tool_call_name_scores) if tool_call_name_scores else 0.0
        )
        comparison['tool_calls_comparison']['arguments_average_score'] = (
            sum(tool_call_arguments_scores) / len(tool_call_arguments_scores) if tool_call_arguments_scores else 0.0
        )
        
        # 计算非retrieval_tool的name和arguments分别的平均分
        comparison['tool_calls_comparison']['non_retrieval_name_average_score'] = (
            sum(non_retrieval_name_scores) / len(non_retrieval_name_scores) if non_retrieval_name_scores else 0.0
        )
        comparison['tool_calls_comparison']['non_retrieval_arguments_average_score'] = (
            sum(non_retrieval_arguments_scores) / len(non_retrieval_arguments_scores) if non_retrieval_arguments_scores else 0.0
        )
        
        # 2. 比较工具响应 (tool_responses)
        tool_response_scores = []
        max_tool_responses = max(len(server_data['tool_responses']), len(original_data['observations']))
        
        for i in range(max_tool_responses):
            server_response = server_data['tool_responses'][i] if i < len(server_data['tool_responses']) else None
            original_response = original_data['observations'][i] if i < len(original_data['observations']) else None
            
            if server_response is None:
                # 服务器缺少该响应
                score_detail = {
                    'index': i,
                    'server_present': False,
                    'original_present': True,
                    'match_score': 0.0,
                    'original_response': original_response
                }
            elif original_response is None:
                # 原始数据缺少该响应
                score_detail = {
                    'index': i,
                    'server_present': True,
                    'original_present': False,
                    'match_score': 0.0,
                    'server_response': server_response
                }
            else:
                # 两者都存在，比较完全一致性
                responses_match = server_response == original_response
                match_score = 1.0 if responses_match else 0.0
                
                score_detail = {
                    'index': i,
                    'server_present': True,
                    'original_present': True,
                    'match_score': match_score,
                    'responses_match': responses_match,
                    'server_response': server_response,  # 直接存储JSON对象
                    'original_response': original_response  # 直接存储JSON对象
                }
            
            comparison['tool_responses_comparison']['detailed_scores'].append(score_detail)
            tool_response_scores.append(score_detail['match_score'])
        
        # 计算工具响应平均分
        comparison['tool_responses_comparison']['average_score'] = (
            sum(tool_response_scores) / len(tool_response_scores) if tool_response_scores else 0.0
        )
        
        # 3. 计算总体评分
        comparison['overall_scores']['tool_responses_avg'] = comparison['tool_responses_comparison']['average_score']
        
        # 保持向后兼容性
        comparison['tool_calls_match'] = (
            comparison['tool_calls_comparison']['name_average_score'] == 1.0 and 
            comparison['tool_calls_comparison']['arguments_average_score'] == 1.0
        )
        comparison['tool_responses_match'] = comparison['overall_scores']['tool_responses_avg'] == 1.0
        
        return comparison
    
    def calculate_global_scores(self, results: List[Dict]) -> Dict:
        """计算多个结果的全局评分"""
        if not results:
            return {
                'global_tool_responses_avg': 0.0,
                'global_tool_calls_name_avg': 0.0,
                'global_tool_calls_arguments_avg': 0.0,
                'global_non_retrieval_name_avg': 0.0,
                'global_non_retrieval_arguments_avg': 0.0,
                'total_queries': 0
            }
        
        # 收集所有评分
        all_tool_responses_scores = []
        all_tool_calls_name_scores = []
        all_tool_calls_arguments_scores = []
        all_non_retrieval_name_scores = []
        all_non_retrieval_arguments_scores = []
        
        for result in results:
            comparison = result.get('comparison', {})
            overall_scores = comparison.get('overall_scores', {})
            tool_calls_comparison = comparison.get('tool_calls_comparison', {})
            
            tool_responses_avg = overall_scores.get('tool_responses_avg', 0.0)
            
            # 收集每个工具调用的name和arguments分数
            detailed_scores = tool_calls_comparison.get('detailed_scores', [])
            for score_detail in detailed_scores:
                if score_detail.get('server_present') and score_detail.get('original_present'):
                    all_tool_calls_name_scores.append(score_detail.get('name_score', 0.0))
                    all_tool_calls_arguments_scores.append(score_detail.get('arguments_score', 0.0))
                    
                    # 收集非retrieval_tool的评分
                    server_call = score_detail.get('server_call', {})
                    original_call = score_detail.get('original_call', {})
                    if server_call and original_call:
                        server_name = server_call.get('name', '')
                        original_name = original_call.get('name', '')
                        # 只有当两个都不是retrieval_tool时才计入非retrieval评分
                        if server_name != 'retrieval_tool' and original_name != 'retrieval_tool':
                            all_non_retrieval_name_scores.append(score_detail.get('name_score', 0.0))
                            all_non_retrieval_arguments_scores.append(score_detail.get('arguments_score', 0.0))
            
            all_tool_responses_scores.append(tool_responses_avg)
        
        # 计算全局平均分
        global_tool_responses_avg = sum(all_tool_responses_scores) / len(all_tool_responses_scores) if all_tool_responses_scores else 0.0
        global_tool_calls_name_avg = sum(all_tool_calls_name_scores) / len(all_tool_calls_name_scores) if all_tool_calls_name_scores else 0.0
        global_tool_calls_arguments_avg = sum(all_tool_calls_arguments_scores) / len(all_tool_calls_arguments_scores) if all_tool_calls_arguments_scores else 0.0
        global_non_retrieval_name_avg = sum(all_non_retrieval_name_scores) / len(all_non_retrieval_name_scores) if all_non_retrieval_name_scores else 0.0
        global_non_retrieval_arguments_avg = sum(all_non_retrieval_arguments_scores) / len(all_non_retrieval_arguments_scores) if all_non_retrieval_arguments_scores else 0.0
        
        return {
            'global_tool_responses_avg': global_tool_responses_avg,
            'global_tool_calls_name_avg': global_tool_calls_name_avg,
            'global_tool_calls_arguments_avg': global_tool_calls_arguments_avg,
            'global_non_retrieval_name_avg': global_non_retrieval_name_avg,
            'global_non_retrieval_arguments_avg': global_non_retrieval_arguments_avg,
            'total_queries': len(results)
        }
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save evaluation results to file"""
        try:
            # 计算全局评分
            global_scores = self.calculate_global_scores(results)
            
            # 创建包含全局评分的完整结果
            complete_results = {
                'global_scores': global_scores,
                'results': results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Save results failed: {e}")
    
    def save_checkpoint(self, results: List[Dict], checkpoint_file: str, processed_count: int, total_count: int):
        """保存检查点文件"""
        try:
            checkpoint_data = {
                'processed_count': processed_count,
                'total_count': total_count,
                'results': results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Checkpoint saved: {processed_count}/{total_count} processed")
        except Exception as e:
            logger.error(f"Save checkpoint failed: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict:
        """加载检查点文件"""
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"Checkpoint loaded: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']} processed")
                return checkpoint_data
            else:
                logger.info("No checkpoint file found, starting from beginning")
                return None
        except Exception as e:
            logger.error(f"Load checkpoint failed: {e}")
            return None
    
    def print_progress(self, current: int, total: int, start_time: float):
        """打印进度信息"""
        if total == 0:
            return
        
        elapsed_time = time.time() - start_time
        progress_percent = (current / total) * 100
        
        if current > 0:
            avg_time_per_query = elapsed_time / current
            remaining_queries = total - current
            estimated_remaining_time = remaining_queries * avg_time_per_query
            
            logger.info(f"进度: {current}/{total} ({progress_percent:.1f}%) | "
                       f"成功: {self.success_count} | 错误: {self.error_count} | "
                       f"已用时间: {elapsed_time/60:.1f}分钟 | "
                       f"预计剩余: {estimated_remaining_time/60:.1f}分钟")
        else:
            logger.info(f"进度: {current}/{total} ({progress_percent:.1f}%) | "
                       f"成功: {self.success_count} | 错误: {self.error_count} | "
                       f"已用时间: {elapsed_time/60:.1f}分钟")
    
    def save_progress_report(self, output_file: str, current: int, total: int):
        """保存进度报告"""
        try:
            progress_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'current_progress': current,
                'total_queries': total,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'progress_percentage': (current / total * 100) if total > 0 else 0,
                'elapsed_time_minutes': (time.time() - self.start_time) / 60 if self.start_time else 0
            }
            
            progress_file = f"{output_file}.progress"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Save progress report failed: {e}")
    
    def generate_interruption_report(self, output_file: str, processed_count: int, total_queries: int, error_message: str):
        """生成中断报告"""
        try:
            total_time = time.time() - self.start_time if self.start_time else 0
            
            interruption_report = {
                'interruption_type': 'server_connection_failure',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processed_count': processed_count,
                'total_queries': total_queries,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'progress_percentage': (processed_count / total_queries * 100) if total_queries > 0 else 0,
                'elapsed_time_minutes': total_time / 60,
                'error_message': error_message,
                'resume_instructions': {
                    'checkpoint_file': f"{output_file}.checkpoint",
                    'command': f"await evaluator.evaluate(input_file='data/9.17_evaluate_data_top5_final.json', output_file='{output_file}', resume=True)",
                    'note': '使用 resume=True 参数从检查点恢复评估'
                }
            }
            
            interruption_file = f"{output_file}.interruption_report"
            with open(interruption_file, 'w', encoding='utf-8') as f:
                json.dump(interruption_report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"中断报告已保存到: {interruption_file}")
            
        except Exception as e:
            logger.error(f"Generate interruption report failed: {e}")
    
    async def evaluate(self, input_file: str, output_file: str = "evaluation_results.json", 
                      batch_size: int = 50, start_index: int = 0, max_queries: int = None,
                      checkpoint_file: str = None, resume: bool = True):
        """Execute complete evaluation process with batch processing and checkpoint support"""
        logger.info("Start model evaluation...")
        
        # 初始化时间跟踪
        self.start_time = time.time()
        self.error_count = 0
        self.success_count = 0
        
        # 设置检查点文件
        if checkpoint_file is None:
            checkpoint_file = f"{output_file}.checkpoint"
        
        # 尝试加载检查点
        checkpoint_data = None
        if resume:
            checkpoint_data = self.load_checkpoint(checkpoint_file)
            if checkpoint_data:
                self.results = checkpoint_data.get('results', [])
                start_index = checkpoint_data.get('processed_count', 0)
                self.success_count = len(self.results)  # 假设已处理的结果都是成功的
                logger.info(f"Resuming from checkpoint: {start_index} queries already processed")
        
        # 1. Load data
        data = self.load_data(input_file)
        if not data:
            logger.error("Cannot load data, evaluation terminated")
            return
        
        # 2. Extract queries
        queries = self.extract_human_queries(data)
        if not queries:
            logger.error("No valid queries found, evaluation terminated")
            return
        
        # 3. Apply limits and offsets
        if max_queries:
            queries = queries[:max_queries]
        
        if start_index > 0:
            queries = queries[start_index:]
            logger.info(f"Starting from index {start_index}, processing {len(queries)} queries")
        
        # 4. Process queries in batches
        total_queries = len(queries)
        processed_count = len(self.results)  # 从已有结果开始计数
        
        for batch_start in range(0, total_queries, batch_size):
            batch_end = min(batch_start + batch_size, total_queries)
            batch_queries = queries[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: queries {batch_start + 1}-{batch_end} of {total_queries}")
            
            # Process each query in the current batch
            for i, query_data in enumerate(batch_queries):
                global_index = batch_start + i
                logger.info(f"Process {global_index + 1}/{total_queries} query: {query_data['query'][:50]}...")
                
                try:
                    # 异步调用服务器
                    events = await self.call_server(query_data['query'])
                    
                    # Extract tool calls and responses from server
                    server_data = self.extract_tool_calls_and_observations(events)
                    
                    # Extract function_call and observation from original data
                    original_data = self.extract_original_data(query_data['original_data'])
                    
                    # Compare results
                    comparison = self.compare_results(server_data, original_data)
                    
                    # Save results
                    result = {
                        'index': query_data['index'],
                        'query': query_data['query'],
                        'server_events_count': len(events),
                        'server_tool_calls': server_data['tool_calls'],
                        'server_tool_responses': server_data['tool_responses'],
                        'original_function_calls': original_data['function_calls'],
                        'original_observations': original_data['observations'],
                        'comparison': comparison,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    self.results.append(result)
                    processed_count += 1
                    self.success_count += 1
                    
                    # 每处理10个查询保存一次检查点和进度报告
                    if processed_count % 10 == 0:
                        self.save_checkpoint(self.results, checkpoint_file, processed_count, total_queries)
                        self.save_progress_report(output_file, processed_count, total_queries)
                        self.print_progress(processed_count, total_queries, self.start_time)
                    
                    # Add delay to avoid server pressure
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing query {global_index + 1}: {e}")
                    self.error_count += 1
                    
                    # 检查是否是服务器连接失败（重试后仍然失败）
                    if "Server connection failed after" in str(e) or "Server timeout after" in str(e) or "Unexpected error after" in str(e):
                        logger.error(f"🚨 服务器连接失败，保存检查点并结束评估")
                        logger.error(f"失败查询: {query_data['query'][:50]}...")
                        
                        # 保存当前进度到检查点
                        self.save_checkpoint(self.results, checkpoint_file, processed_count, total_queries)
                        self.save_progress_report(output_file, processed_count, total_queries)
                        
                        # 生成中断报告
                        self.generate_interruption_report(output_file, processed_count, total_queries, str(e))
                        
                        logger.error(f"评估因服务器连接失败而中断")
                        logger.error(f"已处理 {processed_count}/{total_queries} 个查询")
                        logger.error(f"检查点已保存到: {checkpoint_file}")
                        logger.error(f"可以稍后使用 resume=True 从检查点恢复")
                        
                        return  # 直接结束评估
                    else:
                        # 其他类型的错误，继续处理下一个查询
                        logger.warning(f"查询处理失败，继续处理下一个查询: {e}")
                        continue
            
            # Save intermediate results after each batch
            batch_output_file = f"{output_file}.batch_{batch_start//batch_size + 1}"
            self.save_results(self.results, batch_output_file)
            logger.info(f"Batch {batch_start//batch_size + 1} completed, results saved to {batch_output_file}")
            
            # 保存检查点和进度报告
            self.save_checkpoint(self.results, checkpoint_file, processed_count, total_queries)
            self.save_progress_report(output_file, processed_count, total_queries)
            self.print_progress(processed_count, total_queries, self.start_time)
        
        # 5. Save final results
        self.save_results(self.results, output_file)
        
        # 6. 删除检查点文件（处理完成）
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info("Checkpoint file removed after successful completion")
        
        # 7. Generate summary report
        self.generate_summary_report()
        
        # 8. 最终进度报告
        total_time = time.time() - self.start_time
        logger.info(f"=== 评估完成 ===")
        logger.info(f"总查询数: {total_queries}")
        logger.info(f"成功处理: {self.success_count}")
        logger.info(f"处理失败: {self.error_count}")
        logger.info(f"总用时: {total_time/60:.1f}分钟")
        logger.info(f"平均每查询用时: {total_time/total_queries:.1f}秒")
    
    def generate_summary_report(self):
        """生成详细的评估摘要报告"""
        if not self.results:
            return
        
        total_queries = len(self.results)
        
        # 计算全局评分
        global_scores = self.calculate_global_scores(self.results)
        
        # 收集详细评分信息
        query_details = []
        
        for i, result in enumerate(self.results):
            comparison = result['comparison']
            overall_scores = comparison.get('overall_scores', {})
            tool_calls_comparison = comparison.get('tool_calls_comparison', {})
            
            tool_responses_avg = overall_scores.get('tool_responses_avg', 0.0)
            tool_calls_name_avg = tool_calls_comparison.get('name_average_score', 0.0)
            tool_calls_arguments_avg = tool_calls_comparison.get('arguments_average_score', 0.0)
            non_retrieval_name_avg = tool_calls_comparison.get('non_retrieval_name_average_score', 0.0)
            non_retrieval_arguments_avg = tool_calls_comparison.get('non_retrieval_arguments_average_score', 0.0)
            
            query_details.append({
                'index': i,
                'query': result['query'][:50] + '...' if len(result['query']) > 50 else result['query'],
                'tool_calls_name_score': tool_calls_name_avg,
                'tool_calls_arguments_score': tool_calls_arguments_avg,
                'non_retrieval_name_score': non_retrieval_name_avg,
                'non_retrieval_arguments_score': non_retrieval_arguments_avg,
                'tool_responses_score': tool_responses_avg
            })
        
        # 兼容性统计（完全匹配）
        tool_calls_perfect_matches = sum(1 for r in self.results if r['comparison']['tool_calls_match'])
        tool_responses_perfect_matches = sum(1 for r in self.results if r['comparison']['tool_responses_match'])
        
        # 生成报告
        report = f"""
=== 模型评估详细摘要报告 ===

【整体统计】
总查询数: {total_queries}
工具调用完全匹配数: {tool_calls_perfect_matches} ({tool_calls_perfect_matches/total_queries*100:.1f}%)
工具响应完全匹配数: {tool_responses_perfect_matches} ({tool_responses_perfect_matches/total_queries*100:.1f}%)

【全局平均评分】
工具名称匹配平均分: {global_scores['global_tool_calls_name_avg']:.3f}
工具参数匹配平均分: {global_scores['global_tool_calls_arguments_avg']:.3f}
非retrieval工具名称匹配平均分: {global_scores['global_non_retrieval_name_avg']:.3f}
非retrieval工具参数匹配平均分: {global_scores['global_non_retrieval_arguments_avg']:.3f}
工具响应全局平均分: {global_scores['global_tool_responses_avg']:.3f}

【各查询详细评分】"""
        
        for detail in query_details:
            report += f"""
Query {detail['index']}: {detail['query']}
  - 工具名称评分: {detail['tool_calls_name_score']:.3f}
  - 工具参数评分: {detail['tool_calls_arguments_score']:.3f}
  - 非retrieval工具名称评分: {detail['non_retrieval_name_score']:.3f}
  - 非retrieval工具参数评分: {detail['non_retrieval_arguments_score']:.3f}
  - 工具响应评分: {detail['tool_responses_score']:.3f}"""
        
        report += f"""

【评分说明】
- 工具名称匹配分: 工具名称完全一致为1分，否则为0分
- 工具参数匹配分: 工具参数完全一致为1分，否则为0分
- 非retrieval工具名称匹配分: 排除retrieval_tool后，工具名称完全一致为1分，否则为0分
- 非retrieval工具参数匹配分: 排除retrieval_tool后，工具参数完全一致为1分，否则为0分
- 工具响应评分: 完全一致为1分，否则为0分

详细结果请查看 evaluation_results.json 文件
"""
        
        print(report)
        logger.info("详细摘要报告已生成")

    def test_sse_parsing(self):
        """测试SSE解析功能"""
        test_data_tool_call = """id: 3
event: tool_call.created
data: {"conversation_id": "c_9c5b3617", "message_id": "m_1248", "sequence": 3, "role": "assistant", "timestamp": "2025-09-18T13:01:34.230464Z", "content": "", "tool_call": {"name": "intelligent_route_analysis", "arguments": {"access": "1"}}}
"""
        test_data_tool_response = """
id: 4
event: tool_response.completed
data: {"conversation_id": "c_9c5b3617", "message_id": "m_1248", "sequence": 4, "role": "tool", "timestamp": "2025-09-18T13:01:34.358678Z", "tool_call_id": "tool_2", "result_delta": {"chat_log_id": 1234, "content": "", "markdown": "智能路由分析结果\\n\\n访问链接: [上传派团单](https://testai.compassaihz.com/#/$&!upload \\"成功匹配到对应页面\\")\\n\\n", "result": {"success": true, "url": "https://testai.compassaihz.com/#/$&!upload", "message": "成功匹配到对应页面"}, "ambulance": "", "potential_tools": [{"api": "/intelligent_route_analysis", "api_cn": "页面跳转工具", "queryData": [{"type": "select", "key": "access", "label": "数字访问码", "value": {"options": [{"label": "1", "value": "1"}, {"label": "2", "value": "2"}, {"label": "3", "value": "3"}, {"label": "4", "value": "4"}, {"label": "5", "value": "5"}, {"label": "6", "value": "6"}, {"label": "7", "value": "7"}, {"label": "8", "value": "8"}, {"label": "9", "value": "9"}, {"label": "10", "value": "10"}]}, "default": "1", "multiple": false}], "description": "智能页面路由工具，通过输入1-10的数字快速跳转到对应的业务功能页面。业务功能包括：上传派团单，新增资源，新增产品，新增协议，新增协议模版，离线上传协议，离线上传价格政策，新增价格政策，新增报表，新增审批流"}], "tool_calling_chain": [{"role": "function", "tool_call": {"name": "intelligent_route_analysis", "arguments": {"access": "1"}}, "tool_response": {"success": true, "url": "https://testai.compassaihz.com/#/$&!upload", "message": "成功匹配到对应页面"}}], "api_Info": {"api": "/intelligent_route_analysis", "api_cn": "页面跳转工具", "queryData": [{"type": "select", "key": "access", "label": "数字访问码", "value": {"options": [{"label": "1", "value": "1"}, {"label": "2", "value": "2"}, {"label": "3", "value": "3"}, {"label": "4", "value": "4"}, {"label": "5", "value": "5"}, {"label": "6", "value": "6"}, {"label": "7", "value": "7"}, {"label": "8", "value": "8"}, {"label": "9", "value": "9"}, {"label": "10", "value": "10"}]}, "default": "1", "multiple": false}], "description": "智能页面路由工具，通过输入1-10的数字快速跳转到对应的业务功能页面。业务功能包括：上传派团单，新增资源，新增产品，新增协议，新增协议模版，离线上传协议，离线上传价格政策，新增价格政策，新增报表，新增审批流"}}, "success": true, "execution_time": 0.0}

"""
        
        logger.info("=== 开始测试SSE解析功能 ===")
        
        # 合并测试数据
        combined_test_data = test_data_tool_call + test_data_tool_response
        
        # 使用封装的方法解析SSE内容
        events = self.parse_sse_events(
            combined_test_data, 
            filter_events=['tool_call.created', 'tool_response.completed']
        )
        
        logger.info(f"=== 测试解析结果总结 ===")
        logger.info(f"总共解析到 {len(events)} 个事件")
        for event in events:
            logger.info(f"事件: {event['event']}, ID: {event['id']}")
            logger.info(f"   数据摘要: {str(event['data'])[:100]}...")
        
        # 测试提取功能
        extracted = self.extract_tool_calls_and_observations(events)
        logger.info(f"Extraction results from one tool calling: {extracted}")
        
        return events

async def main():
    """Main function"""
    evaluator = ModelEvaluator()
    
    # 首先测试SSE解析功能
    # logger.info("Test SSE parsing function...")
    # evaluator.test_sse_parsing()
    
    # Use the JSON file in the current directory
    input_file = "data/9.17_evaluate_data_top5_final.json"
    output_file = "eval_results/evaluation_results.json"
    
    # 使用新的参数进行评估
    # batch_size: 每批处理50个查询
    # max_queries: 可以限制处理的查询数量（用于测试）
    # resume: 支持断点续传
    await evaluator.evaluate(
        input_file=input_file, 
        output_file=output_file,
        batch_size=50,  # 每批50个查询
        max_queries=None,  # 处理所有查询，可以设置为较小数字进行测试
        checkpoint_file="eval_results/evaluation_results.json.checkpoint",
        resume=True  # 支持断点续传
    )

if __name__ == "__main__":
    asyncio.run(main())
