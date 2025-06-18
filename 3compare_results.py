#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果对比脚本：比较PyTorch和ONNX推理结果的一致性和性能
"""

import json
import os
from typing import Dict, List
import numpy as np

def load_results(file_path: str) -> List[Dict]:
    """
    从JSON文件加载推理结果
    
    Args:
        file_path: 结果文件路径
        
    Returns:
        推理结果列表
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return []

def compare_entities(pytorch_entities: List[Dict], onnx_entities: List[Dict]) -> Dict:
    """
    比较两组实体识别结果
    
    Args:
        pytorch_entities: PyTorch识别的实体
        onnx_entities: ONNX识别的实体
        
    Returns:
        比较统计结果
    """
    # 将实体转换为可比较的格式 (text, label, start, end)
    def entities_to_set(entities):
        return set((e['text'], e['label'], e['start'], e['end']) for e in entities)
    
    pytorch_set = entities_to_set(pytorch_entities)
    onnx_set = entities_to_set(onnx_entities)
    
    # 计算重叠情况
    common = pytorch_set & onnx_set
    pytorch_only = pytorch_set - onnx_set
    onnx_only = onnx_set - pytorch_set
    
    return {
        'total_pytorch': len(pytorch_set),
        'total_onnx': len(onnx_set),
        'common': len(common),
        'pytorch_only': len(pytorch_only),
        'onnx_only': len(onnx_only),
        'pytorch_only_entities': list(pytorch_only),
        'onnx_only_entities': list(onnx_only),
        'common_entities': list(common)
    }

def compare_performance(pytorch_results: List[Dict], onnx_results: List[Dict]) -> Dict:
    """
    比较性能数据
    
    Args:
        pytorch_results: PyTorch推理结果
        onnx_results: ONNX推理结果
        
    Returns:
        性能比较结果
    """
    # 提取推理时间
    pytorch_times = [r.get('inference_time_ms', 0) for r in pytorch_results]
    onnx_times = [r.get('inference_time_ms', 0) for r in onnx_results]
    
    if not pytorch_times or not onnx_times:
        return {"error": "缺少推理时间数据"}
    
    # 计算统计数据
    pytorch_stats = {
        'mean': np.mean(pytorch_times),
        'std': np.std(pytorch_times),
        'min': np.min(pytorch_times),
        'max': np.max(pytorch_times)
    }
    
    onnx_stats = {
        'mean': np.mean(onnx_times),
        'std': np.std(onnx_times),
        'min': np.min(onnx_times),
        'max': np.max(onnx_times)
    }
    
    # 计算加速比
    speedup = pytorch_stats['mean'] / onnx_stats['mean'] if onnx_stats['mean'] > 0 else 0
    
    return {
        'pytorch_stats': pytorch_stats,
        'onnx_stats': onnx_stats,
        'speedup': speedup,
        'pytorch_times': pytorch_times,
        'onnx_times': onnx_times
    }

def calculate_accuracy_metrics(comparisons: List[Dict]) -> Dict:
    """
    计算准确性指标
    
    Args:
        comparisons: 每个句子的比较结果
        
    Returns:
        准确性指标
    """
    total_pytorch = sum(c['total_pytorch'] for c in comparisons)
    total_onnx = sum(c['total_onnx'] for c in comparisons)
    total_common = sum(c['common'] for c in comparisons)
    
    # 计算精确率、召回率、F1
    precision = total_common / total_onnx if total_onnx > 0 else 0
    recall = total_common / total_pytorch if total_pytorch > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 完全匹配的句子数
    exact_matches = sum(1 for c in comparisons if c['total_pytorch'] == c['total_onnx'] == c['common'])
    total_sentences = len(comparisons)
    exact_match_rate = exact_matches / total_sentences if total_sentences > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'exact_match_rate': exact_match_rate,
        'exact_matches': exact_matches,
        'total_sentences': total_sentences,
        'total_pytorch_entities': total_pytorch,
        'total_onnx_entities': total_onnx,
        'total_common_entities': total_common
    }

def print_detailed_comparison(pytorch_results: List[Dict], onnx_results: List[Dict]):
    """
    打印详细的比较结果
    
    Args:
        pytorch_results: PyTorch推理结果
        onnx_results: ONNX推理结果
    """
    print("\n" + "="*80)
    print("详细推理结果对比")
    print("="*80)
    
    comparisons = []
    
    for i, (pytorch_result, onnx_result) in enumerate(zip(pytorch_results, onnx_results)):
        text = pytorch_result.get('text', f'句子{i+1}')
        pytorch_entities = pytorch_result.get('entities', [])
        onnx_entities = onnx_result.get('entities', [])
        
        comparison = compare_entities(pytorch_entities, onnx_entities)
        comparisons.append(comparison)
        
        print(f"\n句子 {i+1}: {text}")
        print(f"PyTorch推理时间: {pytorch_result.get('inference_time_ms', 0):.2f} ms")
        print(f"ONNX推理时间: {onnx_result.get('inference_time_ms', 0):.2f} ms")
        
        print(f"\nPyTorch识别实体 ({comparison['total_pytorch']}个):")
        for entity in pytorch_entities:
            print(f"  - {entity['text']} [{entity['label']}] ({entity['start']}-{entity['end']})")
        
        print(f"\nONNX识别实体 ({comparison['total_onnx']}个):")
        for entity in onnx_entities:
            print(f"  - {entity['text']} [{entity['label']}] ({entity['start']}-{entity['end']})")
        
        if comparison['common'] == comparison['total_pytorch'] == comparison['total_onnx']:
            print("✅ 结果完全一致")
        else:
            print(f"⚠️  结果不完全一致 - 共同: {comparison['common']}, 仅PyTorch: {comparison['pytorch_only']}, 仅ONNX: {comparison['onnx_only']}")
            
            if comparison['pytorch_only_entities']:
                print("  仅PyTorch识别:")
                for entity in comparison['pytorch_only_entities']:
                    print(f"    - {entity}")
            
            if comparison['onnx_only_entities']:
                print("  仅ONNX识别:")
                for entity in comparison['onnx_only_entities']:
                    print(f"    - {entity}")
        
        print("-" * 60)
    
    return comparisons

def main():
    """主函数"""
    print("BERT span NER 推理结果对比分析")
    print("="*80)
    
    # 文件路径
    pytorch_file = "pytorch_inference_results.json"
    onnx_file = "onnx_inference_results.json"  # 假设ONNX脚本也会保存结果
    
    # 加载结果
    pytorch_results = load_results(pytorch_file)
    onnx_results = load_results(onnx_file)
    
    if not pytorch_results:
        print(f"错误: 无法加载PyTorch推理结果")
        print("请先运行 2infersourcemodel.py 生成结果文件")
        return
    
    if not onnx_results:
        print(f"警告: 无法加载ONNX推理结果")
        print("请运行 2inferonnxnermodel.py 并确保保存结果到 onnx_inference_results.json")
        print("目前只显示PyTorch推理结果:")
        
        for i, result in enumerate(pytorch_results):
            print(f"\n句子 {i+1}: {result['text']}")
            print(f"推理时间: {result.get('inference_time_ms', 0):.2f} ms")
            entities = result.get('entities', [])
            if entities:
                for entity in entities:
                    print(f"  - {entity['text']} [{entity['label']}] ({entity['start']}-{entity['end']})")
            else:
                print("  - 未识别到实体")
        return
    
    # 检查结果长度是否一致
    if len(pytorch_results) != len(onnx_results):
        print(f"警告: 结果数量不一致 - PyTorch: {len(pytorch_results)}, ONNX: {len(onnx_results)}")
        min_len = min(len(pytorch_results), len(onnx_results))
        pytorch_results = pytorch_results[:min_len]
        onnx_results = onnx_results[:min_len]
    
    # 详细比较
    comparisons = print_detailed_comparison(pytorch_results, onnx_results)
    
    # 总体统计
    print("\n" + "="*80)
    print("总体统计分析")
    print("="*80)
    
    # 准确性指标
    accuracy_metrics = calculate_accuracy_metrics(comparisons)
    print(f"\n实体识别一致性:")
    print(f"  精确率 (ONNX vs PyTorch): {accuracy_metrics['precision']:.4f}")
    print(f"  召回率 (ONNX vs PyTorch): {accuracy_metrics['recall']:.4f}")
    print(f"  F1分数: {accuracy_metrics['f1_score']:.4f}")
    print(f"  完全匹配率: {accuracy_metrics['exact_match_rate']:.4f} ({accuracy_metrics['exact_matches']}/{accuracy_metrics['total_sentences']})")
    print(f"  总实体数 - PyTorch: {accuracy_metrics['total_pytorch_entities']}, ONNX: {accuracy_metrics['total_onnx_entities']}")
    print(f"  共同识别: {accuracy_metrics['total_common_entities']}")
    
    # 性能比较
    performance = compare_performance(pytorch_results, onnx_results)
    if 'error' not in performance:
        print(f"\n推理性能对比:")
        print(f"  PyTorch平均时间: {performance['pytorch_stats']['mean']:.2f} ms (±{performance['pytorch_stats']['std']:.2f})")
        print(f"  ONNX平均时间: {performance['onnx_stats']['mean']:.2f} ms (±{performance['onnx_stats']['std']:.2f})")
        print(f"  ONNX加速比: {performance['speedup']:.2f}x")
        
        if performance['speedup'] > 1:
            print(f"  🚀 ONNX比PyTorch快 {(performance['speedup']-1)*100:.1f}%")
        elif performance['speedup'] < 1:
            print(f"  🐌 ONNX比PyTorch慢 {(1-performance['speedup'])*100:.1f}%")
        else:
            print(f"  ⚖️ 性能基本相当")
    
    # 保存对比报告
    report = {
        'accuracy_metrics': accuracy_metrics,
        'performance_metrics': performance,
        'detailed_comparisons': comparisons,
        'summary': {
            'total_sentences': len(comparisons),
            'consistent_results': accuracy_metrics['exact_matches'],
            'consistency_rate': accuracy_metrics['exact_match_rate'],
            'average_speedup': performance.get('speedup', 0)
        }
    }
    
    try:
        with open('comparison_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📊 详细对比报告已保存到 comparison_report.json")
    except Exception as e:
        print(f"保存报告失败: {e}")
    
    print("\n" + "="*80)
    print("对比分析完成")
    print("="*80)

if __name__ == "__main__":
    main() 