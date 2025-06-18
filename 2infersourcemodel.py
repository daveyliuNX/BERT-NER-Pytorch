#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原始模型推理脚本：使用PyTorch加载原始BERT span NER模型进行推理
"""

import torch
import numpy as np
import time
import json
from transformers import BertTokenizer, BertConfig
from models.bert_for_ner import BertSpanForNer
from processors.utils_ner import bert_extract_item
from typing import List, Tuple, Dict
import os

class PyTorchBertSpanNER:
    """PyTorch格式的BERT span NER推理器"""
    
    def __init__(self, model_path: str):
        """
        初始化PyTorch推理器
        
        Args:
            model_path: 模型文件路径
        """
        print("正在加载PyTorch模型...")
        
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置和分词器
        self.config = BertConfig.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 加载模型
        self.model = BertSpanForNer.from_pretrained(model_path, config=self.config)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成，使用设备: {self.device}")
        print(f"模型配置: num_labels={self.config.num_labels}, soft_label={self.config.soft_label}")
        
        # 设置标签映射
        if hasattr(self.config, 'id2label') and self.config.id2label:
            self.id2label = {int(k): v for k, v in self.config.id2label.items()}
            self.label2id = {v: int(k) for k, v in self.config.id2label.items()}
        else:
            # 默认标签映射
            self.id2label = {0: 'O', 1: 'CONT', 2: 'ORG', 3: 'LOC', 4: 'EDU', 5: 'NAME', 6: 'PRO', 7: 'RACE', 8: 'TITLE'}
            self.label2id = {v: k for k, v in self.id2label.items()}
        
        self.num_labels = len(self.id2label)
        print(f"标签映射: {self.id2label}")
        
    def preprocess(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        预处理输入文本
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            
        Returns:
            模型输入字典
        """
        # 分词
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # 移动到相应设备
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        
        return inputs, encoded
    
    def extract_entities(self, start_logits: torch.Tensor, end_logits: torch.Tensor) -> List[Tuple]:
        """
        从logits中提取实体
        
        Args:
            start_logits: 起始位置logits
            end_logits: 结束位置logits
            
        Returns:
            实体列表 [(实体类型ID, 起始位置, 结束位置), ...]
        """
        # 使用原始的bert_extract_item函数
        entities = bert_extract_item(start_logits, end_logits)
        return entities
    
    def postprocess(self, entities: List[Tuple], tokens: List[str], text: str) -> List[Dict]:
        """
        后处理，将实体位置映射回原文本
        
        Args:
            entities: 实体列表
            tokens: 分词结果
            text: 原始文本
            
        Returns:
            实体信息列表
        """
        results = []
        
        for entity_type_id, start_idx, end_idx in entities:
            entity_type = self.id2label.get(entity_type_id, f"UNKNOWN_{entity_type_id}")
            
            # 获取实体tokens (注意start_idx和end_idx已经是相对于去掉[CLS]和[SEP]后的位置)
            # 需要调整索引以包含[CLS]
            actual_start_idx = start_idx + 1  # +1 因为[CLS]
            actual_end_idx = end_idx + 1      # +1 因为[CLS]
            
            if actual_end_idx < len(tokens) - 1:  # 确保不超出范围
                entity_tokens = tokens[actual_start_idx:actual_end_idx+1]
                
                # 去除特殊token并合并
                entity_text = "".join([token.replace("##", "") for token in entity_tokens if not token.startswith("[") and not token.endswith("]")])
                
                results.append({
                    "text": entity_text,
                    "label": entity_type,
                    "start": start_idx,
                    "end": end_idx,
                    "confidence": 1.0  # PyTorch推理暂时不计算置信度
                })
            
        return results
    
    def predict(self, text: str, max_length: int = 128) -> Tuple[List[Dict], float]:
        """
        预测文本中的命名实体
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            
        Returns:
            (实体列表, 推理时间)
        """
        # 预处理
        inputs, encoded = self.preprocess(text, max_length)
        
        # 推理计时
        start_time = time.time()
        
        # PyTorch推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits, end_logits = outputs[:2]
        
        inference_time = time.time() - start_time
        
        # 提取实体
        entities = self.extract_entities(start_logits, end_logits)
        
        # 获取tokens用于后处理
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        
        # 后处理
        results = self.postprocess(entities, tokens, text)
        
        return results, inference_time
    
    def batch_predict(self, texts: List[str], max_length: int = 128) -> Tuple[List[List[Dict]], float]:
        """
        批量预测
        
        Args:
            texts: 文本列表
            max_length: 最大序列长度
            
        Returns:
            (批量实体列表, 总推理时间)
        """
        all_results = []
        total_time = 0
        
        for text in texts:
            results, inference_time = self.predict(text, max_length)
            all_results.append(results)
            total_time += inference_time
            
        return all_results, total_time
    
    def warmup(self, num_warmup: int = 5):
        """
        模型预热
        
        Args:
            num_warmup: 预热次数
        """
        print(f"正在进行模型预热 ({num_warmup} 次)...")
        warmup_text = "这是一个预热测试文本"
        
        for i in range(num_warmup):
            _, _ = self.predict(warmup_text)
            
        print("模型预热完成")

def benchmark_inference(model: PyTorchBertSpanNER, texts: List[str], num_runs: int = 100):
    """
    推理性能基准测试
    
    Args:
        model: PyTorch模型推理器
        texts: 测试文本列表
        num_runs: 运行次数
    """
    print(f"\n开始性能基准测试 (运行 {num_runs} 次)...")
    
    # 预热
    model.warmup(5)
    
    all_times = []
    
    for i in range(num_runs):
        for text in texts:
            _, inference_time = model.predict(text)
            all_times.append(inference_time)
    
    avg_time = np.mean(all_times)
    min_time = np.min(all_times)
    max_time = np.max(all_times)
    std_time = np.std(all_times)
    
    print(f"性能统计:")
    print(f"  平均推理时间: {avg_time*1000:.2f} ms")
    print(f"  最短推理时间: {min_time*1000:.2f} ms") 
    print(f"  最长推理时间: {max_time*1000:.2f} ms")
    print(f"  标准差: {std_time*1000:.2f} ms")
    print(f"  吞吐量: {1/avg_time:.2f} samples/sec")

def compare_with_onnx_results():
    """
    与ONNX推理结果进行比较
    """
    print("\n" + "="*60)
    print("PyTorch vs ONNX 推理结果对比")
    print("="*60)
    
    # 这里可以加载ONNX推理结果进行对比
    # 暂时只是提示信息
    print("提示: 运行完成后可以与ONNX推理结果进行对比")
    print("- 检查实体识别的准确性")
    print("- 比较推理速度")
    print("- 验证结果一致性")

def main():
    """主函数"""
    # 初始化PyTorch推理器
    model_path = "outputs/cner_output/bert/checkpoint-956"
    
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        print("请确认模型路径是否正确")
        return
    
    try:
        ner_model = PyTorchBertSpanNER(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 测试文本 (与ONNX推理脚本保持一致)
    test_texts = [
        "张三在北京大学学习计算机科学，中共党员",
        "李四是上海复旦大学的教授，中国国籍",
        "王五在深圳腾讯公司工作",
        "马云创建了阿里巴巴集团",
        "华为公司总部位于深圳市南山区"
    ]
    
    print("\n开始单句推理测试...")
    print("=" * 50)
    
    # 单句测试
    for text in test_texts:
        print(f"\n输入文本: {text}")
        
        try:
            entities, inference_time = ner_model.predict(text)
            print(f"推理时间: {inference_time*1000:.2f} ms")
            print(f"识别实体:")
            
            if entities:
                for entity in entities:
                    print(f"  - {entity['text']} [{entity['label']}] (位置: {entity['start']}-{entity['end']})")
            else:
                print("  - 未识别到实体")
                
        except Exception as e:
            print(f"推理失败: {e}")
    
    # 批量测试  
    print(f"\n开始批量推理测试...")
    print("=" * 50)
    
    try:
        batch_results, total_time = ner_model.batch_predict(test_texts)
        print(f"批量推理总时间: {total_time*1000:.2f} ms")
        print(f"平均每句推理时间: {total_time/len(test_texts)*1000:.2f} ms")
        
        for i, (text, entities) in enumerate(zip(test_texts, batch_results)):
            print(f"\n句子 {i+1}: {text}")
            if entities:
                for entity in entities:
                    print(f"  - {entity['text']} [{entity['label']}]")
            else:
                print("  - 未识别到实体")
                
    except Exception as e:
        print(f"批量推理失败: {e}")
    
    # 性能基准测试
    try:
        benchmark_inference(ner_model, test_texts[:2], num_runs=50)  # 使用前两句进行基准测试
    except Exception as e:
        print(f"基准测试失败: {e}")
    
    # 提供对比信息
    compare_with_onnx_results()
    
    # 保存推理结果用于对比
    print(f"\n保存推理结果用于对比...")
    try:
        results_for_comparison = []
        for text in test_texts:
            entities, inference_time = ner_model.predict(text)
            results_for_comparison.append({
                "text": text,
                "entities": entities,
                "inference_time_ms": inference_time * 1000
            })
        
        # 保存结果
        with open("pytorch_inference_results.json", "w", encoding="utf-8") as f:
            json.dump(results_for_comparison, f, ensure_ascii=False, indent=2)
        
        print("PyTorch推理结果已保存到 pytorch_inference_results.json")
        
    except Exception as e:
        print(f"保存结果失败: {e}")

if __name__ == "__main__":
    main() 