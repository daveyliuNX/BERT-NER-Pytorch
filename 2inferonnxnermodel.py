#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ONNX推理脚本：使用ONNX Runtime进行BERT span NER推理
"""

import onnxruntime as ort
import numpy as np
import time
import json
from transformers import BertTokenizer
from typing import List, Tuple, Dict
import os

class ONNXBertSpanNER:
    """ONNX格式的BERT span NER推理器"""
    
    def __init__(self, onnx_model_path: str, vocab_path: str = None, label_mapping_path: str = None):
        """
        初始化ONNX推理器
        
        Args:
            onnx_model_path: ONNX模型文件路径
            vocab_path: 词汇表文件路径
            label_mapping_path: 标签映射文件路径
        """
        print("正在加载ONNX模型...")
        
        # 设置ONNX Runtime会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 创建推理会话
        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')
            
        self.session = ort.InferenceSession(onnx_model_path, sess_options, providers=providers)
        
        # 获取输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"输入名称: {self.input_names}")
        print(f"输出名称: {self.output_names}")
        print(f"使用的执行提供程序: {self.session.get_providers()}")
        
        # 加载分词器
        if vocab_path and os.path.exists(vocab_path):
            self.tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
        else:
            # 如果没有指定vocab文件，尝试从当前目录加载
            if os.path.exists("vocab.txt"):
                self.tokenizer = BertTokenizer(vocab_file="vocab.txt", do_lower_case=True)
            else:
                print("警告: 未找到vocab.txt，将使用预训练分词器")
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        
        # 加载标签映射
        if label_mapping_path and os.path.exists(label_mapping_path):
            with open(label_mapping_path, "r", encoding="utf-8") as f:
                label_info = json.load(f)
                self.id2label = {int(k): v for k, v in label_info["id2label"].items()}
                self.label2id = label_info["label2id"]
                self.num_labels = label_info["num_labels"]
        else:
            # 默认标签映射（根据你的训练配置）
            self.id2label = {0: 'O', 1: 'CONT', 2: 'ORG', 3: 'LOC', 4: 'EDU', 5: 'NAME', 6: 'PRO', 7: 'RACE', 8: 'TITLE'}
            self.label2id = {v: k for k, v in self.id2label.items()}
            self.num_labels = len(self.id2label)
        
        print(f"标签映射: {self.id2label}")
        
    def preprocess(self, text: str, max_length: int = 128) -> Dict[str, np.ndarray]:
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
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # 准备ONNX输入 (按照导出时的顺序: input_ids, token_type_ids, attention_mask)
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "token_type_ids": encoded["token_type_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64)
        }
        
        return inputs, encoded
    
    def extract_entities(self, start_logits: np.ndarray, end_logits: np.ndarray) -> List[Tuple]:
        """
        从logits中提取实体，使用与原始模型相同的逻辑
        
        Args:
            start_logits: 起始位置logits
            end_logits: 结束位置logits
            
        Returns:
            实体列表 [(实体类型ID, 起始位置, 结束位置), ...]
        """
        # 转换为torch tensor以使用原始的bert_extract_item函数
        import torch
        start_logits_tensor = torch.from_numpy(start_logits)
        end_logits_tensor = torch.from_numpy(end_logits)
        
        # 使用原始的实体提取函数
        S = []
        start_pred = torch.argmax(start_logits_tensor, -1).cpu().numpy()[0][1:-1]
        end_pred = torch.argmax(end_logits_tensor, -1).cpu().numpy()[0][1:-1]
        
        for i, s_l in enumerate(start_pred):
            if s_l == 0:
                continue
            for j, e_l in enumerate(end_pred[i:]):
                if s_l == e_l:
                    S.append((s_l, i, i + j))
                    break
        return S
    
    def postprocess(self, entities: List[Tuple], tokens: List[str], text: str) -> List[Dict]:
        """
        后处理，将实体位置映射回原文本，与原始脚本保持一致
        
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
                    "confidence": 1.0  # ONNX推理暂时不计算置信度
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
        
        # ONNX推理
        outputs = self.session.run(self.output_names, inputs)
        
        inference_time = time.time() - start_time
        
        # 获取logits
        start_logits, end_logits = outputs[0], outputs[1]
        
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

def benchmark_inference(model: ONNXBertSpanNER, texts: List[str], num_runs: int = 100):
    """
    推理性能基准测试
    
    Args:
        model: ONNX模型推理器
        texts: 测试文本列表
        num_runs: 运行次数
    """
    print(f"\n开始性能基准测试 (运行 {num_runs} 次)...")
    
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

def main():
    """主函数"""
    # 初始化ONNX推理器
    onnx_model_path = "bert_span_ner.onnx"
    vocab_path = "vocab.txt"
    label_mapping_path = "label_mapping.json"
    
    if not os.path.exists(onnx_model_path):
        print(f"错误: 未找到ONNX模型文件 {onnx_model_path}")
        print("请先运行 1converttoonnx.py 生成ONNX模型")
        return
    
    try:
        ner_model = ONNXBertSpanNER(onnx_model_path, vocab_path, label_mapping_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 测试文本
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
    
    # 保存推理结果用于对比
    print(f"\n保存ONNX推理结果用于对比...")
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
        with open("onnx_inference_results.json", "w", encoding="utf-8") as f:
            json.dump(results_for_comparison, f, ensure_ascii=False, indent=2)
        
        print("ONNX推理结果已保存到 onnx_inference_results.json")
        
    except Exception as e:
        print(f"保存结果失败: {e}")

if __name__ == "__main__":
    main() 