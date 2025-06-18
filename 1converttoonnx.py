#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型转换脚本：将训练好的BERT span NER模型转换为ONNX格式
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertConfig
from models.bert_for_ner import BertSpanForNer
import os

def convert_model_to_onnx():
    """将PyTorch模型转换为ONNX格式"""
    
    # 配置路径
    model_path = "outputs/cner_output/bert/checkpoint-956"
    onnx_path = "bert_span_ner.onnx"
    
    print("正在加载模型...")
    
    # 加载配置和分词器
    config = BertConfig.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # 加载模型
    model = BertSpanForNer.from_pretrained(model_path, config=config)
    model.eval()
    
    print(f"模型配置: num_labels={config.num_labels}, soft_label={config.soft_label}")
    
    # 创建示例输入 (batch_size=1, seq_len=128)
    batch_size = 1
    seq_len = 128
    
    dummy_input_ids = torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    dummy_token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    # 定义输入 (按照模型forward函数的参数顺序)
    dummy_inputs = (
        dummy_input_ids,      # input_ids
        dummy_token_type_ids, # token_type_ids  
        dummy_attention_mask  # attention_mask
    )
    
    print("正在转换为ONNX格式...")
    
    # 转换为ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,                          # 模型
            dummy_inputs,                   # 模型输入
            onnx_path,                      # 输出文件路径
            export_params=True,             # 存储训练好的参数
            opset_version=11,               # ONNX版本
            do_constant_folding=True,       # 是否执行常量折叠优化
            input_names=['input_ids', 'token_type_ids', 'attention_mask'],
            output_names=['start_logits', 'end_logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'token_type_ids': {0: 'batch_size', 1: 'sequence'},
                'start_logits': {0: 'batch_size', 1: 'sequence'},
                'end_logits': {0: 'batch_size', 1: 'sequence'}
            },
            verbose=True
        )
    
    print(f"模型已成功转换为ONNX格式: {onnx_path}")
    
    # 验证ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型验证通过!")
        
        # 显示模型信息
        print(f"模型大小: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
        
    except ImportError:
        print("未安装onnx包，跳过验证")
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")
    
    # 保存标签映射和分词器信息
    label_info = {
        "id2label": {int(k): v for k, v in config.id2label.items()},
        "label2id": config.label2id,
        "num_labels": config.num_labels
    }
    
    import json
    with open("label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)
    
    # 保存分词器
    tokenizer.save_vocabulary("./")
    
    print("标签映射和分词器信息已保存")
    
def test_pytorch_model():
    """测试原始PyTorch模型的推理"""
    print("\n正在测试PyTorch模型...")
    
    model_path = "outputs/cner_output/bert/checkpoint-956"
    
    # 加载模型
    config = BertConfig.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertSpanForNer.from_pretrained(model_path, config=config)
    model.eval()
    
    # 测试文本
    text = "张三在北京工作"
    print(f"测试文本: {text}")
    
    # 分词
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # 推理
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded["token_type_ids"]
        )
        start_logits, end_logits = outputs[:2]
    
    print(f"Start logits shape: {start_logits.shape}")
    print(f"End logits shape: {end_logits.shape}")
    
    # 提取实体
    from processors.utils_ner import bert_extract_item
    entities = bert_extract_item(start_logits, end_logits)
    print(f"提取的实体: {entities}")

if __name__ == "__main__":
    # 测试PyTorch模型
    test_pytorch_model()
    
    # 转换为ONNX
    convert_model_to_onnx()
    
    print("\n转换完成!") 