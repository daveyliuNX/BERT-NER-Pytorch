#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一致性测试脚本：快速验证ONNX和PyTorch模型推理结果的一致性
"""

import os
import sys

def test_consistency():
    """测试两种推理方式的一致性"""
    
    # 检查模型文件
    pytorch_model_path = "outputs/cner_output/bert/checkpoint-956"
    onnx_model_path = "bert_span_ner.onnx"
    
    if not os.path.exists(pytorch_model_path):
        print("错误: 未找到PyTorch模型文件")
        return False
    
    if not os.path.exists(onnx_model_path):
        print("警告: 未找到ONNX模型文件，请先运行 1converttoonnx.py")
        return False
    
    print("测试ONNX和PyTorch模型推理一致性...")
    
    # 测试文本
    test_text = "张三在北京大学学习计算机科学"
    
    # 测试PyTorch推理
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("source_model", "2infersourcemodel.py")
        source_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source_module)
        PyTorchBertSpanNER = source_module.PyTorchBertSpanNER
        print("\n=== PyTorch推理 ===")
        pytorch_model = PyTorchBertSpanNER(pytorch_model_path)
        pytorch_entities, pytorch_time = pytorch_model.predict(test_text)
        print(f"推理时间: {pytorch_time*1000:.2f} ms")
        print(f"识别实体: {pytorch_entities}")
        
    except Exception as e:
        print(f"PyTorch推理失败: {e}")
        return False
    
    # 测试ONNX推理
    try:
        spec2 = importlib.util.spec_from_file_location("onnx_model", "2inferonnxnermodel.py") 
        onnx_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(onnx_module)
        ONNXBertSpanNER = onnx_module.ONNXBertSpanNER
        print("\n=== ONNX推理 ===")
        onnx_model = ONNXBertSpanNER(onnx_model_path, "vocab.txt", "label_mapping.json")
        onnx_entities, onnx_time = onnx_model.predict(test_text)
        print(f"推理时间: {onnx_time*1000:.2f} ms")
        print(f"识别实体: {onnx_entities}")
        
    except Exception as e:
        print(f"ONNX推理失败: {e}")
        return False
    
    # 比较结果
    print("\n=== 结果对比 ===")
    
    # 提取实体的核心信息用于比较
    def extract_key_info(entities):
        return [(e['text'], e['label'], e['start'], e['end']) for e in entities]
    
    pytorch_key = extract_key_info(pytorch_entities)
    onnx_key = extract_key_info(onnx_entities)
    
    print(f"PyTorch结果: {pytorch_key}")
    print(f"ONNX结果: {onnx_key}")
    
    if pytorch_key == onnx_key:
        print("✅ 结果完全一致！")
        speedup = pytorch_time / onnx_time if onnx_time > 0 else 0
        print(f"ONNX加速比: {speedup:.2f}x")
        return True
    else:
        print("❌ 结果不一致，请检查实现")
        
        # 详细分析差异
        pytorch_set = set(pytorch_key)
        onnx_set = set(onnx_key)
        
        only_pytorch = pytorch_set - onnx_set
        only_onnx = onnx_set - pytorch_set
        
        if only_pytorch:
            print(f"仅PyTorch识别: {only_pytorch}")
        if only_onnx:
            print(f"仅ONNX识别: {only_onnx}")
            
        return False

if __name__ == "__main__":
    success = test_consistency()
    sys.exit(0 if success else 1) 