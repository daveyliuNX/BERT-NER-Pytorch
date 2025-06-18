# BERT span NER 模型 ONNX 转换和推理指南

## 概述

本项目提供了将训练好的 BERT span NER 模型转换为 ONNX 格式，并使用 ONNX Runtime 进行高效推理的完整解决方案。

## 依赖安装

在运行脚本之前，请安装必要的依赖包：

```bash
# 安装 ONNX 相关包
pip install onnx onnxruntime

# 如果你有 GPU 并想使用 GPU 推理，安装:
pip install onnxruntime-gpu

# 确保其他依赖已安装
pip install torch transformers numpy
```

## 文件说明

- `1converttoonnx.py`: 模型转换脚本，将 PyTorch 模型转换为 ONNX 格式
- `2inferonnxnermodel.py`: ONNX 推理脚本，支持单句和批量推理，包含性能测试
- `README_ONNX.md`: 本说明文档

## 使用步骤

### 1. 转换模型为 ONNX 格式

```bash
python 1converttoonnx.py
```

这个脚本会：
- 加载训练好的 PyTorch 模型 (`outputs/cner_output/bert/checkpoint-956`)
- 测试 PyTorch 模型的推理功能
- 将模型转换为 ONNX 格式 (`bert_span_ner.onnx`)
- 保存标签映射文件 (`label_mapping.json`)
- 保存词汇表文件 (`vocab.txt`)
- 验证 ONNX 模型的正确性

### 2. 使用 ONNX 模型进行推理

```bash
python 2inferonnxnermodel.py
```

这个脚本会：
- 加载 ONNX 模型和相关配置文件
- 对测试文本进行单句推理
- 进行批量推理测试
- 执行性能基准测试
- 输出详细的推理时间统计

## 输出示例

### 转换过程输出
```
正在测试PyTorch模型...
测试文本: 张三在北京工作
Start logits shape: torch.Size([1, 128, 9])
End logits shape: torch.Size([1, 128, 9])
提取的实体: [(5, 0, 1), (3, 3, 4)]

正在加载模型...
模型配置: num_labels=9, soft_label=True
正在转换为ONNX格式...
模型已成功转换为ONNX格式: bert_span_ner.onnx
ONNX模型验证通过!
模型大小: 392.45 MB
标签映射和分词器信息已保存
```

### 推理过程输出
```
正在加载ONNX模型...
输入名称: ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']
输出名称: ['start_logits', 'end_logits']
使用的执行提供程序: ['CPUExecutionProvider']

开始单句推理测试...
==================================================

输入文本: 张三在北京大学学习计算机科学
推理时间: 125.45 ms
识别实体:
  - 张三 [NAME] (位置: 0-1)
  - 北京大学 [EDU] (位置: 2-5)

性能统计:
  平均推理时间: 120.35 ms
  最短推理时间: 115.22 ms
  最长推理时间: 130.18 ms
  标准差: 8.45 ms
  吞吐量: 8.31 samples/sec
```

## 标签映射

模型支持以下实体类型：
- `O`: 非实体
- `CONT`: 联系方式
- `ORG`: 组织机构
- `LOC`: 地点
- `EDU`: 教育机构
- `NAME`: 人名
- `PRO`: 专业
- `RACE`: 种族
- `TITLE`: 称谓

## 性能优化建议

1. **CPU 推理优化**：
   - 使用 `CPUExecutionProvider`
   - 设置合适的线程数：`ort.SessionOptions().intra_op_num_threads`

2. **GPU 推理优化**：
   - 安装 `onnxruntime-gpu`
   - 使用 `CUDAExecutionProvider`

3. **批量推理**：
   - 对于大量文本，使用批量推理可以提高吞吐量
   - 根据内存限制调整批次大小

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   错误: 未找到ONNX模型文件 bert_span_ner.onnx
   ```
   解决：先运行 `1converttoonnx.py` 生成 ONNX 模型

2. **ONNX Runtime 未安装**
   ```
   ModuleNotFoundError: No module named 'onnxruntime'
   ```
   解决：`pip install onnxruntime`

3. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：减少批处理大小或使用 CPU 推理

4. **分词器问题**
   ```
   警告: 未找到vocab.txt，将使用预训练分词器
   ```
   解决：确保 `vocab.txt` 文件在正确位置，或者让程序使用默认分词器

## 自定义使用

可以在自己的代码中使用 ONNX 模型：

```python
from 2inferonnxnermodel import ONNXBertSpanNER

# 初始化模型
ner_model = ONNXBertSpanNER("bert_span_ner.onnx", "vocab.txt", "label_mapping.json")

# 单句推理
text = "你的测试文本"
entities, inference_time = ner_model.predict(text)

print(f"推理时间: {inference_time*1000:.2f} ms")
for entity in entities:
    print(f"{entity['text']} [{entity['label']}]")
```

## 注意事项

1. 转换后的 ONNX 模型文件较大（约 390MB），请确保有足够的存储空间
2. 首次推理可能较慢，后续推理会更快
3. ONNX 模型的精度与原始 PyTorch 模型基本一致
4. 推理时间会因硬件配置而异

## 支持的输入格式

- 文本长度：最大 128 个 token（可调整）
- 编码：UTF-8
- 语言：中文（基于 bert-base-chinese）

通过这套完整的 ONNX 解决方案，你可以在生产环境中高效部署 BERT span NER 模型。 