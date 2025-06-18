#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """下载文件并显示进度条"""
    print(f"正在下载: {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

def main():
    # 创建目录
    model_dir = "prev_trained_model/bert-base-chinese"
    os.makedirs(model_dir, exist_ok=True)
    
    # BERT中文模型文件URL（使用镜像源）
    base_url = "https://hf-mirror.com/bert-base-chinese/resolve/main"

    
    files_to_download = [
        ("config.json", f"{base_url}/config.json"),
        ("pytorch_model.bin", f"{base_url}/pytorch_model.bin"),
        ("vocab.txt", f"{base_url}/vocab.txt"),
        ("tokenizer_config.json", f"{base_url}/tokenizer_config.json"),
        ("tokenizer.json", f"{base_url}/tokenizer.json")
    ]
    
    print("开始下载BERT中文预训练模型...")
    print(f"保存路径: {os.path.abspath(model_dir)}")
    
    for filename, url in files_to_download:
        filepath = os.path.join(model_dir, filename)
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            print(f"文件已存在，跳过: {filename}")
            continue
            
        try:
            download_file(url, filepath)
            print(f"✓ 下载完成: {filename}")
        except Exception as e:
            print(f"✗ 下载失败: {filename}")
            print(f"错误信息: {e}")
            
            # 如果是网络问题，提供备用方案
            if "huggingface.co" in str(e):
                print("提示: 如果网络访问huggingface.co有问题，你可以:")
                print("1. 使用VPN或代理")
                print("2. 手动从百度网盘等国内镜像下载")
                print("3. 使用国内镜像站点")
    
    print("\n下载完成! 请检查以下文件是否都存在:")
    for filename, _ in files_to_download:
        filepath = os.path.join(model_dir, filename)
        status = "✓" if os.path.exists(filepath) else "✗"
        print(f"{status} {filepath}")

if __name__ == "__main__":
    main() 