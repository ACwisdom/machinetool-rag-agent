#!/usr/bin/env python3
"""
模型微调训练脚本
使用Llama Factory进行SFT/LoRA微调
"""

import os
import subprocess
import json

def create_train_config():
    """创建训练配置文件"""
    config = {
        "stage": "sft",
        "do_train": True,
        "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "dataset": "diagnosis_finetune_data.json",
        "template": "qwen",
        "finetuning_type": "lora",
        "output_dir": "./sft_lora_model",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": 1000,
        "learning_rate": 5e-5,
        "num_train_epochs": 3.0,
        "plot_loss": True,
        "quantization_bit": 4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "val_size": 0.1,
        "eval_strategy": "epoch",
        "load_best_model_at_end": True
    }
    
    with open("train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    return config

def run_training(config):
    """运行训练命令"""
    # 构建训练命令
    cmd = [
        "python", "src/train_bash.py",
        "--stage", config["stage"],
        "--do_train", str(config["do_train"]),
        "--model_name_or_path", config["model_name_or_path"],
        "--dataset", config["dataset"],
        "--template", config["template"],
        "--finetuning_type", config["finetuning_type"],
        "--output_dir", config["output_dir"],
        "--per_device_train_batch_size", str(config["per_device_train_batch_size"]),
        "--gradient_accumulation_steps", str(config["gradient_accumulation_steps"]),
        "--lr_scheduler_type", config["lr_scheduler_type"],
        "--logging_steps", str(config["logging_steps"]),
        "--save_steps", str(config["save_steps"]),
        "--learning_rate", str(config["learning_rate"]),
        "--num_train_epochs", str(config["num_train_epochs"]),
        "--plot_loss", str(config["plot_loss"]),
        "--quantization_bit", str(config["quantization_bit"]),
        "--lora_rank", str(config["lora_rank"]),
        "--lora_alpha", str(config["lora_alpha"]),
        "--lora_dropout", str(config["lora_dropout"]),
        "--val_size", str(config["val_size"]),
        "--eval_strategy", config["eval_strategy"],
        "--load_best_model_at_end", str(config["load_best_model_at_end"])
    ]
    
    print("开始训练...")
    print("命令:", " ".join(cmd))
    
    # 执行训练命令
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 保存训练日志
    with open("train_logs.txt", "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)
    
    if result.returncode == 0:
        print("训练完成！")
    else:
        print("训练过程中出现错误，请查看train_logs.txt")
    
    return result.returncode

def merge_lora_weights():
    """合并LoRA权重到基座模型"""
    cmd = [
        "python", "src/export_model.py",
        "--model_name_or_path", "Qwen/Qwen2.5-7B-Instruct",
        "--adapter_name_or_path", "./sft_lora_model",
        "--template", "qwen",
        "--finetuning_type", "lora",
        "--export_dir", "./merged_model",
        "--export_size_limit", "20G"
    ]
    
    print("合并LoRA权重...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    with open("merge_logs.txt", "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)
    
    if result.returncode == 0:
        print("LoRA权重合并完成！")
    else:
        print("合并过程中出现错误，请查看merge_logs.txt")
    
    return result.returncode

def evaluate_model():
    """在测试集上评估模型效果"""
    cmd = [
        "python", "src/evaluate.py",
        "--model_name_or_path", "./merged_model",
        "--template", "qwen",
        "--dataset", "diagnosis_finetune_data.json",
        "--split", "test",
        "--batch_size", 4,
        "--predictions_file", "predictions.json"
    ]
    
    print("评估模型效果...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    with open("eval_logs.txt", "w", encoding="utf-8") as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)
    
    if result.returncode == 0:
        print("模型评估完成！")
    else:
        print("评估过程中出现错误，请查看eval_logs.txt")
    
    return result.returncode

def main():
    # 创建训练配置
    config = create_train_config()
    print("训练配置已保存到train_config.json")
    
    # 运行训练
    train_status = run_training(config)
    if train_status != 0:
        return
    
    # 合并LoRA权重
    merge_status = merge_lora_weights()
    if merge_status != 0:
        return
    
    # 评估模型
    eval_status = evaluate_model()
    if eval_status != 0:
        return
    
    print("\n=== 模型微调流程完成 ===")
    print("训练日志: train_logs.txt")
    print("合并日志: merge_logs.txt")
    print("评估日志: eval_logs.txt")
    print("合并后的模型: ./merged_model")
    print("预测结果: predictions.json")

if __name__ == "__main__":
    main()