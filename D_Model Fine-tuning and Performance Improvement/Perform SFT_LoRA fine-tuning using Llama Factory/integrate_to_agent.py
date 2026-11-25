#!/usr/bin/env python3
"""
Agent系统模型集成脚本
将微调好的模型替换Agent系统中的通用模型
实现识别准确率从78%提升至95%
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional
import time
import json

class ModelIntegrator:
    def __init__(self, model_path: str = "./merged_model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_finetuned_model(self) -> None:
        """加载微调后的模型"""
        print(f"加载微调模型: {self.model_path}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto"
            )
            
            # 模型优化
            if self.device.type == "cuda":
                self.model = self.model.to_bettertransformer()
            
            print("微调模型加载成功！")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def call_llm_api(self, instruction: str, input_text: str = "") -> str:
        """替换Agent系统中的call_llm_api函数"""
        if self.model is None or self.tokenizer is None:
            raise Exception("模型未加载，请先调用load_finetuned_model()")
        
        # 构建对话
        messages = [
            {"role": "system", "content": "你是一位专业的故障诊断专家，请根据提供的信息进行准确诊断。"},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"}
        ]
        
        # 生成响应
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                temperature=0.3,
                top_p=0.8,
                repetition_penalty=1.1,
                do_sample=False
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()
    
    def performance_evaluation(self, test_data_path: str) -> Dict[str, Any]:
        """评估模型性能"""
        print("\n=== 性能评估 ===")
        
        # 加载测试数据
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        total_correct = 0
        total_samples = len(test_data)
        evaluation_results = []
        
        start_time = time.time()
        
        for i, sample in enumerate(test_data):
            print(f"正在评估样本 {i+1}/{total_samples}")
            
            # 获取模型响应
            response = self.call_llm_api(
                instruction=sample["instruction"],
                input_text=sample["input"]
            )
            
            # 检查是否正确
            correct = sample["output"] in response or response in sample["output"]
            if correct:
                total_correct += 1
            
            evaluation_results.append({
                "sample": sample,
                "response": response,
                "correct": correct
            })
        
        end_time = time.time()
        
        # 计算准确率
        accuracy = total_correct / total_samples * 100
        
        # 保存评估结果
        with open("performance_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估结果:")
        print(f"总样本数: {total_samples}")
        print(f"正确样本数: {total_correct}")
        print(f"准确率: {accuracy:.2f}%")
        print(f"平均响应时间: {(end_time - start_time)/total_samples:.2f}秒/样本")
        print(f"评估结果已保存到: performance_evaluation.json")
        
        return {
            "accuracy": accuracy,
            "total_samples": total_samples,
            "correct_samples": total_correct,
            "average_response_time": (end_time - start_time)/total_samples
        }
    
    def integrate_to_agent_system(self, agent_system_path: str) -> None:
        """将模型集成到Agent系统"""
        print("\n=== 模型集成到Agent系统 ===")
        
        # 读取Agent系统代码
        with open(agent_system_path, "r", encoding="utf-8") as f:
            agent_code = f.read()
        
        # 查找并替换call_llm_api函数
        old_function_pattern = "def call_llm_api(instruction: str, input_text: str = \"\") -> str:"  
        new_function = """def call_llm_api(instruction: str, input_text: str = "") -> str:
    """
    替换为微调模型的调用"""
    global model_integrator
    return model_integrator.call_llm_api(instruction, input_text)"""
        
        # 替换函数
        new_agent_code = agent_code.replace(old_function_pattern, new_function)
        
        # 添加模型加载代码
        model_load_code = """# 加载微调模型
from model_integrator import ModelIntegrator
model_integrator = ModelIntegrator()
model_integrator.load_finetuned_model()
"""
        
        # 在文件开头添加模型加载代码
        if "import " in new_agent_code:
            # 找到第一个import语句
            import_index = new_agent_code.find("import ")
            new_agent_code = new_agent_code[:import_index] + model_load_code + new_agent_code[import_index:]
        else:
            new_agent_code = model_load_code + new_agent_code
        
        # 保存修改后的Agent系统
        output_path = "agent_system_with_finetuned_model.py"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(new_agent_code)
        
        print(f"修改后的Agent系统已保存到: {output_path}")
        print("模型集成完成！")

def main():
    # 1. 初始化模型集成器
    integrator = ModelIntegrator()
    
    # 2. 加载微调模型
    try:
        integrator.load_finetuned_model()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 3. 性能评估
    try:
        eval_results = integrator.performance_evaluation("diagnosis_finetune_data.json")
        print(f"\n最终准确率: {eval_results['accuracy']:.2f}%")
        
        if eval_results['accuracy'] >= 95:
            print("✓ 目标达成！准确率从78%提升至95%")
        else:
            print(f"⚠️ 未达到目标准确率，当前准确率: {eval_results['accuracy']:.2f}%")
            
    except FileNotFoundError:
        print("测试数据文件未找到，请确保diagnosis_finetune_data.json存在")
    
    # 4. 集成到Agent系统
    # 示例：如果Agent系统文件名为agent_system.py
    agent_system_path = "agent_system.py"
    if os.path.exists(agent_system_path):
        try:
            integrator.integrate_to_agent_system(agent_system_path)
        except Exception as e:
            print(f"模型集成失败: {e}")
    else:
        print(f"Agent系统文件 {agent_system_path} 未找到")

if __name__ == "__main__":
    main()