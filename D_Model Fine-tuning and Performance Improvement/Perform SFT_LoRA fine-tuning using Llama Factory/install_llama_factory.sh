#!/bin/bash
"""
Llama Factory安装与配置脚本
按照官方GitHub仓库的README操作
"""

# 安装Python 3.10或更高版本
if ! command -v python3.10 &> /dev/null; then
    echo "正在安装Python 3.10..."
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3.10-dev
fi

# 创建并激活虚拟环境
mkdir -p llama_factory_env
cd llama_factory_env
python3.10 -m venv .venv
source .venv/bin/activate

# 安装Git
if ! command -v git &> /dev/null; then
    echo "正在安装Git..."
    sudo apt install -y git
fi

# 克隆Llama Factory仓库
echo "正在克隆Llama Factory仓库..."
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装依赖
echo "正在安装依赖..."
pip install --upgrade pip
pip install -e .[torch,metrics]

# 安装额外依赖
pip install accelerate peft trl bitsandbytes

# 配置环境变量
echo "export LLAMA_FACTORY_HOME=./data" >> ~/.bashrc
source ~/.bashrc

# 验证安装
python -c "from llama_factory import train_bash; print('Llama Factory安装成功！')"

echo "Llama Factory安装与配置完成！"
echo "请运行以下命令激活虚拟环境："
echo "source llama_factory_env/.venv/bin/activate"
echo "然后进入LLaMA-Factory目录开始训练："
echo "cd llama_factory_env/LLaMA-Factory"