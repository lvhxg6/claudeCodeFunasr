#!/bin/bash
#
# FunASR 服务安装脚本
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${GREEN}=== FunASR 服务安装 ===${NC}"
echo "项目目录: $PROJECT_ROOT"
echo ""

# 1. 检查 Python 版本
echo -e "${YELLOW}[1/6] 检查 Python 版本...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}错误: Python 版本需要 >= 3.10，当前版本: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 版本: $PYTHON_VERSION${NC}"
echo ""

# 2. 创建虚拟环境
echo -e "${YELLOW}[2/6] 创建虚拟环境...${NC}"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ 虚拟环境创建成功${NC}"
fi
echo ""

# 3. 激活虚拟环境并安装依赖
echo -e "${YELLOW}[3/6] 安装 Python 依赖...${NC}"
source venv/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel

# 安装依赖
pip install -r requirements.txt

echo -e "${GREEN}✓ 依赖安装完成${NC}"
echo ""

# 4. 下载 FunASR 模型
echo -e "${YELLOW}[4/6] 下载 FunASR 模型...${NC}"
echo "首次运行会自动下载模型（约 200-500MB），请耐心等待..."

# 创建测试脚本来触发模型下载
cat > /tmp/funasr_download_models.py << 'EOF'
import sys
from funasr import AutoModel

print("正在下载 ASR 模型...")
asr_model = AutoModel(model="paraformer-zh-streaming", device="cpu")
print("✓ ASR 模型下载完成")

print("正在下载 VAD 模型...")
vad_model = AutoModel(model="fsmn-vad", device="cpu")
print("✓ VAD 模型下载完成")

print("正在下载标点模型...")
punc_model = AutoModel(model="ct-punc", device="cpu")
print("✓ 标点模型下载完成")

print("\n所有模型下载完成！")
EOF

python /tmp/funasr_download_models.py
rm /tmp/funasr_download_models.py

echo -e "${GREEN}✓ 模型下载完成${NC}"
echo ""

# 5. 创建必要的目录
echo -e "${YELLOW}[5/6] 创建必要的目录...${NC}"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/temp"
echo -e "${GREEN}✓ 目录创建完成${NC}"
echo ""

# 6. 创建符号链接
echo -e "${YELLOW}[6/6] 创建命令行工具符号链接...${NC}"

# 给脚本添加执行权限
chmod +x "$PROJECT_ROOT/scripts/funasr-service.sh"
chmod +x "$PROJECT_ROOT/scripts/funasr-transcribe.sh"

# 创建符号链接到 /usr/local/bin
if [ -w "/usr/local/bin" ]; then
    ln -sf "$PROJECT_ROOT/scripts/funasr-service.sh" /usr/local/bin/funasr-service
    ln -sf "$PROJECT_ROOT/scripts/funasr-transcribe.sh" /usr/local/bin/funasr-transcribe
    echo -e "${GREEN}✓ 符号链接创建成功${NC}"
    echo "  - funasr-service: 服务管理"
    echo "  - funasr-transcribe: 命令行转录工具"
else
    echo -e "${YELLOW}⚠ 无法创建符号链接到 /usr/local/bin（需要 sudo 权限）${NC}"
    echo "请手动运行："
    echo "  sudo ln -sf $PROJECT_ROOT/scripts/funasr-service.sh /usr/local/bin/funasr-service"
    echo "  sudo ln -sf $PROJECT_ROOT/scripts/funasr-transcribe.sh /usr/local/bin/funasr-transcribe"
fi
echo ""

# 安装完成
echo -e "${GREEN}=== 安装完成 ===${NC}"
echo ""
echo "使用方法："
echo "  funasr-service start    # 启动服务"
echo "  funasr-service stop     # 停止服务"
echo "  funasr-service status   # 查看状态"
echo "  funasr-service logs     # 查看日志"
echo ""
echo "测试服务："
echo "  curl http://localhost:10095/health"
echo ""
