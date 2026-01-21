#!/bin/bash
#
# FunASR 命令行转录工具
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 显示使用方法
usage() {
    echo "FunASR 命令行转录工具"
    echo ""
    echo "使用方法:"
    echo "  funasr-transcribe <audio_file>"
    echo ""
    echo "示例:"
    echo "  funasr-transcribe recording.wav"
    echo "  funasr-transcribe audio.mp3"
    echo ""
}

# 检查参数
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

AUDIO_FILE="$1"

# 检查文件是否存在
if [ ! -f "$AUDIO_FILE" ]; then
    echo -e "${RED}错误: 文件不存在: $AUDIO_FILE${NC}"
    exit 1
fi

# 检查服务是否运行
echo -e "${YELLOW}检查 FunASR 服务...${NC}"
if ! curl -s http://localhost:10095/health > /dev/null 2>&1; then
    echo -e "${RED}错误: FunASR 服务未运行${NC}"
    echo "请先启动服务: funasr-service start"
    exit 1
fi

echo -e "${GREEN}✓ 服务运行正常${NC}"
echo ""

# 调用 API 进行转录
echo -e "${YELLOW}正在转录音频文件...${NC}"
echo "文件: $AUDIO_FILE"
echo ""

RESULT=$(curl -s -X POST http://localhost:10095/v1/audio/transcriptions \
    -F "file=@$AUDIO_FILE" \
    -F "response_format=json")

# 提取文本
TEXT=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('text', ''))")

if [ -n "$TEXT" ]; then
    echo -e "${GREEN}转录结果:${NC}"
    echo "$TEXT"
else
    echo -e "${RED}转录失败${NC}"
    echo "$RESULT"
    exit 1
fi
