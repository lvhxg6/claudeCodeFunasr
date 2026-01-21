#!/bin/bash
#
# FunASR 服务管理脚本
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/logs/service.pid"
LOG_FILE="$PROJECT_ROOT/logs/service.log"

# 显示使用方法
usage() {
    echo "FunASR 服务管理工具"
    echo ""
    echo "使用方法:"
    echo "  funasr-service start    启动服务"
    echo "  funasr-service stop     停止服务"
    echo "  funasr-service restart  重启服务"
    echo "  funasr-service status   查看状态"
    echo "  funasr-service logs     查看日志"
    echo ""
}

# 检查服务是否运行
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# 启动服务
start_service() {
    echo -e "${YELLOW}启动 FunASR 服务...${NC}"

    if is_running; then
        echo -e "${RED}服务已在运行中${NC}"
        exit 1
    fi

    # 激活虚拟环境
    cd "$PROJECT_ROOT"
    source venv/bin/activate

    # 启动服务
    nohup python server.py > "$LOG_FILE" 2>&1 &
    PID=$!

    # 保存 PID
    echo $PID > "$PID_FILE"

    # 等待服务启动
    sleep 2

    # 检查服务是否成功启动
    if is_running; then
        echo -e "${GREEN}✓ 服务启动成功 (PID: $PID)${NC}"
        echo "日志文件: $LOG_FILE"
        echo "测试服务: curl http://localhost:10095/health"
    else
        echo -e "${RED}✗ 服务启动失败${NC}"
        echo "查看日志: funasr-service logs"
        exit 1
    fi
}

# 停止服务
stop_service() {
    echo -e "${YELLOW}停止 FunASR 服务...${NC}"

    if ! is_running; then
        echo -e "${RED}服务未运行${NC}"
        exit 1
    fi

    PID=$(cat "$PID_FILE")
    kill "$PID"

    # 等待进程结束
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    # 如果还在运行，强制杀死
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}强制停止服务...${NC}"
        kill -9 "$PID"
    fi

    rm -f "$PID_FILE"
    echo -e "${GREEN}✓ 服务已停止${NC}"
}

# 重启服务
restart_service() {
    echo -e "${YELLOW}重启 FunASR 服务...${NC}"
    if is_running; then
        stop_service
        sleep 1
    fi
    start_service
}

# 查看状态
show_status() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${GREEN}服务运行中${NC}"
        echo "PID: $PID"
        echo ""
        echo "进程信息:"
        ps -p "$PID" -o pid,ppid,%cpu,%mem,etime,command
        echo ""
        echo "端口监听:"
        lsof -i :10095 2>/dev/null || echo "未找到端口监听信息"
    else
        echo -e "${RED}服务未运行${NC}"
    fi
}

# 查看日志
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo -e "${RED}日志文件不存在${NC}"
    fi
}

# 主函数
main() {
    case "${1:-}" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"
