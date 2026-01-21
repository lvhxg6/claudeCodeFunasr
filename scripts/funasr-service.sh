#!/bin/bash
#
# FunASR 服务管理脚本
# 用法: funasr-service {start|stop|restart|status|logs|enable|disable}
#

set -e

# 项目目录
PROJECT_DIR="/Users/liubu/hx/AI-SystemService/claudeCodeFunasr"
VENV_DIR="$PROJECT_DIR/venv"
LOG_FILE="$PROJECT_DIR/logs/service.log"
PLIST_FILE="$HOME/Library/LaunchAgents/com.liubu.funasr.plist"
SERVICE_LABEL="com.liubu.funasr"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

print_info() {
    echo -e "  $1"
}

# 检查虚拟环境
check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        print_error "虚拟环境不存在: $VENV_DIR"
        print_info "请先运行安装脚本: $PROJECT_DIR/scripts/install.sh"
        exit 1
    fi
}

# 检查 plist 文件是否存在
check_plist() {
    if [ ! -f "$PLIST_FILE" ]; then
        print_error "服务配置文件不存在: $PLIST_FILE"
        print_info "请重新运行安装脚本"
        exit 1
    fi
}

# 获取服务 PID
get_pid() {
    launchctl list | grep "$SERVICE_LABEL" | awk '{print $1}' | grep -v '^-$' || echo ""
}

# 检查服务是否运行
is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && [ "$pid" != "-" ]; then
        return 0
    else
        return 1
    fi
}

# 检查服务是否已加载
is_loaded() {
    launchctl list | grep -q "$SERVICE_LABEL"
}

# 启动服务
start_service() {
    check_venv
    check_plist

    if is_running; then
        print_warning "FunASR 服务已在运行中"
        print_info "PID: $(get_pid)"
        return 0
    fi

    print_info "正在启动 FunASR 服务..."

    # 确保日志目录存在
    mkdir -p "$(dirname "$LOG_FILE")"

    # 加载服务
    if ! is_loaded; then
        launchctl load "$PLIST_FILE"
    else
        # 如果已加载但未运行，先卸载再加载
        launchctl unload "$PLIST_FILE" 2>/dev/null
        launchctl load "$PLIST_FILE"
    fi

    # 等待服务启动
    sleep 3

    if is_running; then
        local pid=$(get_pid)
        print_success "FunASR 服务已启动"
        print_info "PID: $pid"
        print_info "地址: http://127.0.0.1:10095"
        print_info "日志: $LOG_FILE"

        # 等待模型加载
        print_info "等待模型加载..."
        local max_wait=120
        local waited=0
        while [ $waited -lt $max_wait ]; do
            local health=$(curl -s http://127.0.0.1:10095/health 2>/dev/null)
            if echo "$health" | grep -q '"status":"ok"'; then
                print_success "模型加载完成"
                return 0
            fi
            sleep 2
            waited=$((waited + 2))
        done

        print_warning "模型加载超时，请检查日志"
    else
        print_error "服务启动失败，请检查日志: $LOG_FILE"
        exit 1
    fi
}

# 停止服务
stop_service() {
    if ! is_loaded; then
        print_warning "FunASR 服务未加载"
        return 0
    fi

    if ! is_running; then
        print_warning "FunASR 服务未运行"
        launchctl unload "$PLIST_FILE" 2>/dev/null
        return 0
    fi

    local pid=$(get_pid)
    print_info "正在停止 FunASR 服务 (PID: $pid)..."

    # 卸载服务
    launchctl unload "$PLIST_FILE"

    # 等待进程结束
    local max_wait=10
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if ! is_running; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    print_success "FunASR 服务已停止"
}

# 重启服务
restart_service() {
    print_info "正在重启 FunASR 服务..."
    stop_service
    sleep 1
    start_service
}

# 查看服务状态
show_status() {
    echo "=== FunASR 服务状态 ==="
    echo

    if is_running; then
        local pid=$(get_pid)
        print_success "FunASR 服务正在运行"
        print_info "PID: $pid"
        print_info "地址: http://127.0.0.1:10095"

        # 检查健康状态
        echo
        print_info "健康检查:"
        local health=$(curl -s http://127.0.0.1:10095/health 2>/dev/null)
        if [ -n "$health" ]; then
            echo "  $health"
        else
            print_warning "无法获取健康状态"
        fi

        # 显示资源使用
        echo
        print_info "资源使用:"
        ps -p $pid -o pid,rss,vsz,%cpu,%mem,etime 2>/dev/null | head -2
    else
        print_error "FunASR 服务未运行"
    fi

    # 显示开机自启动状态
    echo
    if is_loaded; then
        print_info "开机自启动: 已启用"
    else
        print_info "开机自启动: 未启用"
    fi
}

# 查看日志
show_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        print_warning "日志文件不存在: $LOG_FILE"
        exit 1
    fi

    local lines=${1:-50}
    print_info "显示最近 $lines 行日志 (Ctrl+C 退出):"
    echo
    tail -f -n "$lines" "$LOG_FILE"
}

# 启用开机自启动
enable_service() {
    check_plist

    if is_loaded; then
        print_warning "服务已启用开机自启动"
    else
        launchctl load "$PLIST_FILE"
        print_success "已启用开机自启动"
    fi
}

# 禁用开机自启动
disable_service() {
    if ! is_loaded; then
        print_warning "服务未启用开机自启动"
    else
        launchctl unload "$PLIST_FILE"
        print_success "已禁用开机自启动"
    fi
}

# 显示帮助
show_help() {
    echo "FunASR 服务管理脚本"
    echo
    echo "用法: funasr-service <命令> [参数]"
    echo
    echo "命令:"
    echo "  start     启动服务"
    echo "  stop      停止服务"
    echo "  restart   重启服务"
    echo "  status    查看服务状态"
    echo "  logs [n]  查看日志 (默认最近 50 行)"
    echo "  enable    启用开机自启动"
    echo "  disable   禁用开机自启动"
    echo "  help      显示此帮助信息"
    echo
    echo "示例:"
    echo "  funasr-service start      # 启动服务"
    echo "  funasr-service status     # 查看状态"
    echo "  funasr-service logs 100   # 查看最近 100 行日志"
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
            show_logs "${2:-50}"
            ;;
        enable)
            enable_service
            ;;
        disable)
            disable_service
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: ${1:-}"
            echo
            show_help
            exit 1
            ;;
    esac
}

main "$@"
