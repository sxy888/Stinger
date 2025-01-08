#!/bin/bash

# 服务名称，用于日志和状态信息
SERVICE_NAME="Streamlit Service"

# 启动服务的命令
START_CMD="python -m streamlit run ui/attack.py"

# 检查参数个数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {start|stop|restart}"
    exit 1
fi

# 获取操作类型
ACTION=$1

# 启动服务
start_service() {
    echo "Starting $SERVICE_NAME..."
    nohup $START_CMD > web_ui.log  2>&1 &
    echo "$SERVICE_NAME started."
}

# 停止服务
stop_service() {
    echo "Stopping $SERVICE_NAME..."
    # 使用 pgrep 获取所有匹配的 PID
    PIDS=$(pgrep -f "$START_CMD")
    if [ -z "$PIDS" ]; then
        echo "No $SERVICE_NAME process found."
    else
        kill -9 $PIDS
        echo "$SERVICE_NAME stopped."
    fi
}

# 重启服务
restart_service() {
    echo "Restarting $SERVICE_NAME..."
    stop_service
    sleep 2 # 等待一会儿再启动服务
    start_service
}

# 根据操作类型执行相应的函数
case $ACTION in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    *)
        echo "Invalid action: $ACTION"
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac

exit 0