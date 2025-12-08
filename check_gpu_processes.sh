#!/bin/bash
# 查看GPU上的进程信息

echo "======================================"
echo "GPU 使用情况总览"
echo "======================================"
nvidia-smi

echo ""
echo "======================================"
echo "详细进程信息"
echo "======================================"

# 查看进程 3122315
echo ""
echo "进程 3122315 详情："
ps -f -p 3122315 2>/dev/null || echo "  进程不存在或已结束"
ps aux | grep 3122315 | grep -v grep 2>/dev/null

# 查看进程 3868016
echo ""
echo "进程 3868016 详情："
ps -f -p 3868016 2>/dev/null || echo "  进程不存在或已结束"
ps aux | grep 3868016 | grep -v grep 2>/dev/null

echo ""
echo "======================================"
echo "所有占用GPU的Python进程"
echo "======================================"
ps aux | grep python | grep -E "qiyang_shi|ziqiang" | grep -v grep

echo ""
echo "======================================"
echo "GPU 6 和 GPU 7 的详细信息"
echo "======================================"
nvidia-smi -i 6,7 --query-compute-apps=pid,process_name,used_memory --format=csv

echo ""
echo "建议："
echo "1. 如果是您自己的其他训练任务，可以等待完成"
echo "2. 如果是旧的残留进程，可以使用: kill -9 <PID>"
echo "3. 如果进程已经不存在，说明显存可能有碎片化，重启GPU或等待释放"


