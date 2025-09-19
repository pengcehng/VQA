# 设置CUDA设备
$env:CUDA_VISIBLE_DEVICES = 0

# 切换到src目录
Set-Location -Path "src"

# 创建logs目录（如果不存在）
if (-not (Test-Path "../logs")) {
    New-Item -ItemType Directory -Path "../logs" -Force
}

# 运行测试并记录日志
python -u test_demo.py --method_name single-scale --dist ../LSVQ/test1_3.mp4 --output ../result.txt --is_gpu | Out-File -FilePath "../logs/test_demo.log"

# 返回原目录
Set-Location -Path ".."