#!/usr/bin/env bash
set -euo pipefail

# 取消代理设置，COCO 在 AWS S3 上可以直连
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

# COCO2017 数据集下载脚本
# 下载训练集图片、验证集图片和注释文件到 data/coco2017 目录

DATA_DIR="/data/YBJ/cleansight/data/coco2017"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "========================================"
echo " 下载 COCO2017 数据集到: $DATA_DIR"
echo "========================================"

# 1. 下载训练集图片 (约18GB)
if [ ! -d "train2017" ]; then
    echo "[1/3] 下载 train2017 图片..."
    wget -c http://images.cocodataset.org/zips/train2017.zip
    echo "解压 train2017..."
    unzip -q train2017.zip
    rm train2017.zip
    echo "train2017 完成 ✓"
else
    echo "[1/3] train2017 已存在，跳过"
fi

# 2. 下载验证集图片 (约1GB)
if [ ! -d "val2017" ]; then
    echo "[2/3] 下载 val2017 图片..."
    wget -c http://images.cocodataset.org/zips/val2017.zip
    echo "解压 val2017..."
    unzip -q val2017.zip
    rm val2017.zip
    echo "val2017 完成 ✓"
else
    echo "[2/3] val2017 已存在，跳过"
fi

# 3. 下载注释文件 (约241MB)
if [ ! -d "annotations" ]; then
    echo "[3/3] 下载 annotations..."
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    echo "解压 annotations..."
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    echo "annotations 完成 ✓"
else
    echo "[3/3] annotations 已存在，跳过"
fi

echo ""
echo "========================================"
echo " COCO2017 下载完成!"
echo " 目录结构:"
echo "   $DATA_DIR/"
echo "   ├── train2017/     (训练集图片)"
echo "   ├── val2017/       (验证集图片)"
echo "   └── annotations/   (标注文件)"
echo "========================================"
