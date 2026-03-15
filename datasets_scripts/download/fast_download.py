#!/usr/bin/env python3
"""
多线程分段下载器 - 模拟 aria2c 的多连接下载
用法: python fast_download.py <url> <output_file> [num_threads]
"""
import os
import sys
import time
import threading
import urllib.request

# 取消代理
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(k, None)

def get_file_size(url):
    req = urllib.request.Request(url, method='HEAD')
    resp = urllib.request.urlopen(req, timeout=30)
    return int(resp.headers['Content-Length'])

def download_chunk(url, start, end, output_file, chunk_id, progress):
    """下载文件的一个分段"""
    req = urllib.request.Request(url)
    req.add_header('Range', f'bytes={start}-{end}')
    
    resp = urllib.request.urlopen(req, timeout=60)
    
    block_size = 1024 * 64  # 64KB
    downloaded = 0
    
    with open(output_file, 'r+b') as f:
        f.seek(start)
        while True:
            data = resp.read(block_size)
            if not data:
                break
            f.write(data)
            downloaded += len(data)
            progress[chunk_id] = downloaded

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"

def format_speed(speed):
    return format_size(speed) + "/s"

def download_file(url, output_file, num_threads=16):
    file_size = get_file_size(url)
    print(f"文件大小: {format_size(file_size)}")
    print(f"使用 {num_threads} 个线程并行下载")

    # 检查是否支持 Range
    # 创建空文件
    with open(output_file, 'wb') as f:
        f.seek(file_size - 1)
        f.write(b'\0')

    chunk_size = file_size // num_threads
    threads = []
    progress = {}

    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size - 1 if i < num_threads - 1 else file_size - 1
        progress[i] = 0

        t = threading.Thread(target=download_chunk, args=(url, start, end, output_file, i, progress))
        t.daemon = True
        threads.append(t)

    start_time = time.time()
    for t in threads:
        t.start()

    # 监控进度
    last_downloaded = 0
    while any(t.is_alive() for t in threads):
        time.sleep(2)
        total_downloaded = sum(progress.values())
        elapsed = time.time() - start_time
        speed = (total_downloaded - last_downloaded) / 2 if last_downloaded > 0 else total_downloaded / elapsed
        last_downloaded = total_downloaded
        pct = total_downloaded / file_size * 100
        eta = (file_size - total_downloaded) / speed if speed > 0 else 0
        print(f"\r  [{pct:5.1f}%] {format_size(total_downloaded)}/{format_size(file_size)}  "
              f"速度: {format_speed(speed)}  预计剩余: {eta:.0f}秒", end='', flush=True)

    for t in threads:
        t.join()

    print(f"\n✓ 下载完成: {output_file} ({format_size(file_size)})")
    elapsed = time.time() - start_time
    print(f"  总耗时: {elapsed:.1f}秒, 平均速度: {format_speed(file_size / elapsed)}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python fast_download.py <url> <output_file> [num_threads]")
        sys.exit(1)
    
    url = sys.argv[1]
    output = sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    
    download_file(url, output, threads)
