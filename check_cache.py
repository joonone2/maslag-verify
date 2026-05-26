"""
check_cache.py
HuggingFace 캐시에 모델이 있는지 확인
"""
import os

cache_dir = r"C:\Users\user\.cache\huggingface\hub"

if os.path.exists(cache_dir):
    items = os.listdir(cache_dir)
    print("캐시 폴더 내용:")
    for item in items:
        print(f"  {item}")
else:
    print("캐시 폴더 없음")