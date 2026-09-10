import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

# 1. Đọc dataset
csv_path = 'clickbait_dataset_vietnamese.csv'
df = pd.read_csv(csv_path)

# 2. Tạo thư mục images nếu chưa có
images_dir = 'images'
os.makedirs(images_dir, exist_ok=True)


def download_image(row):
  url = str(row['thumbnail_url'])
  if pd.isna(url) or not url.startswith('http'):
    return

  # Xử lý tên file từ URL (loại bỏ query params nếu có)
  clean_url = url.split('?')[0]
  img_name = os.path.basename(clean_url)
  if not img_name or not img_name.lower().endswith(
      ('.jpg', '.jpeg', '.png', '.webp', '.jfif')
  ):
    img_name = f"thumb_{row.name}.jpg"

  img_path = os.path.join(images_dir, img_name)

  # Nếu đã tải rồi thì bỏ qua để tiết kiệm thời gian
  if os.path.exists(img_path):
    return

  try:
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
    }
    response = requests.get(url, headers=headers, timeout=6)
    if response.status_code == 200:
      with open(img_path, 'wb') as f:
        f.write(response.content)
  except Exception:
    pass


print(f'Bắt đầu crawl ảnh cho {len(df)} mẫu dữ liệu...')

# Sử dụng đa luồng (ThreadPoolExecutor) để tải song song cực nhanh
with ThreadPoolExecutor(max_workers=16) as executor:
  list(executor.map(download_image, [row for _, row in df.iterrows()]))

# Kiểm tra kết quả
downloaded_files = os.listdir(images_dir)
print(
    f'Tải hoàn tất! Số lượng ảnh thực tế trong thư mục images:'
    f' {len(downloaded_files)}/{len(df)}'
)