import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, ViTImageProcessor, AutoModel, ViTModel
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# 1. Đường dẫn dữ liệu tại máy local
repo_dir = '.'
csv_path = os.path.join(repo_dir, 'clickbait_dataset_vietnamese.csv')
images_dir = os.path.join(repo_dir, 'images')

df = pd.read_csv(csv_path)
print(f"Đã tải xong! Tổng số lượng mẫu dữ liệu: {len(df)}")

# 2. Khởi tạo Tokenizer và Image Processor
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# 3. Định nghĩa PyTorch Dataset (Đồng bộ cách gọi tên ảnh với script download)
class ViClickbaitDataset(Dataset):
    def __init__(self, dataframe, img_dir, tokenizer, processor, max_length=256):
        self.df = dataframe
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Xử lý văn bản
        title = str(row['title'])
        text_encoding = self.tokenizer(
            title,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Xử lý hình ảnh (Khớp chuẩn logic đặt tên file với download_images.py)
        url = str(row['thumbnail_url'])
        clean_url = url.split('?')[0]
        img_name = os.path.basename(clean_url)
        if not img_name or not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.jfif')):
            img_name = f"thumb_{idx}.jpg"
            
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback nếu ảnh lỗi hoặc thiếu
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        image_encoding = self.processor(images=image, return_tensors="pt")
        
        # Nhãn nhị phân
        label = 1.0 if row['label'] == 'clickbait' else 0.0
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'pixel_values': image_encoding['pixel_values'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

dataset = ViClickbaitDataset(df, images_dir, tokenizer, image_processor)

# 4. Chia tập Train / Val (num_workers=0 để tránh lỗi đa tiến trình trên Windows)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Giảm batch_size xuống 4 để an toàn tuyệt đối với VRAM 8GB của RTX 4060 khi chạy đa phương thức
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

print(f"Số lượng mẫu Train: {len(train_dataset)} | Số lượng mẫu Validation: {len(val_dataset)}")

# 5. Định nghĩa Kiến trúc Multimodal Model (PhoBERT + ViT + Cross-Modal Attention)
class MultimodalClickbaitModel(nn.Module):
    def __init__(self, phobert_name="vinai/phobert-base", vit_name="google/vit-base-patch16-224", num_heads=8):
        super(MultimodalClickbaitModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(phobert_name)
        self.image_encoder = ViTModel.from_pretrained(vit_name)
        
        text_hidden_size = self.text_encoder.config.hidden_size
        image_hidden_size = self.image_encoder.config.hidden_size
        
        if text_hidden_size != image_hidden_size:
            self.image_projection = nn.Linear(image_hidden_size, text_hidden_size)
        else:
            self.image_projection = nn.Identity()
            
        self.cross_attention = nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=num_heads, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state 
        
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = image_outputs.last_hidden_state
        image_features = self.image_projection(image_features)
        
        attn_output, _ = self.cross_attention(query=text_features, key=image_features, value=image_features)
        
        text_pooled = torch.mean(text_features, dim=1)
        multimodal_pooled = torch.mean(attn_output, dim=1)
        
        combined = torch.cat((text_pooled, multimodal_pooled), dim=1)
        out = self.classifier(combined)
        return out.squeeze(1)

# Đoạn mã huấn luyện chính
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalClickbaitModel().to(device)
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Không tìm thấy GPU'
    print(f"Đã khởi tạo mô hình trên thiết bị: {device} | Tên GPU: {gpu_name}")

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = (
        nn.BCEWithLogitsLoss()
    )
    scaler = GradScaler("cuda")

    def evaluate_model(model, dataloader, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)
                
                with autocast("cuda"):
                    outputs = model(input_ids, attention_mask, pixel_values)
                    loss = criterion(outputs, labels)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    num_epochs = 3
    print("\n--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN ĐA PHƯƠNG THỨC TRÊN GPU ---")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            # Sử dụng FP16 (autocast) để tối ưu tốc độ và tiết kiệm VRAM
            with autocast("cuda"):
                outputs = model(input_ids, attention_mask, pixel_values)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * input_ids.size(0)
            preds = (outputs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Hoàn tất:")
        print(f" - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f" - Val Loss:   {val_loss:.6f}   | Val Acc:   {val_acc*100:.2f}%\n")

    print("Quá trình huấn luyện đa phương thức đã hoàn thành thành công!")
    torch.save(model.state_dict(), "multimodal_clickbait_model.pth")
    print("Đã lưu trọng số mô hình thành multimodal_clickbait_model.pth!")