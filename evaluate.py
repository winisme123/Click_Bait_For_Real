import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, ViTImageProcessor, AutoModel, ViTModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.amp import autocast

# 1. Cấu hình đường dẫn
repo_dir = '.'
csv_path = os.path.join(repo_dir, 'clickbait_dataset_vietnamese.csv')
images_dir = os.path.join(repo_dir, 'images')

df = pd.read_csv(csv_path)

# 2. Khởi tạo Tokenizer và Processor
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# 3. Định nghĩa Dataset (giống hệt lúc train)
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
        title = str(row['title'])
        text_encoding = self.tokenizer(
            title, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )
        
        url = str(row['thumbnail_url'])
        clean_url = url.split('?')[0]
        img_name = os.path.basename(clean_url)
        if not img_name or not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.jfif')):
            img_name = f"thumb_{idx}.jpg"
            
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        image_encoding = self.processor(images=image, return_tensors="pt")
        label = 1.0 if row['label'] == 'clickbait' else 0.0
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'pixel_values': image_encoding['pixel_values'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

dataset = ViClickbaitDataset(df, images_dir, tokenizer, image_processor)

# Chia đúng tỷ lệ 80/20 như lúc train để lấy đúng tập Validation/Test
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# 4. Khai báo lại kiến trúc mô hình
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
            nn.Linear(256, 1)
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalClickbaitModel().to(device)
    
    # Load trọng số model đã train (nếu bạn có lưu file .pth, hoặc nếu bạn vừa train xong)
    model_weights_path = 'multimodal_clickbait_model.pth'
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path))
        print("Đã tải trọng số mô hình thành công từ file checkpoint!")
    else:
        print("CẢNH BÁO: Chưa tìm thấy file checkpoint .pth! Hãy đảm bảo bạn đã lưu model sau khi train.")

    model.eval()
    all_preds = []
    all_labels = []

    print("Đang tiến hành đánh giá chi tiết trên tập Validation...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            with autocast("cuda"):
                outputs = model(input_ids, attention_mask, pixel_values)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Tính toán các chỉ số đánh giá theo yêu cầu khóa luận
    print("\n================ BẢNG KẾT QUẢ ĐÁNH GIÁ CHI TIẾT ================")
    print(classification_report(all_labels, all_preds, target_names=['Non-Clickbait (0)', 'Clickbait (1)'], digits=4))
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Macro F1-score chính: {macro_f1:.4f}")
    
    print("\nConfusion Matrix (Ma trận nhầm lẫn):")
    print(confusion_matrix(all_labels, all_preds))
    print("================================================================")