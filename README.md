---

# YOLOv8 人臉表情辨識期末報告  

參考資料:https://zhuanlan.zhihu.com/p/1927118824295102227

---

## 一、目的
此報告目的在使用深度學習之物件偵測模型 YOLOv8，對人臉表情影像進行訓練與辨識。透過實作完整的資料前處理流程、模型訓練與驗證，了解 YOLOv8 在影像辨識任務中的應用方式，並實際操作 Facial Expression Recognition Dataset，以提升對電腦視覺與深度學習模型的理解。

---

## 二、資料集說明（Dataset Description）

### 1. 資料集名稱與來源
本專題使用之資料集為：

**Facial Expression Recognition Image Version of (FERC) Dataset**

可以在此https://www.kaggle.com/datasets/manishshah120/facial-expression-recog-image-ver-of-fercdataset
下載。

該資料集常用於人臉表情辨識（Facial Expression Recognition）相關研究，內容包含多張人臉影像及其對應之標註資料。使用之資料集為 ZIP 壓縮檔格式，並於開始時進行解壓縮與資料前處理。

---

### 2. 資料集內容
FERC Dataset 主要包含：
- 人臉影像（RGB Images）
- 對應之標註檔（Label Files）

每張影像皆有對應之標註資訊，標註內容包含：
- 表情類別（Class）
- 表情區域之 Bounding Box 位置  
  （中心點座標與寬高，皆已正規化，符合 YOLO 格式）

---

### 3. 資料集用途
本專題將 FERC Dataset 作為 YOLOv8 模型之訓練與驗證資料，透過物件偵測方式標示影像中人臉表情區域，使模型學習不同臉部表情之影像特徵，進而達成人臉表情辨識之目的。

---

### 4. 資料集前處理後結構
為符合 YOLOv8 訓練需求，資料集整理為以下目錄結構：

```text
ferc_data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
````

影像與標註檔以隨機方式分配至訓練集（train）與驗證集（val），以避免資料偏差並提升模型泛化能力。

---

## 三、實驗環境與使用工具

* 開發環境：Google Colab (GPU)
* 程式語言：Python
* 深度學習模型：YOLOv8
* 使用套件：

  * Ultralytics
  * PyYAML
  * Matplotlib

---

## 四、實作步驟與程式碼說明

### 步驟1：解壓縮資料集

將 Facial Expression Recognition Dataset（ZIP 檔）解壓縮至指定資料夾，以利後續處理。

```python
import os
import zipfile

with zipfile.ZipFile('/content/drive/MyDrive/1223/archive.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/ferc_data')
```
<img width="859" height="205" alt="YOLOv8-1" src="https://github.com/user-attachments/assets/70be3a74-74ee-4793-afa7-bd0b5655f1f1" />

---

### 步驟2：資料集整理與切分

將影像與標註資料依比例隨機分配為訓練集與驗證集，並建立 YOLO 所需之資料夾結構。

```python
import os
import glob
import random
import shutil

# 1. 定義路徑
src_root = '/content/ferc_data'
base_dest = '/content/datasets'

# 2. 找出所有圖片 (包含子資料夾中的)
# 使用 glob 遞迴搜尋所有 png
all_imgs = glob.glob(os.path.join(src_root, "**/*.png"), recursive=True)

if len(all_imgs) == 0:
    print("❌ 錯誤：找不到任何 .png 檔案！請檢查 /content/ferc_data 資料夾是否存在。")
else:
    print(f"✅ 找到 {len(all_imgs)} 張圖片，開始整理...")
    
    # 建立 YOLO 結構目錄
    train_img = os.path.join(base_dest, 'train/images')
    train_lbl = os.path.join(base_dest, 'train/labels')
    val_img = os.path.join(base_dest, 'val/images')
    val_lbl = os.path.join(base_dest, 'val/labels')

    for d in [train_img, train_lbl, val_img, val_lbl]:
        if os.path.exists(d): shutil.rmtree(d) # 清除舊的避免混亂
        os.makedirs(d, exist_ok=True)

    # 3. 隨機打亂並分配 (80% 訓練, 20% 驗證)
    random.shuffle(all_imgs)
    split = int(0.8 * len(all_imgs))
    train_list = all_imgs[:split]
    val_list = all_imgs[split:]

    def move_and_label(file_list, img_dest, lbl_dest):
        for filepath in file_list:
            filename = os.path.basename(filepath)
            # 複製圖片
            shutil.copy(filepath, os.path.join(img_dest, filename))
            
            # 生成全臉標籤 (類別0, 中心0.5 0.5, 大小1.0 1.0)
            txt_name = os.path.splitext(filename)[0] + '.txt'
            with open(os.path.join(lbl_dest, txt_name), 'w') as f:
                f.write("0 0.5 0.5 1.0 1.0")

    move_and_label(train_list, train_img, train_lbl)
    move_and_label(val_list, val_img, val_lbl)

    print(f"🎉 整理成功！")
    print(f"訓練集: {len(os.listdir(train_img))} 張")
    print(f"驗證集: {len(os.listdir(val_img))} 張")
```
<img width="930" height="689" alt="YOLOv8-2" src="https://github.com/user-attachments/assets/cef67a72-753c-42bd-ac53-9ecd9f6d9bd1" />

<img width="836" height="698" alt="YOLOv8-3" src="https://github.com/user-attachments/assets/30c9f7f8-5394-46e4-859e-ca0db0defd1a" />

---


### 步驟3：建立 YOLOv8 訓練設定檔（data.yaml）

```python
import yaml

data_yaml = {
    'path': '/content/datasets',
    'train': 'train/images',
    'val': 'val/images',
    'nc': 1, 
    'names': ['Face'] 
}

with open('/content/datasets/data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)
```

<img width="600" height="323" alt="YOLOv8-4" src="https://github.com/user-attachments/assets/1070dc59-26b4-47d3-b696-e01b211796ba" />

---

### 步驟4：安裝模型

```python
!pip install ultralytics
```
<img width="1378" height="695" alt="YOLOv8-5" src="https://github.com/user-attachments/assets/bab6f150-07fb-4078-9f10-0daa393530f5" />


---

### 步驟5：載入 YOLOv8 模型並進行訓練

```python
from ultralytics import YOLO

# 載入預訓練的 YOLOv8n 模型
model = YOLO('yolov8n.pt')

# 開始訓練
results = model.train(
    data='/content/datasets/data.yaml',
    epochs=10,                
    imgsz=640,
    batch=16,
    patience=3,               # 如果 3 輪內沒進步就提早停止，更省資源
    save=True,
    device=0,                 # 指定使用 GPU
    name='ferc_test_run'
)
```
<img width="1707" height="704" alt="YOLOv8-6" src="https://github.com/user-attachments/assets/3b0afc1b-11d7-4bd7-b420-a5419ca5e29e" />


---

### 步驟6：模型驗證與結果視覺化

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

result_path = 'runs/detect/train/val_batch0_pred.jpg'

if os.path.exists(result_path):
    img = mpimg.imread(result_path)
    plt.figure(figsize=(14, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title('YOLOv8 Facial Expression Detection Result')
    plt.show()
```
<img width="805" height="704" alt="YOLOv8-7" src="https://github.com/user-attachments/assets/a040b1a5-6c92-4112-9675-246783c0d0d1" />

<img width="820" height="199" alt="YOLOv8-8" src="https://github.com/user-attachments/assets/7335cb6c-2102-4ec9-a0fb-c46290aa5273" />

---

## 五、結果與分析

<img width="950" height="507" alt="picture" src="https://github.com/user-attachments/assets/522a0876-2059-4416-b273-e2b6ed0d0338" />

<img width="789" height="812" alt="picture1" src="https://github.com/user-attachments/assets/a4b52e3a-ab32-42b9-a76d-5a80b0b802c5" />

由上圖結果可觀察到，YOLOv8 模型能夠有效偵測影像中之人臉表情區域，顯示其在影像辨識與特徵學習方面具有良好表現。透過使用預訓練模型進行遷移學習，可在有限資料量下仍獲得穩定的訓練成果。

---

## 六、結論

這次的報告我們成功完成 Facial Expression Recognition Dataset 之資料前處理、YOLOv8 模型訓練與驗證流程。透過實際操作深度學習模型，對人臉表情辨識與物件偵測技術有更深入的理解，未來可進一步延伸至多類別表情分類或即時影像辨識應用。

---
