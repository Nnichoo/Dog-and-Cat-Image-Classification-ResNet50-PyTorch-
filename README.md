# Dogs and Cats Classification with ResNet50 (Kaggle Notebook)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-torch%2Ftorchvision-red)
![License](https://img.shields.io/badge/License-MIT-green)

Project ini membangun pipeline **end-to-end** untuk **klasifikasi gambar kucing vs anjing** menggunakan **ResNet50 pretrained (ImageNet)**, dijalankan sepenuhnya di **Kaggle Notebook**. Pipeline dibuat **scalable & robust**: indexing data tanpa copy dataset, split stratified, augmentasi, mixed precision training (AMP), early stopping, serta evaluasi lengkap menggunakan **Confusion Matrix + ROC-AUC**.

---

## Overview

Dengan dataset **Dogs vs Cats** (25.000 gambar train), notebook ini:

- Melakukan auto-detect folder yang benar (menghindari error path / dataset kosong)
- Membuat index gambar secara rekursif (aman jika struktur folder berubah)
- Mengambil label dari filename (`cat.xxx.jpg` / `dog.xxx.jpg`) atau dari nama folder (`cat/` dan `dog/`)
- Membagi data menjadi train/validation secara stratified (proporsi kelas tetap seimbang)
- Melatih model ResNet50 pretrained dengan AdamW + Cosine Annealing
- Menggunakan Mixed Precision (torch.cuda.amp) agar training lebih cepat di GPU Kaggle
- Mengevaluasi performa dengan Validation Accuracy, Confusion Matrix, Classification Report, ROC Curve + AUC

---

## Dataset

- Source (Kaggle Competition): https://www.kaggle.com/competitions/dogs-vs-cats/data
- Total Train Images: 25,000
- Classes: 2 (`cat`, `dog`)
- Label mapping: `0 = cat`, `1 = dog`
- Naming pattern:
  - Train: `cat.0.jpg`, `dog.1.jpg`, dst
  - Test (opsional): `1.jpg`, `2.jpg`, dst

Dataset tidak diupload ke repo (ukuran besar). Repo hanya menyimpan notebook & hasil evaluasi.

---

## Preprocessing & Data Handling

- Tidak melakukan copy + resize + save ke folder baru (hemat storage & lebih scalable)
- Indexing rekursif menggunakan `rglob()` (aman untuk folder nested)
- Safe loading pada train (`LOAD_TRUNCATED_IMAGES` + retry sample) untuk menghindari crash jika ada file bermasalah
- Stratified split untuk menjaga proporsi cat/dog tetap seimbang
- Reproducible menggunakan seed tetap (`SEED = 42`)

---

## Model Architecture

- Backbone: `torchvision.models.resnet50`
- Pretrained Weights: ImageNet (Torchvision `ResNet50_Weights.DEFAULT`)
- Modified Head (Binary Logit):
  ```python
  model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
  model.fc = nn.Linear(model.fc.in_features, 1)  # output 1 logit
  ```

## Training Setup

- Loss Function: BCEWithLogitsLoss
- Optimizer: AdamW
- LR Scheduler: CosineAnnealingLR
- Precision: Mixed-precision training via torch.cuda.amp (autocast + GradScaler)
- Early Stopping: patience = 3

## Data Augmentation

### Train Transforms:

- RandomResizedCrop(224, scale=(0.75, 1.0)
- RandomHorizontalFlip(p=0.5)
- RandomRotation(8)
- ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)
- Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
### Validation Transforms:

- Resize(int(224*1.14))
- CenterCrop(224)
- Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Training Configuration

| Parameter                 | Nilai Default      | Keterangan                                                                                                         |
| :------------------------ | :----------------- | :----------------------------------------------------------------------------------------------------------------- |
| **Epochs**                | 8 (max)            | Jumlah iterasi penuh pelatihan melalui seluruh dataset (training bisa berhenti lebih cepat karena early stopping). |
| **Batch Size**            | 64                 | Jumlah sampel gambar yang diproses per iterasi selama pelatihan.                                                   |
| **Image Size**            | 224×224            | Resolusi akhir gambar input setelah augmentasi dan pra-pemrosesan (standar ResNet50).                              |
| **Optimizer**             | AdamW              | Optimizer yang stabil untuk fine-tuning model pretrained dengan regularisasi weight decay.                         |
| **Initial Learning Rate** | 1e-4               | Laju pembelajaran awal sebelum dijadwalkan menurun oleh scheduler.                                                 |
| **Weight Decay**          | 1e-4               | Regularisasi (L2 via AdamW) untuk membantu mencegah overfitting pada bobot model.                                  |
| **LR Scheduler**          | Cosine Annealing   | Menurunkan learning rate mengikuti kurva kosinus selama pelatihan.                                                 |
| **Loss Function**         | BCEWithLogitsLoss  | Loss yang tepat untuk **binary classification** dengan output **1 logit** (lebih stabil daripada sigmoid+BCELoss). |
| **Random Seed**           | 42                 | Nilai seed untuk memastikan hasil yang dapat direproduksi (PyTorch, NumPy, Python).                                |
| **GPU**                   | P100 / T4 (Kaggle) | Akselerasi pelatihan menggunakan GPU yang tersedia di lingkungan Kaggle.                                           |


## Result Validation

| Metric                                         |                          Value |
| :--------------------------------------------- | -----------------------------: |
| **Best Validation Accuracy (best_val_acc)**    |                     **0.9936** |
| **Training Accuracy (train_acc @ best epoch)** |                     **0.9853** |

## Evaluation Outputs

Notebook menghasilkan:
- Training vs Validation Curves (Loss & Accuracy)
- Confusion Matrix Heatmap
- Classification Report (precision/recall/F1)
- ROC Curve + AUC (Validation)

## How to Run on Kaggle

1. **Open a new Kaggle Notebook**  
2. **Add the dataset**:  
   - Click **“+ Add data”** in the notebook editor  
   - Search for **“Dogs vs Cats (Competition)”**  
   - Click **Add** to attach it to your notebook  
3. **Enable GPU**:  
   - Go to **Notebook Settings → Accelerator → GPU (P100 or T4)**  
4. **Ensure the input structuret**
   - /kaggle/input/dogs-and-cats-dataset/train
   - /kaggle/input/dogs-and-cats-dataset/test
5. **Run the dog-and-cats.ipynb notebook from top to bottom and evaluate validation**
