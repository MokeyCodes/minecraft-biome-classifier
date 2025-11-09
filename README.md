# Minecraft Biome Classifier 🌍
Convolutional Neural Network (CNN) model trained to classify 29 different Minecraft biomes from images.

## 🧠 Overview
This project uses a fine-tuned **ResNet-18** model trained with **PyTorch** to identify Minecraft biomes such as forests, deserts, and oceans.  
Developed as part of a hackathon challenge where participants built machine learning models to recognize biomes from in-game screenshots.

- Achieved **0.8454 private leaderboard accuracy**
- Implemented **transfer learning** with frozen early ResNet layers
- Used **data augmentation**: horizontal flips and random rotations
- Experimented with multiple models — CNN performed best

## 🧩 Model
- **Architecture:** ResNet-18 (transfer learning)
- **Framework:** PyTorch / Torchvision
- **Loss Function:** Cross-Entropy
- **Optimizer:** Adam (lr = 1e-4)
- **Early Stopping:** delta = 1e-4
- **Augmentations:** Resize, Normalize, Random Horizontal Flip, Random Rotation

## 📂 Folder Structure
- `assets/` – Hackathon logo and visuals
- `models/` – Jupyter notebook & trained model (`.pth`)
- `output/` – Sample and generated prediction CSVs
- `userkits/` – Custom dataset & utils for PyTorch

> Note: Other ML notebooks (`rf.ipynb`, `ridge.ipynb`, `xgboost.ipynb`) are excluded from the main workflow.

## ⚙️ Installation/Usage

1. **Clone the repository:**

```bash
git clone https://github.com/<MokeyCodes>/minecraft-biome-classifier.git
cd minecraft-biome-classifier

```
2. **Install Dependencies:**
```
pip install -r requirements.txt
```
3. **Run the Jupyter notebook**
```
jupyter notebook models/pytorch.ipynb
```
## 📊 Results

| Metric              | Score  |
| ------------------- | ------ |
| Public Leaderboard  | **0.8058** |
| Private Leaderboard | **0.8454** |



## 🙌 Acknowledgements
Originally inspired by the Blockography AI Hackathon.
