# Minecraft Biome Classifier 🌍
Convolutional Neural Network (CNN) model trained to classify 29 different Minecraft biomes from images.

## 🧠 Overview
This project uses a fine-tuned **ResNet-18** model trained with **PyTorch** to identify Minecraft biomes such as forests, deserts, and oceans.  
Developed as part of a hackathon challenge where participants built machine learning models to recognize biomes from in-game screenshots.

- Achieved **0.84 private leaderboard accuracy** (top score)
- Implemented **transfer learning** with frozen early ResNet layers
- Used **data augmentation** (horizontal flips, random rotations)
- Experimented with multiple models — CNN performed best

## 🧩 Model
- **Architecture:** ResNet-18 (transfer learning)
- **Framework:** PyTorch / Torchvision
- **Loss Function:** Cross-Entropy
- **Optimizer:** Adam (lr = 1e-3)
- **Early Stopping:** delta = 1e-4
- **Augmentations:** Resize, Normalize, Random Flip, Random Rotation

## 📂 Folder Structure
assets/ # Hackathon logo and visuals
models/ # Jupyter notebooks & trained model (.pth)
output/ # Sample and generated prediction CSVs
userkits/ # Custom dataset + utils for PyTorch


## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/minecraft-biome-classifier.git
cd minecraft-biome-classifier
pip install -r requirements.txt

jupyter notebook models/pytorch.ipynb

## Results

| Metric              | Score  |
| ------------------- | ------ |
| Public Leaderboard  | 0.8058 |
| Private Leaderboard | 0.8454 |

🙌 Acknowledgements
Originally inspired by the Blockography AI Hackathon.
