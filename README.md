# 🌍 Land Segmentation using U-Net (PyTorch)

This project demonstrates semantic land segmentation from aerial/satellite images using a U-Net architecture implemented in **PyTorch**. It is designed to detect and isolate different land types (like vegetation, water, roads, etc.) at the pixel level, aiding in remote sensing, urban planning, agriculture, and environmental monitoring.

---

## 📷 Sample Results

<p align="center">
  <img src="Notebook/sample_result.png" alt="Prediction Example" width="500"/>
</p>

---

## 🧠 Model Architecture

- **U-Net**: A convolutional neural network tailored for semantic segmentation.
- Developed using **PyTorch** for full control and flexibility.
- Architecture:
  - Encoder: Downsampling with convolution and max-pooling.
  - Bottleneck: High-level feature extraction.
  - Decoder: Upsampling with skip connections from encoder layers.
- Suitable for pixel-wise binary segmentation.

---

## 📁 Project Structure

Land-Segmentation/
├── Dataset/
├── images/       # Raw input images (.jpg/.png)
└── masks/        # Binary mask images (.jpg/.png)# Contains training images and masks (optional - may be downloaded separately)
├── Model/
│ └── unet_model.py # U-Net model definition
├── Notebook/
│ └── LandSegmentation.ipynb # Jupyter notebook for experiments
├── utils.py # Helper functions (e.g., preprocessing)
├── metrics.py # Dice and IoU metric functions
├── train.py # Script to train the model
├── requirements.txt # Python dependencies
└── README.md # Project documentation




---

## 🧪 Evaluation Metrics

Implemented:
- ✅ **Dice Coefficient** – Measures overlap between predicted and ground truth masks.
- ✅ **Intersection over Union (IoU)** – Ratio of intersection to union of predicted and true segments.
- ✅ **Pixel Accuracy** – Percentage of correctly classified pixels.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/aayushkarkiii/Land-Segmentation.git
cd Land-Segmentation

Train the model
-python train.py --epochs 20 --batch-size 16 --lr 0.001 --device cuda

In train.py:
--epochs         # Number of epochs
--batch-size     # Batch size
--lr             # Learning rate
--device         # Device (cuda or cpu)
--save-path      # Path to save model checkpoints


📈 Future Enhancements
 Add data augmentation (e.g., horizontal/vertical flips, rotations)

- Integrate the test pipeline with more visual results

 -Deploy as a web app using Streamlit

- Use transfer learning with pretrained encoders (e.g., ResNet as encoder)


👨‍💻 Author
Aayush Karki
🎓 Computer Science Engineer
🔗 GitHub: @aayushkarkiii
🔗 LinkedIn: linkedin.com/in/aayush-karki-4a96ba259

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
U-Net paper: https://arxiv.org/abs/1505.04597
PyTorch segmentation tutorials and community
