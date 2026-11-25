<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Land Segmentation</h3>

  <p align="center">
    Semantic land segmentation on aerial and satellite imagery using a PyTorch U-Net architecture.
    <br />
    <a href="https://github.com/aayushkarkiii/Land-Segmentation"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/aayushkarkiii/Land-Segmentation/issues/new?labels=bug">Report Bug</a>
    ·
    <a href="https://github.com/aayushkarkiii/Land-Segmentation/issues/new?labels=enhancement">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
        <li><a href="#repository-structure">Repository Structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- Uncomment and update path once you add a screenshot or sample prediction image -->
<!-- [![Product Name Screen Shot][product-screenshot]](https://github.com/aayushkarkiii/Land-Segmentation) -->

**Land-Segmentation** is a deep learning project for pixel-wise semantic segmentation of aerial and satellite imagery. Built with a **U-Net** encoder–decoder architecture in **PyTorch**, it isolates land cover regions from raw imagery for remote sensing, agricultural mapping, urban planning, and environmental monitoring.

Key highlights:
* **U-Net Architecture (`Model/unet_model.py`):** Convolutional encoder–decoder with skip connections to preserve fine spatial boundaries.
* **Custom Training CLI (`train.py`):** Configurable epochs, batch size, learning rate, and CPU/CUDA device selection.
* **Segmentation Metrics (`metrics.py`):** Evaluates predictions using Dice Coefficient, Intersection over Union (IoU), and Pixel Accuracy.
* **Interactive Notebook (`Notebook/LandSegmentation.ipynb`):** Supports experimentation and visual inspection.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python.org]][Python-url]
* [![PyTorch][PyTorch.org]][PyTorch-url]
* [![Jupyter][Jupyter.org]][Jupyter-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Repository Structure

```text
Land-Segmentation/
├── Dataset/
│   ├── images/                  # Raw input images (.jpg / .png)
│   └── masks/                   # Binary segmentation masks (.jpg / .png)
├── Model/
│   └── unet_model.py            # U-Net model definition
├── Notebook/
│   └── LandSegmentation.ipynb   # Experimentation and visualization notebook
├── metrics.py                   # Dice Coefficient, IoU, and Pixel Accuracy
├── train.py                     # Main training script
├── utils.py                     # Preprocessing and helper utilities
├── requirements.txt             # Python dependencies
└── README.md
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

* Python 3.x
* `pip`
* (Optional) CUDA-compatible GPU for faster training

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/aayushkarkiii/Land-Segmentation.git
   cd Land-Segmentation
   ```
2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Place your paired images and binary masks inside the `Dataset/` directory:
   ```text
   Dataset/
   ├── images/
   └── masks/
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
## Usage

Train the U-Net model using `train.py` with your preferred hyperparameters and device (`cuda` or `cpu`):

```sh
python train.py --epochs 20 --batch-size 16 --lr 0.001 --device cuda
```

For interactive experiments and visualization, open and run `Notebook/LandSegmentation.ipynb` in Jupyter.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [ ] Add data augmentation pipelines (horizontal/vertical flips, rotations)
- [ ] Integrate test pipeline with expanded visual results
- [ ] Deploy an interactive web app (e.g., Streamlit)
- [ ] Add transfer learning with pretrained encoders (e.g., ResNet)

See the [open issues](https://github.com/aayushkarkiii/Land-Segmentation/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are welcome and greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Aayush Karki - [@aayushkarkiii](https://github.com/aayushkarkiii) - your_email@example.com

Project Link: [https://github.com/aayushkarkiii/Land-Segmentation](https://github.com/aayushkarkiii/Land-Segmentation)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
* [PyTorch](https://pytorch.org/)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/aayushkarkiii/Land-Segmentation.svg?style=for-the-badge
[contributors-url]: https://github.com/aayushkarkiii/Land-Segmentation/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/aayushkarkiii/Land-Segmentation.svg?style=for-the-badge
[forks-url]: https://github.com/aayushkarkiii/Land-Segmentation/network/members
[stars-shield]: https://img.shields.io/github/stars/aayushkarkiii/Land-Segmentation.svg?style=for-the-badge
[stars-url]: https://github.com/aayushkarkiii/Land-Segmentation/stargazers
[issues-shield]: https://img.shields.io/github/issues/aayushkarkiii/Land-Segmentation.svg?style=for-the-badge
[issues-url]: https://github.com/aayushkarkiii/Land-Segmentation/issues
[license-shield]: https://img.shields.io/github/license/aayushkarkiii/Land-Segmentation.svg?style=for-the-badge
[license-url]: https://github.com/aayushkarkiii/Land-Segmentation/blob/main/LICENSE
[product-screenshot]: images/screenshot.png
[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch.org]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Jupyter.org]: https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white
[Jupyter-url]: https://jupyter.org/