
# Transformer for Time Series Regression

This project demonstrates how Transformer-based architectures can be effectively applied to **time series regression tasks**. The focus is on implementing and experimenting with advanced Transformer variants designed specifically for time series data.

## 📌 Project Objective

The goal of this project is to show how Transformer models can be adapted and used for time series forecasting and regression, providing a hands-on demonstration of the potential and performance of these architectures compared to traditional models.

## 🧠 Models Included

This project specifically demonstrates the use of the following Transformer-based models:

- **Autoformer**: A decomposition-based Transformer model that captures trend and seasonal components in time series data.  
  📄 *Reference:* [Autoformer: Decomposition Transformers with Auto-Correlation Mechanism for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)

- **PatchTST**: A recent model that adapts the Transformer by introducing patching mechanisms for better temporal representation in univariate/multivariate time series.  
  📄 *Reference:* [PatchTST: Training-free Timeseries Classification with Patch Transformer](https://arxiv.org/abs/2211.14730)

## 📁 Repository Structure

```
.
├── test_trans_REG.ipynb       # Main notebook demonstrating the time series regression using Transformers
├── README.md                  # This file
```

## 📊 Dataset

The demonstration is based on a synthetic or pre-processed dataset tailored for univariate or multivariate time series regression. You can replace it with your own dataset to evaluate the models in a different context.

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/transformer-time-series-demo.git
   cd transformer-time-series-demo
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook and run the code:
   ```bash
   jupyter notebook test_trans_REG.ipynb
   ```

## 📝 Citation

If you use the models or ideas from this project, please cite the following papers:

```bibtex
@article{wu2021autoformer,
  title={Autoformer: Decomposition Transformers with Auto-Correlation Mechanism for Long-Term Series Forecasting},
  author={Haixu Wu and Yiming Chen and Tian Zhou and Xiyou Zhou and Weijie Wang and Jianmin Wang and Mingsheng Long},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}

@article{nie2022patchtst,
  title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author={Zengyi Nie and Yue Zhang and Yifan Zhang and Minjie Shen and Zhiming Ma},
  journal={arXiv preprint arXiv:2211.14730},
  year={2022}
}
```

## ✨ Acknowledgements

- Thanks to the original authors of **Autoformer** and **PatchTST** for their contributions to time series modeling.
- This project is a learning demonstration and is not affiliated with the authors of the cited works.
