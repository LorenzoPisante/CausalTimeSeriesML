# CausalTimeSeriesML

## Full name: Neurochaos-TimeSeries: A Comprehensive Study of Time Series Cause-Effect Classification and Preservation


## Overview

This repository contains the code and resources for my master's thesis, titled **"A Comprehensive Study of Time Series Cause-Effect Classification and Preservation Using Neurochaos Learning."** The work focuses on evaluating various machine learning methods for time series classification, with a special emphasis on understanding and preserving the causal dynamics inherent in the data.

In this project, I explored three key classification methods:

- **Neurochaos Learning (NL)**: A brain-inspired learning algorithm that leverages chaotic neural dynamics for classification tasks.
- **Multilayer Perceptron (MLP)**: A traditional feedforward neural network.
- **Kolmogorov-Arnold Networks (KAN)**: A novel deep learning architecture with learnable activation functions.

The goal of this research was to assess how well these methods preserve causal relationships in time series data and their impact on the generalization and robustness of the models.

## Key Contributions

- **Preservation of Causal Dynamics**: This project demonstrates that preserving the causal structure in time series data enhances the generalization capabilities of machine learning models.
- **Experimental Evaluation**: Extensive experiments were conducted on synthetic and real-world datasets from the UCR Time Series Classification Archive, including noise introduction, few-shot learning, and transfer learning scenarios.
- **Neurochaos Learning**: The study highlights the superior performance of Neurochaos Learning in maintaining causal relationships and achieving better generalization on synthetic datasets.

## Experiments and Evaluation

### Synthetic Data Experiments

#### 1. **Noise Introduction**
To evaluate the robustness of the models, Gaussian noise was added to the synthetic datasets. The impact of noise on classification accuracy was analyzed to determine the models' ability to maintain performance under perturbations.

#### 2. **Few-Shot Learning**
Few-shot learning experiments were conducted to test the models' generalization capabilities with limited training data. The goal was to assess how well the models could learn from a small number of examples and still generalize effectively to unseen data.

#### 3. **Transfer Learning**
Transfer learning was explored by training the models on one synthetic dataset and then evaluating their performance on a different but related dataset. This experiment tested the adaptability of the models to new domains with similar characteristics.

### Benchmark Evaluation on UCR Datasets

The final evaluation involved a comprehensive benchmark on real-world datasets from the **UCR Time Series Classification Archive**. This archive includes a diverse set of time series datasets across various domains, providing a robust test bed for time series classification algorithms.

#### **Experimental Setup**
- **Datasets**: A selection of datasets from the UCR archive was chosen to represent different types of time series, including those with varying lengths, class distributions, and levels of difficulty.
- **Training and Testing**: Each dataset was divided into predefined training and testing sets, as provided by the UCR archive. The models were trained on the training sets and evaluated on the test sets.
- **Metrics**: Classification accuracy was used as the primary metric for evaluation, along with an analysis of the preservation of causal dynamics using the Kernel Granger Causality method.

#### **Results**
- **Accuracy**: The models were compared based on their classification accuracy across different UCR datasets. Neurochaos Learning consistently showed strong performance, particularly on datasets where preserving causal dynamics is crucial.
- **Causality Preservation**: The preservation of causal relationships was analyzed by comparing the causal structures in the original time series data with those in the predictions made by the models. Neurochaos Learning was found to be superior in maintaining these relationships.

## Methods and Tools

### Algorithms
- **Neurochaos Learning (ChaosNet, ChaosFEX+ML)**
- **Multilayer Perceptron (MLP)**
- **Kolmogorov-Arnold Networks (KAN)**

### Libraries
- **Python**: The core language used for the implementation.
- **NumPy**: For numerical computations.
- **SciPy**: For scientific computing tasks.
- **scikit-learn**: For implementing MLP and other machine learning utilities.
- **TensorFlow/Keras**: For building and training the deep learning models (MLP and KAN).
- **Matplotlib/Seaborn**: For data visualization and analysis.
- **pandas**: For data manipulation and analysis.
- **Neurochaos**: A custom library for implementing Neurochaos Learning methods (if applicable).

### Datasets
- **Synthetic Data**: Generated using coupled skew-tent maps and ARMA models to simulate various levels of causal dependency.
- **UCR Time Series Classification Archive**: Real-world datasets used to test the generalization of the models.

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/username/Neurochaos-TimeSeries.git
    cd Neurochaos-TimeSeries
    ```

2. **Install Dependencies**:
    Make sure you have Python 3.8+ installed. Then, install the necessary packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Experiments**:
    - To run experiments on synthetic data:
        ```bash
        python run_synthetic_experiments.py
        ```
    - To evaluate models on UCR datasets:
        ```bash
        python run_ucr_experiments.py
        ```

4. **Analyze Results**:
    - Results will be saved in the `results/` directory. You can visualize and analyze them using provided Jupyter notebooks.

## Results

- **Accuracy and Causality Preservation**: The results indicate that Neurochaos Learning outperforms traditional methods in preserving causal dynamics while maintaining high accuracy.
- **Generalization and Robustness**: Models trained with Neurochaos Learning show improved generalization on unseen datasets, especially under noisy conditions.

## Future Work

This study opens up several avenues for future research:
- **Extending Neurochaos Learning**: Exploring its application to multivariate time series.
- **Hybrid Models**: Combining Neurochaos with other deep learning architectures.
- **Real-Time Applications**: Adapting these models for real-time time series analysis.

## Acknowledgments

I would like to thank my advisors, Prof. Giacomo Boracchi and Prof. Harikrishnan N. B., for their guidance and support throughout this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

