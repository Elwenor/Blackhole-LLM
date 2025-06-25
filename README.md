# Blackhole-LLM: A Groundbreaking Architecture for Next-Generation LLMs

Welcome to **Blackhole-LLM** – my experimental venture into the realm of **advanced Large Language Model (LLM) architectures**\! Built upon `PyTorch`, this project aims to revolutionize how LLMs process both **textual and numerical data**, with a strong emphasis on robust mathematical reasoning, intelligent handling of structured inputs, and an overarching commitment to system modularity.

---

### **A Future on Hugging Face: Our Integration Vision**

From its inception, Blackhole-LLM is being developed with a clear goal: **to seamlessly integrate all its innovative components, including our custom tokenizer and numerical embedding modules, directly into the Hugging Face ecosystem.** This commitment ensures broad accessibility, ease of use, and a valuable contribution to the open-source community, while laying the groundwork for the next generation of powerful LLMs.

---

> [\!WARNING]
>
> ### ⚠️ Project Status - Important Information\! ⚠️
>
> This project is currently in an **active architectural development phase**. While my **key innovative components—the custom tokenizer system and the numerical embeddings module—are functional and being actively refined**, their design and implementation are still subject to **ongoing improvements and potential significant changes**. This means their current state, though operational, is **experimental and not yet optimized for general utility or stability.**
>
> Test implementations of the core language model, which will leverage these innovative components, are under development.
>
> Blackhole-LLM is made public for transparency and to showcase novel architectural solutions. **However, it is not yet intended for production use or for independent execution by external users.** Its primary purpose at this stage is to demonstrate a conceptual approach to LLM architecture.
>
> To validate my innovative components, I've prepared **internal benchmarks and unit tests** that compare the performance of my unique Tokenizer against solutions like GPT-2 Tokenizer and BERT.

---

### Key Architectural Features

My Blackhole-LLM architecture stands out with the following innovations:

* **Innovative Tokenizer (BlackholeTokenizer)**: A custom extension of `PreTrainedTokenizerFast`, designed for efficient handling of numerical data, mathematical symbols, and structured input. It focuses on reducing vocabulary size while preserving semantic precision.

* **Dual Embedding Architecture**: A unique approach to data embedding that combines traditional textual embeddings with advanced numerical embeddings. This allows the model to gain a deeper understanding of both linguistic and quantitative contexts.

* **Modular Design**: The project is designed as a collection of independent yet closely integrated modules (`tokenizer`, `embedding`, `nova`), facilitating development, testing, and future expansion.

* **Focus on Numerical Data and Mathematics**: The architecture is optimized from the ground up for processing numerical data, making it ideal for applications requiring precise mathematical reasoning.

* **Internal Benchmarks and Unit Tests**: Integration of comprehensive tests and benchmarks for individual architectural components (e.g., tokenizer, embeddings) ensures their high quality and comparability.

---

### Core Components

My Blackhole-LLM project consists of several key packages and scripts that collectively build its architecture:

#### `blackhole/tokenizer/`

This directory contains my innovative tokenizer, responsible for text processing, recognition, and special handling of numerical data, symbols, and formatting.

* **BlackholeTokenizer**: Currently, there are two versions of the tokenizer.
    * The **local version** is maintained for isolated development and testing.
    * The **Hugging Face integrated version (`PreTrainedTokenizerFast`)** is the primary focus of ongoing development and will be the officially supported version for future integrations.

For detailed information on its operation, benefits, limitations, and the results of my **internal benchmarks**, please refer to: [**Tokenizer Details and Benchmarks**](https://github.com/Elwenor/Blackhole-LLM/blob/main/benchmark/TOKENIZER.md)

#### `blackhole/embedding/`

These modules are responsible for creating embeddings, including my advanced system for numerical data that transforms numbers into vectors understandable by the model.

* Learn more about my numerical embeddings architecture, its benefits, challenges, and future plans here: [**Numerical Embeddings Details and Benchmarks**](https://github.com/Elwenor/Blackhole-LLM/blob/main/benchmark/EMBEDDING.md)

* **Choice of Normalization and Loss Function**: Based on our extensive internal simulations and benchmarks, we have selected the **Signed Log + Min-Max Normalization (Approach P5)** as the optimal method for transforming numerical values into embedding vectors. This choice was driven by a key challenge: preventing extreme numerical values from destabilizing the training process.

    * **Formula**: This approach applies a signed logarithmic transformation to compress the range of values while preserving their sign. The transformed values are then scaled using Min-Max normalization. The formula for the transformation is as follows:
        $$f = \text{sgn}(x) \cdot \log_{10}(|x| + 1)$$
        $$f' = \frac{f - f_{min}}{f_{max} - f_{min}}$$

    * **Why P5 is the Best Choice**:
        * **Robustness**: Our simulations showed that this method consistently achieves a very low cumulative loss across a wide range of values—from extreme magnitudes (`1e+9`, `1e+12`) to fractional, negative, and zero values. Unlike simpler normalizations, it effectively prevents feature values from dominating the training process.
        * **Computational Efficiency**: While not the fastest, its computational complexity is far superior to quantile-based methods (e.g., P6, P7), making it a highly practical choice for large-scale training and real-time inference. It offers an excellent balance between performance and computational cost.
        * **High Reconstructibility**: The chosen method allows for the accurate reconstruction of the original numerical value from its embedding, which is crucial for tasks requiring precise outputs (e.g., mathematical reasoning, data generation).
        * **Superiority over alternatives**:
            * **Raw and Z-Score**: These methods struggle with numerical stability, leading to huge loss values for extreme inputs.
            * **Min-Max**: While fast, it is highly sensitive to outliers, which can skew the entire feature range.
            * **Quantile-based**: Although stable, these methods are computationally expensive during inference, as they require a lookup table or a full dataset scan.

    * **Loss Function**: For the numeric prediction head, we employ the **Mean Squared Error (MSE)** loss, which measures the squared difference between the predicted and target numeric embeddings, ensuring the model learns to represent numerical values accurately.

#### `blackhole/nova/`

This is the designated location for the core language model architecture and training-related modules (e.g., a Transformer class and training scripts). It integrates tokens and numerical embeddings to form the complete training pipeline.

#### `scripts/`

This directory contains various scripts for project management, including:

* Unit tests (`scripts/tests/`).
* Benchmark scripts (`scripts/benchmarks/`).
* Model training and evaluation scripts (under development).

---

### Future Development Plans

My long-term goal is to build a full, effective LLM that fully leverages the capabilities of my innovative tokenizer and embedding architecture. Subsequent development stages include:

* Further development and optimization of the main model architecture (`NovaModel`).
* Implementation and refinement of the complete LLM training process, utilizing dual embeddings.
* Adding advanced evaluation and prediction functionalities for the entire model.
* Integration with larger datasets and real-world NLP tasks.

---

### License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

---

💡 **Note**: As an architectural project focused on innovation, there are no immediate installation or execution instructions provided in this `README.md`. The detailed design and benchmark results for the tokenizer and embeddings are available in their respective documentation files.