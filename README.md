# Blackhole-LLM: A Groundbreaking Architecture for Next-Generation LLMs

Welcome to **Blackhole-LLM** – my experimental venture into the realm of **advanced Large Language Model (LLM) architectures**! Built upon `PyTorch`, this project aims to revolutionize how LLMs process both **textual and numerical data**, with a strong emphasis on robust mathematical reasoning, intelligent handling of structured inputs, and an overarching commitment to system modularity.

---

### **A Future on Hugging Face: Our Integration Vision**

From its inception, Blackhole-LLM is being developed with a clear goal: **to seamlessly integrate all its innovative components, including our custom tokenizer and numerical embedding modules, directly into the Hugging Face ecosystem.** This commitment ensures broad accessibility, ease of use, and a valuable contribution to the open-source community, while laying the groundwork for the next generation of powerful LLMs.

---

> [!WARNING]
>
> ### ⚠️ Project Status - Important Information! ⚠️
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

* **Choice of Normalization and Loss Function**: Based on our extensive internal simulations and benchmarks, we have selected **P6: Quantile Normalization** as the optimal method for transforming numerical values into embedding vectors. This choice was driven by a key challenge: preventing extreme numerical values from destabilizing the training process while ensuring **extremely high precision and reconstruction accuracy**.

    * **Formulas of considered normalizations**:
        * **P1 (Raw)**: No normalization.
          $$x' = x$$

        * **P2 (Z-Score)**: Standardizes values.
          $$x' = \frac{x - \mu}{\sigma}$$

        * **P3 (Min-Max)**: Scales values to a range of 0 to 1.
          $$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

        * **P4 (Log + Min-Max)**: Applies a logarithmic transformation.
          $$f = \log_{10}(x + c)$$  $$f' = \frac{f - f_{min}}{f_{max} - f_{min}}$$

        * **P5 (Signed Log + Min-Max)**: Applies a signed logarithmic transformation.
          $$f = \text{sgn}(x) \cdot \log_{10}(|x| + 1)$$  $$f' = \frac{f - f_{min}}{f_{max} - f_{min}}$$

        * **P6 (Quantile)**: Maps values to percentiles.
          $$q = F(x) = P(X \leq x)$$  $$x' = q$$

        * **P7 (Quantile + Signed Log)**: A combination of quantile mapping and signed logarithmic transformation.
          $$f = \text{sgn}(x) \cdot \log_{10}(|x| + 1)$$  $$f' = F(f) = P(F \leq f)$$
          

    * **Why P6: Quantile Normalization Was Chosen**:
    After comprehensive testing, **P6: Quantile Normalization emerged as the clear winner.** While other methods like P1, P2, and P3 struggled with numerical stability and were highly sensitive to outliers, our simulations showed that they would lead to training instability. Log-based methods (P4, P5) performed much better in terms of stability, but they introduced small yet unacceptable reconstruction errors, which is critical for a model designed for precise mathematical tasks.

    **Quantile Normalization (P6)**, despite being computationally more expensive during inference, offers unparalleled stability and, crucially, **almost perfect reconstruction of the original numerical value**. The minimal increase in cumulative loss in some scenarios is a small price to pay for the ability to maintain perfect data integrity, which is essential for accurate mathematical reasoning.

    * **Examples of P6's Stability and Accuracy**:
        * **Value: `1e+12`**: This extreme value causes massive loss in simpler models (e.g., Raw: `1.66e+24`), but with P6, the **loss is a stable `2.77e+0`**, and the reconstructed value has a **zero error**.
        * **Value: `-500000`**: Again, this large negative value destabilizes most methods, but P6 maintains a **low loss of `2.39e+0`** and **zero reconstruction error**.
        * **Value: `100`**: For a standard integer, P6 achieves the **lowest loss of all tested methods (`1.03e+0`)** and **perfect reconstruction**.

    This confirms that the choice of P6 offers the optimal balance between performance and the critical need for a stable training process and high-precision outputs, which is the core of the Blackhole-LLM architecture.

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