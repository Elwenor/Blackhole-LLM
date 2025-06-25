### **Blackhole-LLM: A Groundbreaking Architecture for Next-Generation LLMs**

Welcome to **Blackhole-LLM** – an experimental venture into the realm of **advanced Large Language Model (LLM) architectures**! Built upon `PyTorch`, this project aims to revolutionize how LLMs process both **textual and numerical data**, with a strong emphasis on robust mathematical reasoning, intelligent handling of structured inputs, and an overarching commitment to system modularity.

---

### **A Future on Hugging Face: Our Integration Vision**

From its inception, Blackhole-LLM is being developed with a clear goal: **to seamlessly integrate all its innovative components, including our custom tokenizer and numerical embedding modules, directly into the Hugging Face ecosystem.** This commitment ensures broad accessibility, ease of use, and a valuable contribution to the open-source community, while laying the groundwork for the next generation of powerful LLMs.

---

> [!WARNING]
>
> ### ⚠️ Project Status - Important Information! ⚠️
>
> This project is currently in an **active architectural development phase**. While the **key innovative components—the custom tokenizer system and the numerical embeddings module—are functional and being actively refined**, their design and implementation are still subject to **ongoing improvements and potential significant changes**. This means their current state, though operational, is **experimental and not yet optimized for general utility or stability.**
>
> Test implementations of the core language model, which will leverage these innovative components, are under development.
>
> Blackhole-LLM is made public for transparency and to showcase novel architectural solutions. **However, it is not yet intended for production use or for independent execution by external users.** Its primary purpose at this stage is to demonstrate a conceptual approach to LLM architecture.
>
> To validate the innovative components, **internal benchmarks and unit tests** have been prepared that compare the performance of the unique Tokenizer against solutions like GPT-2 Tokenizer and BERT.

---

### Key Architectural Features

The Blackhole-LLM architecture stands out with the following innovations:

* **Innovative Tokenizer (BlackholeTokenizer)**: A custom extension of `PreTrainedTokenizerFast`, designed for efficient handling of numerical data, mathematical symbols, and structured input. It focuses on reducing vocabulary size while preserving semantic precision.

* **Dual Embedding Architecture**: A unique approach to data embedding that combines traditional textual embeddings with advanced numerical embeddings. This allows the model to gain a deeper understanding of both linguistic and quantitative contexts.

* **Modular Design**: The project is designed as a collection of independent yet closely integrated modules (`tokenizer`, `embedding`, `nova`), facilitating development, testing, and future expansion.

* **Focus on Numerical Data and Mathematics**: The architecture is optimized from the ground up for processing numerical data, making it ideal for applications requiring precise mathematical reasoning.

* **Internal Benchmarks and Unit Tests**: Integration of comprehensive tests and benchmarks for individual architectural components (e.g., tokenizer, embeddings) ensures their high quality and comparability.

---

### Core Components

The Blackhole-LLM project consists of several key packages and scripts that collectively build its architecture:

#### `blackhole/tokenizer/`

This directory contains the innovative tokenizer, responsible for text processing, recognition, and special handling of numerical data, symbols, and formatting.

* **BlackholeTokenizer**: Currently, there are two versions of the tokenizer.
    * The **local version** is maintained for isolated development and testing.
    * The **Hugging Face integrated version (`PreTrainedTokenizerFast`)** is the primary focus of ongoing development and will be the officially supported version for future integrations.

For detailed information on its operation, benefits, limitations, and the results of the **internal benchmarks**, please refer to: [**Tokenizer Details and Benchmarks**](https://github.com/Elwenor/Blackhole-LLM/blob/main/benchmark/TOKENIZER.md)

#### `blackhole/embedding/`

These modules are responsible for creating embeddings, including the advanced system for numerical data that transforms numbers into vectors understandable by the model.

* Learn more about the numerical embeddings architecture, its benefits, challenges, and future plans here: [**Numerical Embeddings Details and Benchmarks**](https://github.com/Elwenor/Blackhole-LLM/blob/main/benchmark/EMBEDDING.md)

* **Choice of Normalization and Loss Function**: Based on extensive internal simulations and benchmarks, **P6: Quantile Normalization** has been selected as the optimal method for transforming numerical values into embedding vectors. This choice was driven by a key challenge: preventing extreme numerical values from destabilizing the training process while ensuring **extremely high precision and reconstruction accuracy**.

    * **Formulas of considered normalizations**:
        * **P1 (Raw)**: No normalization.
            $$x' = x$$

        * **P2 (Z-Score)**: Standardizes values.
            $$x' = \frac{x - \mu}{\sigma}$$

        * **P3 (Min-Max)**: Scales values to a range of 0 to 1.
            $$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

        * **P4 (Log + Min-Max)**: Applies a logarithmic transformation.
            $$f = \log_{10}(x + c)$$ $$f' = \frac{f - f_{min}}{f_{max} - f_{min}}$$

        * **P5 (Signed Log + Min-Max)**: Applies a signed logarithmic transformation.
            $$f = \text{sgn}(x) \cdot \log_{10}(|x| + 1)$$ $$f' = \frac{f - f_{min}}{f_{max} - f_{min}}$$

        * **P6 (Quantile)**: Maps values to percentiles.
            $$q = F(x) = P(X \leq x)$$ $$x' = q$$

        * **P7 (Quantile + Signed Log)**: A combination of quantile mapping and signed logarithmic transformation.
            $$f = \text{sgn}(x) \cdot \log_{10}(|x| + 1)$$ $$f' = F(f) = P(F \leq f)$$

    * **Why P6: Quantile Normalization Was Chosen**:
        After comprehensive testing, **P6: Quantile Normalization emerged as the clear winner.** While other methods like P1, P2, and P3 struggled with numerical stability and were highly sensitive to outliers, simulations showed that they would lead to training instability. Log-based methods (P4, P5) performed much better in terms of stability, but they introduced small yet unacceptable reconstruction errors, which is critical for a model designed for precise mathematical tasks.

        **Quantile Normalization (P6)**, despite being computationally more expensive during inference, offers unparalleled stability and, crucially, **almost perfect reconstruction of the original numerical value**. The minimal increase in cumulative loss in some scenarios is a small price to pay for the ability to maintain perfect data integrity, which is essential for accurate mathematical reasoning.

    * **Examples of P6's Stability and Accuracy**:
        * **Value: `1e+12`**: This extreme value causes massive loss in simpler models (e.g., Raw: `1.66e+24`), but with P6, the **loss is a stable `2.77e+0`**, and the reconstructed value has a **zero error**.
        * **Value: `-500000`**: Again, this large negative value destabilizes most methods, but P6 maintains a **low loss of `2.39e+0`** and **zero reconstruction error**.
        * **Value: `100`**: For a standard integer, P6 achieves the **lowest loss of all tested methods (`1.03e+0`)** and **perfect reconstruction**.

    This confirms that the choice of P6 offers the optimal balance between performance and the critical need for a stable training process and high-precision outputs, which is the core of the Blackhole-LLM architecture.

    * **Loss Function**: For the numeric prediction head, the **Mean Squared Error (MSE)** loss is employed, which measures the squared difference between the predicted and target numeric embeddings, ensuring the model learns to represent numerical values accurately.

#### `blackhole/nova/` - The Core Model Architecture

This directory is the heart of the Blackhole-LLM project, housing the core language model architecture and the training pipeline that integrates both textual and numerical embeddings. It is here that the tokenized input and the advanced numerical embeddings converge to form a complete, end-to-end training process.

##### Initial Training and a Critical Challenge

First-run tests were conducted on a compact 25-million-parameter model using a very limited dataset. These initial runs were not aimed at achieving linguistic proficiency but rather at validating the architectural integrity and stability of the dual-embedding system. The results were twofold:

1.  **Basic Functionality Confirmed**: The model successfully demonstrated its ability to generate tokens and, most importantly, correctly identify and process the logits corresponding to numerical tokens (`[number]`). This is a crucial first step, confirming that the dual embedding system is fundamentally operational. As expected from such a short training period, the generated sequence, like `tops rehearsingeat popular promoted epil`, is currently nonsensical. This is a normal and anticipated outcome, as the model has not had enough exposure to data to learn meaningful patterns.

2.  **A Critical Instability Highlighted**: The primary goal of these initial runs was to test model stability, which revealed a major challenge with the numerical loss. The total loss is a weighted sum of the language model loss and the numerical loss, calculated by the formula:

    $$ \text{total\_loss} = \text{total\_loss}_{\text{LM}} + (\text{self.config.numeric\_loss\_weight} \cdot \text{numeric\_loss}) $$

    While the language model loss (`LM Loss`) showed a healthy, downward trend, the `numeric_loss` was plagued by **extreme spikes and instability**. This is starkly visible in the Epoch 1 training summary:

    **Epoch 1 Training Summary:**
    * **Avg Total Loss:** **29,982.98**
    * **Avg LM Loss:** **2.43**

    The sheer scale of the `Avg Total Loss`—orders of magnitude higher than the `Avg LM Loss`—underscores the **extreme instability of the old numerical loss**. This behavior was a direct consequence of using the highly sensitive **P1 (Raw)** and **P2 (Z-Score)** normalization methods, which are easily destabilized by outliers in numerical data.

##### Moving Towards Stability: The Quantile Normalization Solution

The observed instability validated our initial hypothesis: for precise mathematical reasoning, a more robust normalization method is essential. Following these initial tests, a subsequent, smaller training run using the **P2 (Z-Score)** method showed a significant improvement in loss stability. This confirms that the integration of a more sophisticated normalization scheme is the correct path forward.

The team is now meticulously implementing **P6: Quantile Normalization** into the training pipeline. Based on our internal benchmarks, this method offers unparalleled stability and precision, and its full integration is expected to eliminate the loss spikes and pave the way for stable, full-scale training.

##### Model Configuration (`NovaModel`)

The configuration of the `NovaModel` used in these initial tests is detailed below. This architecture is designed to be modular and easily scalable, with specific parameters dedicated to integrating numerical embeddings.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `vocab_size` | `len(tokenizer)` | The size of the tokenizer's vocabulary. |
| `hidden_size` | `256` | The dimensionality of the encoder and decoder layers. |
| `num_hidden_layers` | `2` | The number of hidden layers in the Transformer blocks. |
| `num_attention_heads` | `4` | The number of attention heads in the multi-head attention mechanism. |
| `intermediate_size` | `512` | The size of the intermediate feed-forward network layer. |
| `max_position_embeddings` | `MAX_LENGTH` | The maximum sequence length the model can handle. |
| `pad_token_id` | `tokenizer.pad_token_id` | The ID of the padding token. |
| `bos_token_id` | `tokenizer.bos_token_id` | The ID of the beginning-of-sequence token. |
| `eos_token_id` | `tokenizer.eos_token_id` | The ID of the end-of-sequence token. |
| `decoder_start_token_id` | `tokenizer.bos_token_id` | The token ID to use as the decoder's start token. |
| `num_token_id` | `num_token_id` | The special token ID for numerical data, retrieved from the tokenizer. |
| `numeric_input_features` | `6` | The number of features extracted for numerical embedding. |
| `numeric_projection_intermediate_size` | `256` | The size of the intermediate projection layer for numerical embeddings. |
| `numeric_embedding_fusion_type` | `'gating'` | The method used to fuse numerical and textual embeddings. |
| `numeric_heavy_feature_freeze` | `False` | A flag to control freezing of numerical feature layers during training. |
| `numeric_head_output_size` | `6` | **CHANGED**: The model is configured to predict all 6 numerical features. |
| `is_encoder_decoder` | `True` | Indicates that the model uses an encoder-decoder architecture. |
| `numeric_loss_weight` | `0.1` | The weight applied to the numerical loss component. |

#### `scripts/`

This directory contains various scripts for project management, including:

* Unit tests (`scripts/tests/`).
* Benchmark scripts (`scripts/benchmarks/`).
* Model training and evaluation scripts (under development).

---

### Future Development Plans

The long-term goal is to build a full, effective LLM that fully leverages the capabilities of this innovative tokenizer and embedding architecture. Subsequent development stages include:

* Further development and optimization of the main model architecture (`NovaModel`).
* Implementation and refinement of the complete LLM training process, utilizing dual embeddings.
* Adding advanced evaluation and prediction functionalities for the entire model.
* Integration with larger datasets and real-world NLP tasks.

---

### License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

---

💡 **Note**: As an architectural project focused on innovation, there are no immediate installation or execution instructions provided in this `README.md`. The detailed design and benchmark results for the tokenizer and embeddings are available in their respective documentation files.