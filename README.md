Blackhole-LLM: An Innovative Architecture for Next-Generation LLMs
Blackhole-LLM is an experimental Python project focused on developing and refining an advanced architecture for Large Language Models (LLMs). Leveraging PyTorch, it aims to revolutionize how both textual and numerical data are processed, with a strong emphasis on mathematical reasoning, structured input handling, and the overall modularity of the system.

Project Status
This project is currently in an active architectural development phase. At present, the key innovative components—the custom tokenizer system and the numerical embeddings module—are fully functional and undergoing intensive refinement. Test implementations of the core language model, which will leverage these innovative components, are under development. Blackhole-LLM is made public for transparency and to showcase novel architectural solutions, but it is not yet intended for production use or for independent execution by external users.

To validate our innovative components, we've prepared internal benchmarks and unit tests that compare the performance of our unique Tokenizer against solutions like GPT-2 Tokenizer and BERT.

Key Architectural Features
Our Blackhole-LLM architecture stands out with the following innovations:

Innovative Tokenizer: A custom extension of GPT2TokenizerFast, designed for efficient handling of numerical data, mathematical symbols, and structured input. It focuses on reducing vocabulary size while preserving semantic precision.
Dual Embedding Architecture: A unique approach to data embedding that combines traditional textual embeddings with advanced numerical embeddings. This allows the model to gain a deeper understanding of both linguistic and quantitative contexts.
Modular Design: The project is designed as a collection of independent yet closely integrated modules (tokenizer, embedding, nova), facilitating development, testing, and future expansion.
Focus on Numerical Data and Mathematics: The architecture is optimized from the ground up for processing numerical data, making it ideal for applications requiring precise mathematical reasoning.
Internal Benchmarks and Unit Tests: Integration of comprehensive tests and benchmarks for individual architectural components (e.g., tokenizer, embeddings) ensures their high quality and comparability.
Core Components
The Blackhole-LLM project consists of several key packages and scripts that collectively build our architecture:

blackhole/tokenizer/: Contains the innovative tokenizer, responsible for text processing, recognition, and special handling of numerical data, symbols, and formatting.
For detailed information on its operation, benefits, limitations, and the results of our internal benchmarks, please refer to: Tokenizer Details and Benchmarks
blackhole/embedding/: Modules responsible for creating embeddings, including an advanced system for numerical data that transforms numbers into vectors understandable by the model.
Learn more about our numerical embeddings architecture, its benefits, challenges, and future plans here: Numerical Embeddings Details and Benchmarks
blackhole/nova/: The designated location for the core language model architecture (e.g., a Transformer class) that will integrate tokens and numerical embeddings.
scripts/: A directory containing scripts for project management, including:
Unit tests (scripts/tests/).
Benchmark scripts (scripts/benchmarks/).
Model training and evaluation scripts (under development).
Future Development Plans
Our long-term goal is to build a full, effective LLM that fully leverages the capabilities of our innovative tokenizer and embedding architecture. Subsequent development stages include:

Further development and optimization of the main model architecture (NovaModel).
Implementation and refinement of the complete LLM training process, utilizing dual embeddings.
Adding advanced evaluation and prediction functionalities for the entire model.
Integration with larger datasets and real-world NLP tasks.
License
This project is licensed under the MIT License.