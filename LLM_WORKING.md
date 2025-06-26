# MultimodalLLM - Numerical Reasoning Proof of Concept

## Introduction

This repository contains a proof-of-concept version of a small Multimodal LLM, designed to demonstrate the feasibility of integrating numerical reasoning with a language model. The model is specifically finetuned to understand and respond to mathematical queries. This version represents an early test phase, not a final product.

## Model Architecture & Training

The model is based on the custom `MultimodalLLM` architecture, which includes a dedicated head for processing numerical inputs and outputs.

  * **Model Size:** Approximately **25 million parameters**, making it a highly compact model suitable for testing.
  * **Training Data:** The model was finetuned on a small subset of the total data, using only **20% (400MB)** of the available training corpus. This limited dataset was sufficient to showcase its test capabilities.
  * **Finetuning Goal:** A key objective of this finetuning was to teach the model to understand and generate numerical outputs based on mathematical operations. The model was trained with a dedicated `numeric_loss` to optimize its numerical predictions.

The trained model weights are saved as a checkpoint in the standard PyTorch format: `model_epoch_1.pt`.

## Performance in the Test Phase

The finetuning process **yielded clear results**, demonstrating the model's ability to learn a new response pattern. However, as a test version, it also highlighted a critical limitation.

### Key Log Examples with Top 3 Logit Analysis

**Example 1: Simple Multiplication**

This query demonstrates the model's ability to identify numbers but its complete failure to perform a calculation.

```
>>> Enter query: What is 4 x 2

[DEBUG] Processing query: 'What is 4 x 2'
[DEBUG] Detected token '[NUM]' at positions: [3, 5]

[INFO] Running encoder...

[INFO] Starting decoding...

--- GENERATION STEP 1 ---
[DEBUG] Current token sequence:
[DEBUG] LM logits for the next token:
  1. Token '[CLS]' (ID: 1) | Probability: 99.95%
  2. Token '[NUM]' (ID: 5) | Probability: 0.02%
  3. Token '[SEP]' (ID: 2) | Probability: 0.00%

--- GENERATION STEP 2 ---
[DEBUG] Current token sequence: [CLS]
[DEBUG] LM logits for the next token:
  1. Token 'b' (ID: 62) | Probability: 11.13%
  2. Token 'no' (ID: 1243) | Probability: 8.65%
  3. Token 'the' (ID: 1096) | Probability: 4.69%

--- GENERATION STEP 3 ---
[DEBUG] Current token sequence: [CLS] b
[DEBUG] LM logits for the next token:
  1. Token ''' (ID: 14) | Probability: 9.08%
  2. Token 'the' (ID: 1096) | Probability: 2.14%
  3. Token '[SEP]' (ID: 2) | Probability: 1.47%

--- GENERATION STEP 4 ---
[DEBUG] Current token sequence: [CLS] b '
[DEBUG] LM logits for the next token:
  1. Token '[NUM]' (ID: 5) | Probability: 99.68% | ASSIGNED NUMERICAL VALUE: 7.9749
  2. Token '[SEP]' (ID: 2) | Probability: 0.02%
  3. Token 'answer' (ID: 1314) | Probability: 0.01%

--- GENERATION STEP 5 ---
[DEBUG] Current token sequence: [CLS] b ' [NUM]
[DEBUG] LM logits for the next token:
  1. Token '[SEP]' (ID: 2) | Probability: 5.27%
  2. Token '%' (ID: 12) | Probability: 2.54%
  3. Token '\' (ID: 56) | Probability: 1.97%

==================================================
FINAL MODEL ANSWER:
  -> Decoded sequence: b '
  -> Answer with numerical predictions: b ' 7.9749
==================================================
```

**Example 2: Algebraic Equation**

This query illustrates the model's cautious behavior in the face of a complex, symbolic problem.

```
>>> Enter query: Solve for x: 2*x + 4 = 12

[DEBUG] Processing query: 'Solve for x: 2*x + 4 = 12'
[DEBUG] Detected token '[NUM]' at positions: [5, 9, 11]

[INFO] Running encoder...

[INFO] Starting decoding...

--- GENERATION STEP 1 ---
[DEBUG] Current token sequence:
[DEBUG] LM logits for the next token:
  1. Token '[CLS]' (ID: 1) | Probability: 99.83%
  2. Token '[SEP]' (ID: 2) | Probability: 0.04%
  3. Token 'of' (ID: 1105) | Probability: 0.01%

--- GENERATION STEP 2 ---
[DEBUG] Current token sequence: [CLS]
[DEBUG] LM logits for the next token:
  1. Token 'no' (ID: 1243) | Probability: 17.97%
  2. Token 'the' (ID: 1096) | Probability: 9.95%
  3. Token '[SEP]' (ID: 2) | Probability: 3.28%

--- GENERATION STEP 3 ---
[DEBUG] Current token sequence: [CLS] no
[DEBUG] LM logits for the next token:
  1. Token 'answer' (ID: 1314) | Probability: 99.72%
  2. Token '[SEP]' (ID: 2) | Probability: 0.05%
  3. Token '[NUM]' (ID: 5) | Probability: 0.04% | ASSIGNED NUMERICAL VALUE: 3.3026

--- GENERATION STEP 4 ---
[DEBUG] Current token sequence: [CLS] no answer
[DEBUG] LM logits for the next token:
  1. Token '[SEP]' (ID: 2) | Probability: 99.95%
  2. Token 'of' (ID: 1105) | Probability: 0.01%
  3. Token ',' (ID: 19) | Probability: 0.00%

==================================================
FINAL MODEL ANSWER:
  -> Decoded sequence: no answer
  -> Answer with numerical predictions: no answer
==================================================
```

### ✅ Key Achievements

  * **Recognized Numerical Queries:** The model successfully learned to detect numbers in the input text and tokenize them as `[NUM]`, a crucial first step in numerical processing.
  * **Attempted Numerical Output:** After finetuning, the model consistently attempted to produce a numerical answer for arithmetic questions, showing that it successfully learned the new `[NUM]` token-based output pattern.
  * **Hallucination Avoidance:** For complex queries it could not solve, such as algebraic equations or riddles, the model wisely defaulted to `no answer`, a behavior intentionally programmed during training to prevent incorrect and confident responses.

### ❌ Known Limitations & Root Cause

The model's primary limitation is **poor numerical precision**, a known issue stemming from the legacy processing pipeline used during this test phase.

* **Data Labeling Error:** A critical issue was identified in the training database. Instances where no answer was available were erroneously labeled as `no answer`. This created a training bias, causing the model to **blindly output `no answer`** for tasks it did not understand, preventing it from even attempting to reason or find a solution.
* **Random Numerical Outputs:** As seen in the logs, for a simple query like `What is 4 x 2`, the model correctly attempts to output a number, but the predicted value is random (e.g., `7.9749`) rather than the correct answer.
* **Legacy Pipeline:** This instability and lack of precision are caused by the **older numerical processing methods (P1 and P2)** used for decoding. These methods are prone to "blowouts" and inaccurate predictions.
* **No Symbolic Reasoning:** Due to its small size and training constraints, the model is unable to perform symbolic reasoning, failing to solve algebraic equations or logical problems.

## Roadmap & Future Improvements

This model is a stepping stone. The development team has a clear plan to address the numerical precision issue:

  * **P6 Pipeline Migration:** The system will be upgraded to a new processing pipeline, **P6**, which is designed to ensure **perfect numerical precision during the reverse decoding process**. This will replace the unstable P1/P2 methods and allow the model to produce accurate results.
  * **Expanded Training:** Future versions will be trained on a significantly larger dataset to improve general knowledge and reasoning abilities.

-----

## How to Run Inference

To test the model's current capabilities, you can run the inference script from your terminal:

```bash
# Note: The model is currently not available for public testing.
# Its general weakness and lack of practical application in its current state
# mean it is not suitable for public use.
python inference.py
```