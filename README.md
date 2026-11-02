# Fine-Tuning Small Language Models for Question Answering

Fine-tuning and comparing two small language models, **Qwen2.5-0.5B-Instruct** and **Llama 3.2-1B-Instruct**, on an AI question-answering dataset using **QLoRA**. Both models were trained with identical settings on a free Google Colab T4 GPU and evaluated on the same held-out test set.

## Dataset

[Chatbot Dataset (AI Q&A)](https://www.kaggle.com/datasets/yapwh1208/chatbot-ai-q-and-a) from Kaggle: English question–answer pairs about artificial intelligence (closed-book generative QA, no context passage).

| Split | Pairs |
| --- | --- |
| Train | 401 |
| Validation | 50 |
| Test | 51 |
| **Total (after cleaning)** | **502** |

## Method

- **Preprocessing:** removed missing values, extra whitespace and duplicate questions; 80/10/10 split (seed 42); each pair formatted with the model's chat template; prompt tokens masked so loss is computed only on the answer.
- **Fine-tuning:** QLoRA (4-bit NF4 base model + LoRA adapters)
  - LoRA r = 16, alpha = 32, dropout = 0.05 on q, k, v, o, gate, up, down projections
  - 3 epochs, learning rate 2e-4 (cosine), effective batch size 16, FP16, gradient checkpointing
  - Checkpoint with the lowest validation loss kept
- **Evaluation:** greedy decoding, max 50 new tokens; Exact Match, token F1, ROUGE-L and BERTScore.

## Results (test set, %)

| Model | Version | EM | F1 | ROUGE-L | BERTScore | Trainable params | Peak GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | Base | 0.00 | 12.36 | 13.48 | 85.16 | – | – |
| Qwen2.5-0.5B | QLoRA | 0.00 | 14.17 | 16.81 | 87.78 | 8.8 M (1.75%) | 2.59 GB |
| Llama 3.2-1B | Base | 0.00 | 11.97 | 12.82 | 84.81 | – | – |
| Llama 3.2-1B | QLoRA | 0.00 | **17.57** | **19.83** | **88.08** | 11.3 M (0.90%) | 3.77 GB |

**Key findings**

- Fine-tuned **Llama 3.2-1B performed best** on every metric and gained the most from fine-tuning (+5.60 F1 vs +1.81 for Qwen).
- **Qwen2.5-0.5B used ~31% less GPU memory**, making it the better choice for limited hardware.
- EM is 0 because reference answers are full sentences; the high BERTScore but low F1 shows answers are on-topic but often miss the specific fact, which is expected for closed-book QA with small models.
- Validation loss flattened after 1–2 epochs, showing mild overfitting on the small dataset.

## How to run

1. Open `QA_FineTuning.ipynb` in Google Colab and set **Runtime → Change runtime type → T4 GPU**.
2. Add your Hugging Face token in Colab **Secrets** as `HF_TOKEN` (Llama 3.2 requires accepting Meta's license on Hugging Face).
3. Upload the dataset CSV to Google Drive under `MyDrive/QA_FineTune/`.
4. Run all cells in order. Results, predictions, LoRA adapters and plots are saved to the same Drive folder.

## Tech stack

Python · PyTorch · Hugging Face Transformers · PEFT · bitsandbytes · Datasets · Evaluate (ROUGE, BERTScore) · scikit-learn · Google Colab

## Author

**Haider Abbas** · BS Software Engineering, University of Central Punjab, Lahore
[GitHub](https://github.com/Haiderabbas0080) · [LinkedIn](https://www.linkedin.com/in/haider-abbas-75b30118a)
