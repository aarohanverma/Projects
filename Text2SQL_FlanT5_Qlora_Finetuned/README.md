---
license: apache-2.0
datasets:
- Clinton/Text-to-sql-v1
- b-mc2/sql-create-context
- gretelai/synthetic_text_to_sql
- knowrohit07/know_sql
metrics:
- rouge
- bleu
- fuzzy_match
- exact_match
base_model:
- google/flan-t5-base
pipeline_tag: text2text-generation
library_name: transformers
language:
- en
tags:
- text2sql
- transformers
- flan-t5
- seq2seq
- qlora
- peft
- fine-tuning
---
# Model Card

<!-- Provide a quick summary of what the model is/does. -->

This model is a fine-tuned version of [Flan-T5 Base](https://huggingface.co/google/flan-t5-base) optimized to convert natural language queries into SQL statements. It leverages **QLoRA (Quantized Low-Rank Adaptation)** with PEFT for efficient adaptation and has been trained on a concatenation of several high-quality text-to-SQL datasets. A live demo is available, and users can clone and run inference directly from Hugging Face.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is designed to generate SQL queries based on a provided natural language context and query. 
It has been fine-tuned using QLoRA with 4-bit quantization and PEFT on a diverse text-to-SQL dataset. 
The model demonstrates significant improvements over the original base model, making it highly suitable for practical text-to-SQL applications.

- **Developed by:** Aarohan Verma
- **Model type:** Seq2Seq / Text-to-Text Generation (SQL Generation)
- **Language(s) (NLP):** English
- **License:** Apache-2.0
- **Finetuned from model:** [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)


### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [https://huggingface.co/aarohanverma/text2sql-flan-t5-base-qlora-finetuned](https://huggingface.co/aarohanverma/text2sql-flan-t5-base-qlora-finetuned)
- **Demo:** [Gradio Demo](https://huggingface.co/spaces/aarohanverma/text2sql-demo)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model can be used directly for generating SQL queries from natural language inputs. 
It is particularly useful for applications in database querying and natural language interfaces for relational databases.

### Downstream Use 

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The model can be further integrated into applications such as chatbots, data analytics platforms, and business intelligence tools to automate query generation.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model is not designed for tasks outside text-to-SQL generation. 
It may not perform well for non-SQL language generation or queries outside the domain of structured data retrieval.

## Risks and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Risks:** Inaccurate SQL generation may lead to unexpected query behavior, especially in safety-critical environments.
- **Limitations:** The model may struggle with highly complex schemas and queries due to limited training data and inherent model capability constraints, particularly for tasks that require deep, domain-specific knowledge.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should validate the generated SQL queries before deployment in production systems. 
Consider incorporating human-in-the-loop review for critical applications.

## How to Get Started with the Model

To get started, clone the repository or download the model from Hugging Face, then use the provided example code to run inference. 
Detailed instructions and the live demo are available in this model card.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model was fine-tuned on a concatenation of several publicly available text-to-SQL datasets:
1. **[Clinton/Text-to-SQL v1](https://huggingface.co/datasets/Clinton/Text-to-sql-v1)**
2. **[b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)**
3. **[gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)**
4. **[knowrohit07/know_sql](https://huggingface.co/datasets/knowrohit07/know_sql)**

**Data Split:**

| **Split**            | **Percentage**     | **Number of Samples**    |
|----------------------|--------------------|--------------------------|
| **Training**         | 85%                | **338,708**              |
| **Validation**       | 5%                 | **19,925**               |
| **Testing**          | 10%                | **39,848**               |


### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

The raw data was preprocessed as follows:
- **Cleaning:** Removal of extra whitespaces/newlines and standardization of columns (renaming to `query`, `context`, and `response`).
- **Filtering:** Dropping examples with missing values and duplicates; retaining only rows where the prompt is ≤ 500 tokens and the response is ≤ 250 tokens.
- **Tokenization:**

Prompts are constructed in the format:
```
Context:
{context}

Query:
{query}

Response:
```
and tokenized with a maximum length of 512 for inputs and 256 for responses using [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)'s tokenizer.

#### Training Hyperparameters

- **Epochs:** 6  
- **Batch Sizes:**  
  Training: 64 per device  
  Evaluation: 64 per device  
- **Gradient Accumulation:** 2 steps  
- **Learning Rate:** 2e-4  
- **Optimizer:** `adamw_bnb_8bit` (memory-efficient variant of AdamW)  
- **LR Scheduler:** Cosine scheduler with a warmup ratio of 10%  
- **Quantization:** 4-bit NF4 (with double quantization) using `torch.bfloat16`  
- **LoRA Parameters:**  
  **Rank (r):** 32  
  **Alpha:** 64  
  **Dropout:** 0.1  
  **Target Modules:** `["q", "v"]`  
- **Checkpointing:**  
  Model saved at the end of every epoch  
  Early stopping with a patience of 2 epochs based on evaluation loss  
- **Reproducibility:** Random seeds are set across Python, NumPy, and PyTorch (seed = 42)

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

The model was evaluated on 39,848 test samples, representing 10% of the dataset.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Evaluation metrics used:
- **ROUGE:** Measures n-gram overlap between generated and reference SQL.
- **BLEU:** Assesses the quality of translation from natural language to SQL.
- **Fuzzy Match Score:** Uses token-set similarity to provide a soft match percentage.
- **Exact Match Accuracy:** Percentage of queries that exactly match the reference SQL.

### Results

The table below summarizes the evaluation metrics comparing the original base model with the fine-tuned model:

| **Metric**                | **Original Model**            | **Fine-Tuned Model**    | **Comments**                                                                         |
|---------------------------|-------------------------------|-------------------------|--------------------------------------------------------------------------------------|
| **ROUGE-1**               | 0.05647                       | **0.75388**             | Over 13× increase; indicates much better content capture.                            |
| **ROUGE-2**               | 0.01563                       | **0.61039**             | Nearly 39× improvement; higher n-gram quality.                                       |
| **ROUGE-L**               | 0.05031                       | **0.72628**             | More than 14× increase; improved sequence similarity.                                |
| **BLEU Score**            | 0.00314                       | **0.47198**             | Approximately 150× increase; demonstrates significant fluency gains.                 |
| **Fuzzy Match Score**     | 13.98%                        | **85.62%**              | Substantial improvement; generated SQL aligns much closer with human responses.      |
| **Exact Match Accuracy**  | 0.00%                         | **18.29%**              | Non-zero accuracy achieved; critical for production-readiness.                       |

#### Summary

The fine-tuned model shows dramatic improvements across all evaluation metrics, proving its effectiveness in generating accurate and relevant SQL queries from natural language inputs.

## 🔍 Inference & Example Usage

### Inference Code
Below is the recommended Python code for running inference on the fine-tuned model:

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure device is set (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model and tokenizer
model_name = "aarohanverma/text2sql-flan-t5-base-qlora-finetuned"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("aarohanverma/text2sql-flan-t5-base-qlora-finetuned")

# Ensure decoder start token is set
if model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = tokenizer.pad_token_id

def run_inference(prompt_text: str) -> str:
    """
    Runs inference on the fine-tuned model using beam search with fixes for repetition.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        decoder_start_token_id=model.config.decoder_start_token_id, 
        max_new_tokens=100,  
        temperature=0.1,  
        num_beams=5,  
        repetition_penalty=1.2,  
        early_stopping=True, 
    )

    generated_sql = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    generated_sql = generated_sql.split(";")[0] + ";"  # Keep only the first valid SQL query

    return generated_sql

# Example usage:
context = (
    "CREATE TABLE students (id INT PRIMARY KEY, name VARCHAR(100), age INT, grade CHAR(1)); "
    "INSERT INTO students (id, name, age, grade) VALUES "
    "(1, 'Alice', 14, 'A'), (2, 'Bob', 15, 'B'), "
    "(3, 'Charlie', 14, 'A'), (4, 'David', 16, 'C'), (5, 'Eve', 15, 'B');"
)

query = ("Retrieve the names of students who are 15 years old.")


# Construct the prompt
sample_prompt = f"""Context:
{context}

Query:
{query}

Response:
"""

logger.info("Running inference with beam search decoding.")
generated_sql = run_inference(sample_prompt)

print("Prompt:")
print("Context:")
print(context)
print("\nQuery:")
print(query)
print("\nResponse:")
print(generated_sql)

# EXPECTED RESPONSE: SELECT name FROM students WHERE age = 15; 
```

## Citation 

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@misc {aarohan_verma_2025,
	author       = { {Aarohan Verma} },
	title        = { text2sql-flan-t5-base-qlora-finetuned },
	year         = 2025,
	url          = { https://huggingface.co/aarohanverma/text2sql-flan-t5-base-qlora-finetuned },
	doi          = { 10.57967/hf/4887 },
	publisher    = { Hugging Face }
}
```

## Model Card Contact

For inquiries or further information, please contact:

LinkedIn: https://www.linkedin.com/in/aarohanverma/

Email: verma.aarohan@gmail.com