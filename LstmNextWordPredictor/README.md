# Model Card

<!-- Provide a quick summary of what the model is/does. -->

This repository contains an LSTM-based next word prediction model implemented in PyTorch. 
The model utilizes advanced techniques including an extra fully connected layer with ReLU and dropout, layer normalization, label smoothing loss, gradient clipping, and learning rate scheduling to improve performance. 
It also uses SentencePiece for subword tokenization.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The LSTM Next Word Predictor is designed to predict the next word or subword given an input sentence. 
The model is trained on a dataset provided in CSV format (with a 'data' column) and uses an LSTM network with many enhancements.

- **Developed by:** Aarohan Verma
- **Model type:** LSTM-based Next Word Prediction
- **Language(s) (NLP):** English 
- **License:** Apache-2.0

### Model Sources 

- **Repository:** https://huggingface.co/aarohanverma/lstm-next-word-predictor
- **Demo:** [LSTM Next Word Predictor Demo](https://huggingface.co/spaces/aarohanverma/lstm-next-word-predictor-demo)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model can be directly used for next word prediction in text autocompletion.

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The model can be fine-tuned for related tasks such as:
- Text generation.
- Language modeling for specific domains.
  
### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model is not suitable for:
- Tasks requiring deep contextual understanding beyond next-word prediction.
- Applications where transformer-based architectures are preferred for longer contexts.
- Sensitive applications where data bias could lead to unintended outputs.
  
## Risks and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Risks:** Inaccurate or unexpected predictions may occur if the input context is too complex or ambiguous.
- **Limitations:** The model’s performance is bounded by the size and quality of the training data as well as the inherent limitations of LSTM architectures in modeling long-range dependencies.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be aware of the above limitations and conduct appropriate evaluations before deploying the model in production. 
Consider further fine-tuning or additional data preprocessing if the model is applied in sensitive contexts.

## How to Get Started with the Model

Use the code below to get started with the model.

To run the model, follow these steps:

1. **Training:**
   - Ensure you have a CSV file with a column named `data` containing your training sentences.
   - Run training with:
     ```
     python next_word_prediction.py --data_path data.csv --train
     ```
   - This will train the model, save a checkpoint (`best_model.pth`), and export a TorchScript version (`best_model_scripted.pt`).

2. **Inference:**
   - To predict the next word, run:
     ```
     python next_word_prediction.py --inference "Your partial sentence"
     ```
   - The model will output the top predicted word or subword.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

- **Data Source:** CSV file with a column `data` containing sentences.
- **Preprocessing:** Uses SentencePiece for subword tokenization.
- **Dataset:** The training and validation datasets are split based on a user-defined ratio.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->


- **Preprocessing:** Tokenization using a SentencePiece model.
- **Training Hyperparameters:**
  - **Batch Size:** Configurable via `--batch_size` (default: 512)
  - **Learning Rate:** Configurable via `--learning_rate` (default: 0.001)
  - **Epochs:** Configurable via `--num_epochs` (default: 25)
  - **LSTM Parameters:** Configurable number of layers, dropout, and hidden dimensions.
  - **Label Smoothing:** Applied with a configurable factor (default: 0.1)
  - **Optimization:** Uses Adam optimizer with weight decay and gradient clipping.
  - **Learning Rate Scheduling:** ReduceLROnPlateau scheduler based on validation loss.

#### Speeds, Sizes, Times 

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

- **Checkpoint and TorchScript models** are saved during training for later inference.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

- **Testing Data:** Derived from the same CSV, split from the training data.
- **Metrics:** Primary metric is the loss (with label smoothing), with qualitative evaluation based on next-word accuracy.
- **Factors:** Evaluations may vary based on sentence length and dataset diversity.

#### Summary

- The model demonstrates promising performance on next word prediction tasks;
  however, quantitative results (e.g., accuracy, loss) should be validated on your specific dataset.

## Model Examination

<!-- Relevant interpretability work for the model goes here -->

- Interpretability techniques such as examining predicted token distributions can be applied to further understand model behavior.

## Technical Specifications 

### Model Architecture and Objective

- **Architecture:** LSTM-based network with enhancements such as an extra fully connected layer, dropout, and layer normalization.
- **Objective:** Predict the next word/subword given a sequence of tokens.

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@misc {aarohan_verma_2025,
	author       = { {Aarohan Verma} },
	title        = { lstm-next-word-predictor },
	year         = 2025,
	url          = { https://huggingface.co/aarohanverma/lstm-next-word-predictor },
	doi          = { 10.57967/hf/4882 },
	publisher    = { Hugging Face }
}
```

## Model Card Contact

For inquiries or further information, please contact:

LinkedIn: https://www.linkedin.com/in/aarohanverma/

Email: verma.aarohan@gmail.com