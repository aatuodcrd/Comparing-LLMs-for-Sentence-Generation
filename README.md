# Comparing LLMs for Sentence Generation

## Overview

This project aims to compare various large language models (LLMs) for sentence generation tasks. The models are evaluated based on their ability to generate coherent and comprehensive sentences from given input texts.

## Models

The following models are used in this comparison:
- deepseek-r1-distill-llama-70b
- deepseek-r1-distill-qwen-32b
- llama-3.3-70b-specdec
- llama-3.2-11b-vision-preview
- llama-3.2-3b-preview
- llama-3.2-1b-preview

## Installation

To install the necessary packages, run the following command:
```bash
%pip install numpy matplotlib seaborn sentence-transformers groq scikit-learn -q
```

## Usage

1. Get your Groq API key from [Groq Console](https://console.groq.com/keys).

2. Set your Groq API key:
    ```python
    groq_api_key = "your_groq_api_key"
    ```

3. Define the models and input texts:
    ```python
    models = [
        "deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-qwen-32b",
        "llama-3.3-70b-specdec",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview"
    ]

    original_texts = [
        'วันนี้อากาศดีมาก', 'ฉันชอบอ่านหนังสือเกี่ยวกับปัญญาประดิษฐ์', ...
    ]
    ```

4. Run the comparison:
    ```python
    results, generated_embeddings, model_labels, cluster_labels, original_embeddings, clusters = compare_models_with_cosine(original_texts, models)
    ```

5. Visualize the results:
    ```python
    plot_accuracy_and_speed(results_preprocess)
    plot_clusters(results_preprocess)
    ```

## Functions

- `find_optimal_clusters(embeddings, max_clusters=10)`: Finds the optimal number of clusters based on silhouette scores.
- `generate_text(model_name, input_sentence_group)`: Generates text using the specified model.
- `is_text_in_center(new_text, mean_embedding, model)`: Checks if the generated text is close to the mean embedding.
- `compare_models_with_cosine(input_texts, models, max_clusters=100)`: Compares models using cosine similarity.

## Results

The results include the generated texts, their similarity to the input texts, and the time taken for generation. The optimal number of clusters is determined, and the texts are grouped accordingly.

## Visualization

The accuracy and speed of the models are visualized using bar plots. Clusters of input and generated texts are also plotted to show their distribution and similarity.

## Conclusion

This project provides a comprehensive comparison of different LLMs for sentence generation, helping to identify the most effective models for specific tasks.

## Cloning the Repository

To clone this repository, run the following command:
```bash
git clone https://github.com/aatuodcrd/Comparing-LLMs-for-Sentence-Generation.git
```
