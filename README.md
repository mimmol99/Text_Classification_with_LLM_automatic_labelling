# Text_Classification
Text classification using different techniques and an LLM as automatic label generator.


# First phase: load sentences
Load sentences from a document,one per line.

# Second phase: Choose an LLM
Use one LLM from OpenAI family,Groq or Claude to generate labels. Remember to modify the prompt (in utils.py/get_llm_labels) for your use-case and to add the proper API_KEY in the env file.

# Third phase: Try different techniques
Choose one or more techniques to classify the sentences.

- KNN with common train or HalvingRandomSearchCV
- RandomForest with common train or HalvingRandomSearchCV
- LSTM
- GRU
- BERT

# Last phase: comparison of results
Plot a graph to visualize results.

# Requirements

create a api_key.env file within the API_KEYS of the models you prefer to use.

then install necessary packages through pip:

```python 
pip install -r requirements.txt
```
# Usage  Example

run:

```python 
python3 main.py
```
