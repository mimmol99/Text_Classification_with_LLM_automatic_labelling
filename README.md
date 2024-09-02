# Text_Classification
Text classification using different techniques and an LLM as automatic label generator.

# First phase: load sentences
Load sentences from a document,one per line.

# Second phase: Choose an LLM
Use one LLM from OpenAI family,Groq or Claude to generate labels. Remember to modify the prompt (in utils.py/get_llm_labels) for your case.

# Third phase: Try different techniques
Choose one or more techniques to classify the sentences.

- KNN with common train or HalvingRandomSearchCV
- RandomForest with common train or HalvingRandomSearchCV
- LSTM
- GRU
- BERT

# Last phase: comparison of results
Plot a graph to visualize results.

