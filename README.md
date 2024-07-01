# Text_Classification
Text classification on different techniques using an LLM as label generator

# First phase: load sentences
Load sentences from a document,one per line

# Second phase: Choose an LLM
Use one LLM from OpenAI family,Groq or Claude to extract label. Remember to modify the prompt for your case.

# Third phase: Try different techniques
Choose one or more techniques to classify the sentences

- KNN with common train or HalvingRandomSearchCV
- RandomForest with common train or HalvingRandomSearchCV
- LSTM
- GRU
- BERT

# Last phase: comparison of results
Plot a graph to visualize results

