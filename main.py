from llm_models import OpenAIModel
from models import KNNModel, RFModel, LSTMModel, GRUModel, BERTModel
from utils import read_prompts_from_docx, get_llm_labels,create_text_dataset
from dotenv import load_dotenv
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_text as text

load_dotenv(Path("../../api_key.env"))

def main():
    model_name = 'gpt-3.5-turbo'
    model = OpenAIModel(model_name=model_name)
    llm = model.get_model()
    
    file_path = "../Domande_Chatbot.docx"
    prompts = read_prompts_from_docx(file_path)

    labels = get_llm_labels(llm, prompts)

    # Convert labels to numpy array
    labels = np.array(labels)
    len_train_samples = int(len(prompts)*0.9)

    train_prompts = prompts[:len_train_samples]
    train_labels = labels[:len_train_samples]

    test_prompts = prompts[len_train_samples:]
    test_labels = labels[len_train_samples:]

    train_text_dataset = create_text_dataset(train_prompts,train_labels)
    test_text_dataset = create_text_dataset(test_prompts,test_labels)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(prompts, labels, test_size=0.2, random_state=42)
    
    # Initialize and train models
    knn_model = KNNModel(X_train, y_train, X_test, y_test)
    knn_model.halving_random_train()
    rf_model = RFModel(X_train, y_train, X_test, y_test)
    rf_model.halving_random_train()
    lstm_model = LSTMModel(X_train, y_train, X_test, y_test)
    lstm_model.train()
    gru_model = GRUModel(X_train, y_train, X_test, y_test)
    gru_model.train()
    bert_model = BERTModel(train_text_dataset)
    bert_model.train()


    # Evaluate models
    knn_accuracy = knn_model.eval()
    rf_accuracy = rf_model.eval()
    lstm_accuracy = lstm_model.eval()
    gru_accuracy = gru_model.eval()
    bert_accuracy = bert_model.eval(test_text_dataset)

    # Print accuracies
    print(f"KNN Test Accuracy: {knn_accuracy}")
    print(f"Random Forest Test Accuracy: {rf_accuracy}")
    print(f"LSTM Test Accuracy: {lstm_accuracy}")
    print(f"GRU Test Accuracy: {gru_accuracy}")
    print(f"BERT Test Accuracy: {bert_accuracy}")

    # Plotting the accuracies
    model_names = ['KNN', 'Random Forest', 'LSTM', 'GRU', 'BERT']
    accuracies = [knn_accuracy, rf_accuracy, lstm_accuracy, gru_accuracy, bert_accuracy]
    
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'cyan'])  # , 'magenta'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison on Test Set')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    plt.savefig("./plot.png")
    plt.show()

if __name__ == "__main__":
    main()

