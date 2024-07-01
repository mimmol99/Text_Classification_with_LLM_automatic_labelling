from docx import Document
import os
import tensorflow as tf
import tempfile

def save_sentences_to_temp(prompts, labels):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create subdirectories for each label
    unique_labels = set(labels)
    for label in unique_labels:
        os.makedirs(os.path.join(temp_dir, str(label)), exist_ok=True)
    
    # Save sentences to the respective directories
    for idx, (prompt, label) in enumerate(zip(prompts, labels)):
        label_dir = os.path.join(temp_dir,str(label))
        file_path = os.path.join(label_dir, f'{label}_text_{idx}.txt')
        with open(file_path, 'w') as f:
            f.write(prompt)
    
    return temp_dir

def create_text_dataset(prompts, labels):
    temp_dir = save_sentences_to_temp(prompts, labels)
    
    text_dataset = tf.keras.preprocessing.text_dataset_from_directory(
        temp_dir,
        labels='inferred',
        label_mode='int',
        class_names=None,
        batch_size=1,
        max_length=None,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        follow_links=False
    )
    
    return text_dataset

def read_prompts_from_docx(file_path):
    doc = Document(file_path)
    prompts = [para.text for para in doc.paragraphs if para.text.strip()]
    return prompts

def get_llm_labels(llm, prompts):
    llm_prompt = ("Perform binary classification returning 0 if the query is about a personal document "
                  "(e.g. 'Quali sono le garanzie comprese nella mia polizza?'). "
                  "Otherwise, return 1 if the query is generic (e.g. 'È prevista un’auto sostitutiva in caso di furto e incendio?'). "
                  "Query to classify: ")

    labels = []
    for prompt in prompts:
        llm_answer = llm.invoke(llm_prompt + prompt)
        label = int(llm_answer.content)
        labels.append(label)
    
    return labels

