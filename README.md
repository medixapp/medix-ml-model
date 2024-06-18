# Repository for Machine Learning Path

Machine Learning Cohorts are involved in gathering, cleaning, and preprocess required data, and the most important is building Neural Network models. Tensorflow would be our main tools.

# Tools and Languages

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white) ![ScikitLearn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![GSheet](https://img.shields.io/badge/Google%20Sheets-34A853?style=for-the-badge&logo=google-sheets&logoColor=white) ![Git](https://img.shields.io/badge/GIT-E44C30?style=for-the-badge&logo=git&logoColor=white) ![Conda](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white) ![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252) ![VSCode](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white) ![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

We have built 3 DNN models, the purposes are listed here.

1. Predict 14 digestive system diseases based on the symptoms. To achieve this, there is a plan to make 2 different versions of the model. First is a strict version, where the symptoms are predefined and limited by our scope, and the other is using Embedding to handle the limitation. This would allow user to choose whether if the symptoms choices are enough, or they prefer to input the symptoms by themselves in a free-text format.

2. [Extractive QnA Chatbot](https://huggingface.co/wdevinsp/indobert-base-uncased-finetuned-digestive-qna) with predefined context to answer questions regarding stomach diseases. To achieve this, we fine-tuned the [Indolem's IndoBERT Base Uncased](https://huggingface.co/indolem/indobert-base-uncased) from HuggingFace Hub using [a custom dataset](https://huggingface.co/datasets/wdevinsp/digestive_indonesian_qna) collected from 7 articles on a medical website as the context.

# Models' Folders
- [Disease Prediction with Predefined Symptoms](https://github.com/medixapp/medix-ml-model/tree/main/Predefined_Symptoms_Model).
- [Disease Prediction with Word Embedding](https://github.com/medixapp/medix-ml-model/tree/main/Word_Embeddings_Model).
- [Fine-Tuned Extractive QnA Chatbot](https://github.com/medixapp/medix-ml-model/tree/main/Fine_Tuned_QnA_Chatbot).