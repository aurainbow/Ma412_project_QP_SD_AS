Multi-Label Classification of Scientific Literature Using NASA SciX Corpus

------ Overview ------

This project focuses on developing a multi-label classification model to analyze and categorize scientific literature from the NASA SciX corpus. 
The model predicts relevant keywords for a given document based on its title and abstract, enabling automated indexing and enhancing the accessibility of research documents across diverse scientific domains.

The code implements a supervised learning pipeline, leveraging text preprocessing, feature extraction, and a classification algorithm to predict multiple labels for each document. 
Additionally, pseudo-labeling is explored to augment the training process by incorporating partially labeled data.

------ Installation / Prerequisites ------

Python: 3.7 or higher
Libraries: pandas, numpy, scikit-learn, nltk

Ensure the required libraries are installed:
pip install pandas numpy scikit-learn nltk

Download the NLTK resources required for stop-word removal and lemmatization:
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

Replace the train_file and val_file variables with the paths to your dataset files.

------ Dataset ------

The code uses the NASA SciX dataset, provided in Parquet format, containing titles, abstracts, and verified UAT (Unified Astronomy Thesaurus) labels. 
Ensure the dataset files (train.parquet and val.parquet) are placed in the specified paths before running the code.

------ Features ------

Text Preprocessing: Combines title and abstract fields, cleans text, and applies lemmatization and stop-word removal.

Feature Extraction: Converts text data into numerical features using Count Vectorization and TF-IDF transformation.

Multi-Label Classification: Employs an SGDClassifier with a One-vs-Rest strategy to predict multiple labels for each document.

Pseudo-Labeling: Incorporates semi-supervised learning to augment training data with pseudo-labeled examples.

Evaluation Metrics: Includes F1-score, accuracy, and confusion matrix to evaluate model performance.

------ File Structure ------

final_code.py: Main code for preprocessing, training, and evaluating the model.
train.parquet: Training dataset containing titles, abstracts, and labels.
val.parquet: Validation dataset for evaluating the model.

------ Customization ------

Adjust Parameters: Modify hyperparameters such as sgd_params and vectorizer_params to experiment with different configurations.

Add Preprocessing Steps: Extend the preprocess_text function to include additional cleaning steps or custom tokenization.

Change Classifiers: Replace SGDClassifier with other classifiers (e.g., Random Forest, SVM) to compare performance.