# IMDB Sentiment Analysis

This project performs sentiment analysis on movie reviews from the IMDB dataset. It uses Natural Language Processing (NLP) techniques and machine learning models to classify reviews as positive or negative.

## Dataset

The project uses the IMDB Dataset, which contains 50,000 movie reviews labeled as positive or negative. For this analysis, the first 10,000 examples were used.

## Preprocessing

The following preprocessing steps were applied to the data:
1. **HTML Tag Removal:** Removed HTML tags from reviews using regular expressions.
2. **Lowercasing:** Converted all text to lowercase for consistency.
3. **Stop Word Removal:** Removed common English stop words (e.g., "the," "a," "is") using NLTK's stopwords list.


## Feature Extraction

Two feature extraction methods were used:
1. **Bag-of-Words (BoW):** Created a numerical representation of text where each word is assigned a unique number. The frequency of each word in a review is used as a feature.
2. **TF-IDF (Term Frequency-Inverse Document Frequency):** Similar to BoW but gives more weight to words that are important within a document but rare across the entire corpus.


## Models

The following machine learning models were trained and evaluated:
1. **Gaussian Naive Bayes:** A probabilistic classifier based on Bayes' theorem.
2. **Random Forest:** An ensemble learning method that combines multiple decision trees to make predictions.


## Evaluation

Model performance was evaluated using accuracy and confusion matrix.

## Results

The accuracy scores obtained for different models and feature extraction methods are as follows:

* **Gaussian Naive Bayes with BoW:** Achieved an accuracy of around 75%.
* **Random Forest with BoW:** Achieved an accuracy of around 85% when limiting max features to 3000.
* **Random Forest with N-grams (2,2):**  Achieved an accuracy of around 80% with 5000 max features.
* **Random Forest with TF-IDF:** Achieved an accuracy of around 85%.

The results indicate that Random Forest models generally performed better than Gaussian Naive Bayes.

## Usage

To run the code in this project:
1. Upload the IMDB Dataset to the Colab environment.
2. Install the necessary libraries.
3. Run the code cells in the notebook sequentially.

## Libraries

The following libraries were used in the project:

* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
