 
This  repositary focuses **multitask learning** for sarcasm and **stress** detection using deep learning models. Here's a breakdown of the key steps and findings:

**1. Data Manipulation and Pandas:**

*   **Data Loading and Merging:** Four CSV files (`Dataset-Multitask1.csv`, `Dataset-Multitask2.csv`, `Dataset-Multitask3.csv`, `Dataset-Multitask4.csv`) were loaded using `pd.read_csv` and merged into a single DataFrame using `pd.concat`.
*   **Data Cleaning:** The `Unnamed: 0` column was dropped using `df.drop`.

**2. Text Preprocessing:**

*   **Lower Casing:** Text was converted to lowercase using `str.lower`.
*   **HTML Tag Removal:** HTML tags were removed using regular expressions (`re.compile` and `pattern.sub`).
*   **URL Removal:** URLs were removed using regular expressions.
*   **Punctuation Removal:** Punctuation marks were removed using `str.maketrans`.
*   **Spelling Correction:** Spelling errors were corrected using `TextBlob.correct`.
*   **Stemming:** Words were stemmed using `PorterStemmer`.
*   **Stop Word Removal:** Stop words were removed using `nltk.corpus.stopwords`.
*   **Tokenization:** Text was tokenized into words using `word_tokenize`.

**3. Architectures and Evaluation:**

*   **Multitask CNN:** A multitask CNN model was implemented with two output branches for sarcasm and stress prediction.
*   **GRU CNN GRU with Attention:** A more complex model incorporating GRU layers, attention mechanisms, and CNNs.
*   **Pipelined Stress Architecture:** A model where the output of the sarcasm prediction task is used as input for the stress prediction task.

**Evaluation Metrics:**

*   **Precision, Recall, F1-score:** These metrics were used to evaluate the performance of each model on the validation set.

| Architecture                   | Stress F1-score | Stress Precision |
  ----------------- | ---------------- | ----------------- |
| Multitask CNN                | 0.993            | 0.997            
| GRU CNN GRU with Attention | 0.991           | 0.988              
| Pipelined Stress Architecture| 0.878           | 0.784             
| Single task LSTM            | 0.958            | 0.935              


**4. Conclusions and Insights:**

* Conducted extensive text preprocessing, including lowercasing, HTML tag removal, URL removal, punctuation removal, spelling correction, stemming, stop word removal, and tokenization, and utilized Stanford GloVe embeddings to represent text sequences for enhanced model performance.

*   Multitask learning approaches show promising results for sarcasm and stress detection, outperforming the single task baseline model.
*   Implemented and evaluated three multitask learning architectures (Multitask CNN, BiGRU CNN BiGRU with Attention layers, and Pipelined stress architecture) for simultaneous sarcasm and stress detection, achieving significant improvements over baseline models.
  
* Implemented the Multitask CNN architecture, outperforming the baseline Vinella LSTM by **5%** in **F1-score** (0.991 vs. 0.958) for stress detection which shows shared learning.
  
* Implemented the GRU CNN GRU with Attention architecture, outperforming the baseline Vinella LSTM by **4%** in **F1-score** (0.991 vs. 0.958) for stress detection.

*   Further exploration of different architectures, huge dataset and hyperparameters can potentially lead to even better results.
