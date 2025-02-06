 
### Title: Detection of psychological stress through social media using deep learning

The present generation's major challenge is stress. To overcome the negative mental health effects, it is vital to control the stress levels in the early stages. In this regard, the literature evinces that, to build robust and accurate stress detection AI systems, sarcasm could be used as an auxiliary label. The social media platform Twitter has the sarcastic tweets that could be used for research. The aim of the research was to use multitask learning to primarily detect stress using auxiliary sarcasm expressions with deep learning.

As part of my internship, I was involved in the area of building models and analyzing their performance. We had implemented various single-task algorithms and compared them with those of multitask deep learning architectures. As we were dealing with tweets, I have conducted extensive NLP data preprocessing; it includes spelling corrections, removal of HTML tags, stemming, and tokenization. Following, I have leveraged **Stanford GloVe embeddings** to convert text sequences to embeddings, as GloVe embeddings capture semantic meaning in the text.

**Text Preprocessing:**

*   **Lower Casing:** Text was converted to lowercase using `str.lower`.
*   **HTML Tag Removal:** HTML tags were removed using regular expressions (`re.compile` and `pattern.sub`).
*   **URL Removal:** URLs were removed using regular expressions.
*   **Punctuation Removal:** Punctuation marks were removed using `str.maketrans`.
*   **Spelling Correction:** Spelling errors were corrected using `TextBlob.correct`.
*   **Stemming:** Words were stemmed using `PorterStemmer`.
*   **Stop Word Removal:** Stop words were removed using `nltk.corpus.stopwords`.
*   **Tokenization:** Text was tokenized into words using `word_tokenize`


Implemented single-task learning (STL) vanilla deep learning models like Long Short-Term Memory (LSTMs) and Bidirectional Gated Recurrent Units (BiGRUs) and different multi-task deep learning architectures sourced from research papers include multi-task CNN, CNN GRU with attention layers, and pipelined stress architecture. The architectures were different in the following ways:

● **Multi-task CNN**: In this architecture, 1D CNN (since text data) with dropout (to reduce the overfitting) layers was used. During the training process, the weights of both stress and sarcasm would be shared, as weight updates take place according to maximum loss.

● **GRU CNN with Attention Layers architecture**: For both the primary (stress) and secondary (sarcasm) tasks, Gated Recurrent Units (GRU) with an attention layer were concatenated with CNN output. The weight sharing happens during training.

● **BiGRU pipelined stress architecture**: It consists of Bidirectional Gated Recurrent Units (GRU), attention layers, and dense layers. During the training process, the predicted value of sarcasm was used to predict stress in addition to shared learning.

**Evaluation Metrics:**

| Architecture                   | Stress F1-score | Stress Precision |
  ----------------- | ---------------- | ----------------- |
| Multitask CNN                | 0.993            | 0.997            
| GRU CNN GRU with Attention | 0.991           | 0.988              
| Pipelined Stress Architecture| 0.878           | 0.784             
| Single task LSTM            | 0.958            | 0.935 

In addition, for future work, I also studied evolutionary computation algorithms like genetic algorithms and particle-swarm optimization to tune the hyperparameters of text classification neural networks. And also Graph Neural Networks (GNNs) because of their ability to capture semantic meaning in the text.

We were able to see promising results with multi-task learning (MTL) compared with that of single-task learning (STL), and we are going to extend our research in the process of compelling our work to publish in a journal or a conference.



