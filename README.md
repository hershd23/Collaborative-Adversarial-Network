# Collaborative-Adversarial-Network
## For sentence similarity detection

My implementation of the CAN architecture proposed by Chen et al

Link to the paper :- https://dl.acm.org/citation.cfm?id=3210019

Link to the dataset :- https://www.microsoft.com/en-us/download/details.aspx?id=52398

Although the train and test tsv files can be found in the dataset folder.

## Model Description

The CAN model comprises of a siamese ma-LSTM network followed by a generator-discriminator network eventually followed by a softmax layer which gives out the final predictions for each class (classes = 2 in this case)

![What is this?](img/overview.png?raw=true "Title")

The word embeddings are fed into the siamese(shared) BiLSTM layer and the average pool of the hidden states each of the sentences i.e. sentence1/sentence2 in paraphrase detection task or question/answer in the question answering task is taken to be the sentence representation in this case(attention can also be used to find out the sentence representation). The Manhattan distance of the two representations is used to calculate the similarity score between the two sentences.

The hypothesis in the paper is that in case of tasks like paraphrase detection or even in cases like question-answering the common words between the two sentences are more important than the other words for similarity measurement and that the corresponding hidden states would contain more important information (stopwords are removed). So accordingly we take the common words from the two sentences and take their respective hidden states and max-pool them to get another sentence representation or feature vector.

## Common Feature Extractor

![What is this?](img/fgen.png?raw=true "Title")

After the extraction of the elite hidden states we send the feature vectors through generator-discriminator network. As a rule we use the answer sentence as the one which gets sent through the generator:- A Linear layer followed by a tanh function 
In order to get the feature Fg which gets passed into the discriminator:- A Linear layer followed by a softmax to classify the real features from the generated feature vectors.
The generator plays a collaborative adversarial game with the discriminator, i.e. it tries to fool the discriminator when the label is true and assists the discriminator when false. The generated feature vector is concatenated to the similarity score obtained from the LSTM and fed into a softmax layer to get our predictions of the classes.


## Other specifications

Dropout on the final layer = 0.5
Optimizer = Adadelta
Stopwords removed (from the nltk stopwords dictionary)
