# Collaborative-Adversarial-Network
My implementation of the CAN architecture proposed by Chen et al

Link to the paper :- https://dl.acm.org/citation.cfm?id=3210019

Link to the dataset :- https://www.microsoft.com/en-us/download/details.aspx?id=52398

Although the train and test tsv files can be found in the dataset folder.

## Model Description

The CAN model comprises of a siamese ma-LSTM network followed by a generator-discriminator network eventually followed by a softmax layer which gives out the final predictions for each class (classes = 2 in this case)

![Alt text](img/overview.png?raw=true "Title")
