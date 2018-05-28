# ManhattanLSTM-with-Attention

Introduction:
Semantic similarity is a metric defined over a set of documents or terms, where the idea of distance between them is based on the likeness of their meaning or semantic content as opposed to similarity which can be estimated regarding their syntactic representation (e.g. their string format).These are mathematical tools used to estimate the strength of the semantic relationship between units of language, concepts or instances, through a numerical description obtained according to the comparison of information supporting their meaning or describing their nature.

Model: 

It is a siamese network having shared LSTM with an attention layer , Attention is applied to each of the LSTM generated representation in reference to the other ,i.e. Left sentence is attended using the right representation as query vector, and right sentence with left sentence representation as query vector , and then the manhattan distance is calculated of both the representation and compared to the actual distance and the mean square loss is propagated.






Running procedure :

python DataCollection.py
python Training.py
python Testing.py

word2vec used :GoogleNews-vectors-negative300.bin.gz

correlation coefficient command used : 

perl correlation.pl <actualtest.txt> <predicted ouput>


