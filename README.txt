This code uses a kNN classification algorithm
from sklearn.neighbors
It first reads the dataset from a csv file,
Then splits the data into features (X) and labels (y),
Then, it splits tge dataset into training and testing sets,
using tts from sklearn.model_selection
Afterwards, it trains aand tests the classifier
Finally, it calculates the accuracy score,
using accuracy_score from sklearn.metrics