import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import gradio as gr

import warnings
warnings.filterwarnings('ignore')

#load the data

df = pd.read_csv('iris.csv')
df

X = df.drop(columns=['species'])
Y = df['species']

# splitting the data into train test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#load the model
model = LogisticRegression()
model.fit(X_train, Y_train)

y_predicted = model.predict(X_test)



# Let's try with giving input and getting output

def input_and_prediction(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = model.predict(input_data)
    
    # Directly return the predicted species (since it's already a string)
    return f"Predicted species: {prediction[0]}"


#gradio code
interface = gr.Interface(fn = input_and_prediction,
                         inputs = [gr.Number(label = 'Sepal Length (cm)'),
                                   gr.Number(label = 'Sepal Width (cm)'),
                                   gr.Number(label = 'Petal Length (cm)'),
                                   gr.Number(label = 'Petal Width (cm)')],
                         outputs="text")

interface.launch()