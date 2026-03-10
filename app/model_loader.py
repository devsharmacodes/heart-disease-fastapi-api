import pickle

# load the trained model
def load_model():

    with open("models/heart_disease.pkl", "rb") as file:
        model = pickle.load(file)

    return model