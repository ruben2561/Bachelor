from hmmlearn import hmm
import numpy as np

def train_model(train_data_path):
    # Define the number of states in the HMM model
    n_states = 10

    # Define the number of possible emissions
    n_emissions = 32

    # Initialize the HMM model
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=0.01)

    # Initialize the model parameters randomly
    model.startprob_ = np.random.rand(n_states)
    model.startprob_ /= np.sum(model.startprob_)
    model.transmat_ = np.random.rand(n_states, n_states)
    model.transmat_ /= np.tile(np.sum(model.transmat_, axis=1)[:, np.newaxis], (1, n_states))
    model.emissionprob_ = np.random.rand(n_states, n_emissions)
    model.emissionprob_ /= np.tile(np.sum(model.emissionprob_, axis=1)[:, np.newaxis], (1, n_emissions))

    # Load the training data
    train_data = np.load(train_data_path)

    # Train the HMM model
    model.fit(train_data)

    # Save the trained HMM model
    from joblib import dump
    dump(model, 'hmm_model.joblib')