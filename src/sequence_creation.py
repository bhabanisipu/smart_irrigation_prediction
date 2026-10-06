import numpy as np


def create_sequences(X, y, time_steps=20):
    """
    Create time-series sequences for LSTM.

    Parameters:
        X : scaled input features
        y : target values
        time_steps : number of previous observations
                     used to predict the current target

    Returns:
        X_sequences : LSTM input sequences
        y_sequences : corresponding target values
    """

    try:
        X_sequences = []
        y_sequences = []

        for i in range(time_steps, len(X)):

            # Take previous 20 time steps as input
            X_sequences.append(
                X[i - time_steps:i]
            )

            # Take current irrigation value as target
            y_sequences.append(
                y.iloc[i]
            )

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        print("Sequence creation completed successfully.")
        print("X sequence shape:", X_sequences.shape)
        print("y sequence shape:", y_sequences.shape)

        return X_sequences, y_sequences

    except Exception as e:
        print(
            "Error occurred during sequence creation:",
            e
        )
        raise