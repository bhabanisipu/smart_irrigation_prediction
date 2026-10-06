import os

import joblib
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "template"),
    static_folder=os.path.join(BASE_DIR, "static")
)

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "models", "irrigation_lstm.keras")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "scaler.pkl")
)

threshold = joblib.load(
    os.path.join(BASE_DIR, "models", "threshold.pkl")
)

feature_columns = [
    "crop type",
    "Temperature",
    "Humidity",
    "Soil Moisture",
    "Soil Temperature",
    "Soil Moisture Change",
    "Soil Moisture Rolling Mean"
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    error = None

    if request.method == "POST":

        try:

            # Get 20 historical observations
            raw_data = request.form["historical_data"]

            # Split into individual rows
            rows = raw_data.strip().split("\n")

            if len(rows) != 20:
                raise ValueError(
                    "Please provide exactly 20 observations."
                )

            data = []

            for row in rows:

                values = row.strip().split(",")

                if len(values) != 5:
                    raise ValueError(
                        "Each row must contain 5 values."
                    )

                crop_type = float(values[0])
                temperature = float(values[1])
                humidity = float(values[2])
                soil_moisture = float(values[3])
                soil_temperature = float(values[4])

                data.append([
                    crop_type,
                    temperature,
                    humidity,
                    soil_moisture,
                    soil_temperature
                ])

            data = np.array(data, dtype=float)

            soil_moisture = data[:, 3]

            moisture_change = np.diff(
                soil_moisture,
                prepend=soil_moisture[0]
            )

            rolling_mean = []

            for i in range(len(soil_moisture)):

                start = max(0, i - 4)

                mean_value = np.mean(
                    soil_moisture[start:i + 1]
                )

                rolling_mean.append(mean_value)

            final_data = np.column_stack([
                data[:, 0],
                data[:, 1],
                data[:, 2],
                data[:, 3],
                data[:, 4],
                moisture_change,
                rolling_mean
            ])
            scaled_data = scaler.transform(
                final_data
            )

            X_input = scaled_data.reshape(
                1,
                20,
                7
            )

            prediction_probability = model.predict(
                X_input,
                verbose=0
            )[0][0]


            probability = round(
                float(prediction_probability) * 100,
                2
            )


            # Apply saved threshold
            if prediction_probability >= threshold:

                prediction = "Irrigation Required"

            else:

                prediction = "No Irrigation Required"


        except Exception as e:

            error = str(e)


    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )

if __name__ == "__main__":

    app.run(debug=True)