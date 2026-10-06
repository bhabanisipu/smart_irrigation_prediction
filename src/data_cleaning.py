import pandas as pd


def clean_data(df):
    """
    Clean the irrigation dataset.
    """

    try:
        df_clean = df.copy()

        df_clean.rename(
            columns={
                "Soil Tempertuer": "Soil Temperature",
                "irrigation ": "irrigation"
            },
            inplace=True
        )
        if "best time" in df_clean.columns:
            df_clean.drop(
                columns=["best time"],
                inplace=True
            )
        df_clean["Time"] = pd.to_datetime(
            df_clean["Time"]
        )

        df_clean = df_clean.sort_values(
            "Time"
        ).reset_index(drop=True)

        sensor_columns = [
            "Temperature",
            "Humidity",
            "Soil Moisture"
        ]

        # Handle missing sensor values
        df_clean[sensor_columns] = (
            df_clean[sensor_columns]
            .interpolate(method="linear")
        )

        # Remove rows where target is missing
        df_clean = df_clean.dropna(
            subset=["irrigation"]
        ).reset_index(drop=True)

        print("Data cleaning completed successfully.")
        print("Cleaned dataset shape:", df_clean.shape)

        return df_clean

    except Exception as e:
        print("Error occurred during data cleaning:", e)
        raise