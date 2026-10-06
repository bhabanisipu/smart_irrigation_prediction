import pandas as pd


def create_features(df):
    """
    Create additional features for the irrigation prediction model.
    """

    try:
        df_featured = df.copy()

        df_featured["Soil Moisture Change"] = (
            df_featured["Soil Moisture"].diff()
        )

        df_featured["Soil Moisture Rolling Mean"] = (
            df_featured["Soil Moisture"]
            .rolling(window=5)
            .mean()
        )

        df_featured = (
            df_featured
            .dropna()
            .reset_index(drop=True)
        )

        print("Feature engineering completed successfully.")
        print(
            "Feature engineered dataset shape:",
            df_featured.shape
        )

        return df_featured

    except Exception as e:
        print(
            "Error occurred during feature engineering:",
            e
        )
        raise