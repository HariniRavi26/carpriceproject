import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


# ==========================
# TITLE
# ==========================

st.title("🚗 Car Price Prediction")


# ==========================
# LOAD DATASET
# ==========================

df = pd.read_excel(
    "cleaned_carprice_dataset.xlsx"
)


# ==========================
# STORE ORIGINAL CAR NAMES
# ==========================

original_car_names = df["CarName"].copy()


# ==========================
# ENCODE TEXT COLUMNS
# ==========================

for col in df.select_dtypes(
        include=["object"]
):

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(

        df[col].astype(str)

    )


# ==========================
# FEATURES + TARGET
# ==========================

X = df.drop(

    columns=["price"]

)

y = df["price"]


# ==========================
# TRAIN MODEL
# ==========================

model = LinearRegression()

model.fit(

    X,

    y

)


# ==========================
# DROPDOWN
# ==========================

selected_car = st.selectbox(

    "Select Car Name",

    original_car_names.unique()

)


# ==========================
# PREDICT BUTTON
# ==========================

if st.button(

        "Predict Price"

):

    row_index = original_car_names[

        original_car_names == selected_car

    ].index[0]


    sample = X.iloc[[

        row_index

    ]]


    prediction = model.predict(

        sample

    )[0]


    actual = y.iloc[

        row_index

    ]


    st.success(

        f"Predicted Price: ₹ {prediction:,.2f}"

    )


    st.write(

        f"Actual Dataset Price: ₹ {actual:,.2f}"

    )


# ==========================
# DATA PREVIEW
# ==========================

st.subheader(

    "Dataset Preview"

)

st.dataframe(

    pd.read_excel(

        "cleaned_carprice_dataset.xlsx"

    ).head()

)