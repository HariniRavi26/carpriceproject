import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# ===================================
# TITLE
# ===================================

st.title("🚗 Car Price Prediction")

# ===================================
# LOAD DATASET
# ===================================

df = pd.read_excel(
    "cleaned_carprice_dataset.xlsx",
    engine="openpyxl"
)

# Keep original car names before encoding
original_car_names = df["CarName"].copy()

# ===================================
# ENCODE ALL OBJECT COLUMNS
# ===================================

for col in df.select_dtypes(include=['object']).columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )


# ===================================
# FEATURES AND TARGET
# ===================================

X = df.drop(
    columns=["price"]
)

y = df["price"]


# ===================================
# TRAIN MODEL
# ===================================

model = LinearRegression()

model.fit(
    X,
    y
)

# ===================================
# DROPDOWN
# ===================================

selected_car = st.selectbox(

    "Select Car Name",

    original_car_names.unique()

)

# ===================================
# PREDICT
# ===================================

if st.button("Predict Price"):

    # Find selected row in original dataset
    row_index = original_car_names[
        original_car_names == selected_car
    ].index[0]


    sample = X.iloc[[row_index]]

    prediction = model.predict(
        sample
    )[0]


    actual_price = y.iloc[
        row_index
    ]


    st.success(

        f"Predicted Price: ₹ {prediction:,.2f}"

    )


    st.write(

        f"Actual Dataset Price: ₹ {actual_price:,.2f}"

    )


# ===================================
# SHOW DATA
# ===================================

st.subheader(
    "Dataset Preview"
)

st.dataframe(
    pd.read_excel(
        "cleaned_carprice_dataset.xlsx",
        engine="openpyxl"
    ).head()
)