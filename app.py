import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


with open("random_forest_model.pkl", "rb") as f:
    rf_model = joblib.load(f)

model = tf.keras.models.load_model("attack_classification_model.h5")

# ColumnTransformer
with open("col_transformer.pkl", "rb") as f:
    col_trans = joblib.load(f)

# Label encoder
with open("label_encoder.pkl", "rb") as f:
    attack_encoder = joblib.load(f)

# PCA model
with open("pca_model.pkl", "rb") as f:
    pca = joblib.load(f)


# Top 10 features used for input
top_10_features = [
   'Version', 'Record total duration', 'Source to destination transaction bytes', 'Rate', 'Source to destination time to live value', 'Source bits per second', 'Destination bits per second', 'Mean of the packet size transmitted by the destination',
       'No. for each state according to specific range of values for source/destination time to live', 'No. of connections that contain the same service and destination address in 100 connections according to the last time.'
]

st.title("Attack Detection & Classification")

st.write("Enter the values for the following network features:")

user_input = []
for feature in top_10_features:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.2f")
    user_input.append(value)

user_input_df = pd.DataFrame([user_input], columns=top_10_features)

user_input_transformed = col_trans.transform(user_input_df)

user_input_pca = pca.transform(user_input_transformed)

# Run the model
if st.button("Detect Attack"):

    # Attack detection using Random Forest
    pred_attack = rf_model.predict(user_input_pca)[0]

    # Reshape the random forest prediction to append it to PCA features
    rf_pred_reshaped = np.array([ pred_attack]).reshape(-1, 1)
    
    # Combine PCA features with random forest prediction
    input_with_rf = np.hstack((user_input_pca, rf_pred_reshaped))

    if pred_attack == 1:
        st.write("⚠️ Attack Detected!")

        pred_attack_class = model.predict(input_with_rf)
        predicted_class_index = np.argmax(pred_attack_class)

        attack_categories = ["Analysis", "Backdoor", "DoS", "Exploits", 
                         "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"]
    
        #attack_class = attack_categories[predicted_class_index]
        attack_class = attack_encoder.inverse_transform([np.argmax(pred_attack_class)])[0]

        if isinstance(attack_class, (int, np.integer)):
            attack_class = attack_categories[predicted_class_index]

        st.write(f"**Predicted Attack Type:** 🛑 {attack_class}")
    else:
        st.success("✅ No Attack Detected")



        
