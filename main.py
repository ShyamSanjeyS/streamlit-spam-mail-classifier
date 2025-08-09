import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set page config
st.set_page_config(page_title="📧 Spam Classifier Pro", layout="centered")

st.title("📧 Spam Email Classifier - Pro Edition")
st.markdown("✅ A full-featured ML-powered email spam classifier with live evaluation and export features.")

# Sidebar options
st.sidebar.header("🔧 Model & Settings")

uploaded_file = st.sidebar.file_uploader("📁 Upload your dataset (.json with MESSAGE & CATEGORY)", type=["json"])

model_name = st.sidebar.selectbox("Select Model", ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"])

# Hyperparameters
if model_name == "Multinomial Naive Bayes":
    alpha = st.sidebar.slider("Alpha (smoothing)", 0.0, 2.0, 1.0, step=0.1)
elif model_name == "Linear SVM":
    c_value = st.sidebar.slider("C (regularization)", 0.01, 10.0, 1.0, step=0.1)

# Load data
if uploaded_file:
    data = pd.read_json(uploaded_file)
    st.success("✅ Uploaded dataset loaded.")
else:
    try:
        data = pd.read_json("email-text-data.json")
        st.info("Using default dataset: `email-text-data.json`")
    except:
        st.error("❌ No dataset available.")
        st.stop()

# Validate columns
if 'MESSAGE' not in data.columns or 'CATEGORY' not in data.columns:
    st.error("❌ Dataset must contain 'MESSAGE' and 'CATEGORY' columns.")
    st.stop()

# Show data preview
if st.checkbox("👁 Preview Dataset"):
    st.dataframe(data.head())

# Feature Extraction
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['MESSAGE'])
y = data['CATEGORY']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=88)

# Model selection and training
with st.spinner("Training model..."):
    time.sleep(0.5)

    if model_name == "Multinomial Naive Bayes":
        model = MultinomialNB(alpha=alpha)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = LinearSVC(C=c_value)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

st.markdown(f"📈 **Model Accuracy:** `{accuracy * 100:.2f}%`")

# Classification Report
with st.expander("📑 Show Classification Report"):
    report = classification_report(y_test, y_pred, target_names=["HAM", "SPAM"])
    st.code(report)

# Confusion Matrix
with st.expander("📉 Show Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["HAM", "SPAM"], yticklabels=["HAM", "SPAM"], ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# 📊 Live prediction visualization
with st.expander("📊 Predicted Class Distribution"):
    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    class_counts = result_df['Predicted'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    class_counts.plot(kind='bar', color=['green', 'red'], ax=ax2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['HAM', 'SPAM'], rotation=0)
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

# CSV Export
with st.expander("📥 Export Predictions as CSV"):
    export_df = pd.DataFrame({'Message': data.loc[y_test.index, 'MESSAGE'], 'Actual': y_test, 'Predicted': y_pred})
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions CSV", data=csv, file_name='predictions.csv', mime='text/csv')

# Prediction function
def predict_message(msg):
    matrix = vectorizer.transform([msg])
    return model.predict(matrix)[0]

# Prediction Interface
st.header("📨 Test an Email Message")
msg = st.text_area("✍️ Enter a sample email message to classify:", height=150,
                   placeholder="E.g. You've won a free cruise! Click here to claim...")

if st.button("🔍 Classify"):
    if msg.strip():
        pred = predict_message(msg)
        if pred == 1:
            st.error("❌ This message is likely SPAM!")
        else:
            st.success("✅ This message is likely HAM (Not Spam).")
    else:
        st.warning("⚠️ Please enter a message.")

# Footer
st.markdown("---")
st.markdown(
    "<center>🚀 Built by a Shyam Sanjey  | Models: Naive Bayes, Logistic Regression, SVM | Streamlit App</center>",
    unsafe_allow_html=True
)
