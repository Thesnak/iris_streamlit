import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

@st.cache_resource
def load_model():
    model   = joblib.load('iris_model.pkl')
    dataset = joblib.load('iris_dataset.pkl')
    return model, dataset

model, iris = load_model()

# ── Header ───────────────────────────────────────────────────
st.title("🌸 Iris Flower Classifier")
st.markdown("Adjust the sliders to input flower measurements and get a prediction.")

# ── Sidebar inputs ───────────────────────────────────────────
st.sidebar.header("🎚️ Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  2.0, 4.5, 3.4, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.7, 0.1)
petal_width  = st.sidebar.slider("Petal Width (cm)",  0.1, 2.5, 1.5, 0.1)
predict_button = st.sidebar.button("Predict")

# ── Prediction ───────────────────────────────────────────────
if predict_button:
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    pred       = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]
    species    = iris.target_names[pred]

    emoji = {"setosa": "🌼", "versicolor": "🌺", "virginica": "🌷"}

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔍 Prediction")
        st.metric("Predicted Species", f"{emoji.get(species,'')} {species.capitalize()}")
        st.metric("Confidence", f"{pred_proba.max():.1%}")

        st.subheader("📊 Class Probabilities")
        prob_df = pd.DataFrame({'Species': iris.target_names, 'Probability': pred_proba})
        st.dataframe(prob_df.style.format({'Probability': '{:.1%}'}), use_container_width=True)

    with col2:
        # Probability bar chart
        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ['#FF4B4B' if i == pred else '#D3D3D3' for i in range(3)]
        bars   = ax.barh(iris.target_names, pred_proba, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title('Prediction Probabilities')
        for bar, prob in zip(bars, pred_proba):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{prob:.2%}', va='center')
        st.pyplot(fig)

    # ── Feature importance ───────────────────────────────────────
    st.divider()
    st.subheader("🌲 Feature Importance")

    fi = model.named_steps['clf'].feature_importances_
    fi_df = pd.DataFrame({'Feature': iris.feature_names, 'Importance': fi}).sort_values('Importance')

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.barh(fi_df['Feature'], fi_df['Importance'], color='steelblue')
    ax2.set_xlabel('Importance')
    ax2.set_title('Random Forest Feature Importances')
    st.pyplot(fig2)

# ── Dataset explorer ─────────────────────────────────────────
st.divider()
with st.expander("🔬 Explore Dataset"):
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[t] for t in iris.target]
    st.dataframe(df, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for sp in iris.target_names:
        sub = df[df['species'] == sp]
        ax3.scatter(sub['petal length (cm)'], sub['petal width (cm)'], label=sp, s=60, alpha=0.7)
    ax3.scatter([petal_length], [petal_width], marker='*', s=300, color='red', zorder=5, label='Your input')
    ax3.set_xlabel('Petal Length'); ax3.set_ylabel('Petal Width')
    ax3.set_title('Petal Dimensions by Species'); ax3.legend()
    st.pyplot(fig3)