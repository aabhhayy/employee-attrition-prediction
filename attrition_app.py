"""Streamlit Employee Attrition UI

Run with:
    streamlit run attrition_app.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

DATA_PATH = "Employee-Attrition.csv"
FEATURE_COLUMNS = [
    'Age',
    'BusinessTravel',
    'Department',
    'EducationField',
    'EnvironmentSatisfaction',
    'Gender',
    'JobInvolvement',
    'JobRole',
    'JobSatisfaction',
    'MaritalStatus',
    'Over18',
    'OverTime',
    'Education',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'WorkLifeBalance',
]


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    df['Attrition'] = df['Attrition'].map({'No': 0, 'Yes': 1})
    df['OverTime'] = df['OverTime'].map({'No': 0, 'Yes': 1})
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Over18'] = df['Over18'].map({'Y': 1, 'N': 0})

    encoding_cols = [
        'BusinessTravel',
        'Department',
        'EducationField',
        'JobRole',
        'MaritalStatus',
    ]

    label_encoders = {}
    for column in encoding_cols:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        label_encoders[column] = encoder

    return df, label_encoders


def oversample_minority(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    df = pd.concat([X, y], axis=1)
    majority = df[df['Attrition'] == 0]
    minority = df[df['Attrition'] == 1]

    if len(minority) == 0:
        return X, y

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=random_state,
    )
    df_upsampled = pd.concat([majority, minority_upsampled])
    X_resampled = df_upsampled.drop('Attrition', axis=1)
    y_resampled = df_upsampled['Attrition']
    return X_resampled, y_resampled


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df['Attrition']

    X_over, y_over = oversample_minority(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_over, y_over, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=500, solver='liblinear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'fpr_tpr': roc_curve(y_test, y_proba),
    }

    return model, metrics


def build_sidebar_inputs(label_encoders: dict, template: pd.Series):
    st.sidebar.header('Employee details')

    input_data = {
        'Age': st.sidebar.slider('Age', 18, 60, int(template['Age'])),
        'Gender': st.sidebar.selectbox('Gender', ['Male', 'Female']),
        'OverTime': st.sidebar.selectbox('OverTime', ['No', 'Yes']),
        'Over18': st.sidebar.selectbox('Over18', ['Y', 'N']),
        'BusinessTravel': st.sidebar.selectbox(
            'BusinessTravel', list(label_encoders['BusinessTravel'].classes_)
        ),
        'Department': st.sidebar.selectbox(
            'Department', list(label_encoders['Department'].classes_)
        ),
        'EducationField': st.sidebar.selectbox(
            'EducationField', list(label_encoders['EducationField'].classes_)
        ),
        'JobRole': st.sidebar.selectbox('JobRole', list(label_encoders['JobRole'].classes_)),
        'MaritalStatus': st.sidebar.selectbox(
            'MaritalStatus', list(label_encoders['MaritalStatus'].classes_)
        ),
        'Education': st.sidebar.slider('Education', 1, 5, int(template['Education'])),
        'EnvironmentSatisfaction': st.sidebar.slider(
            'EnvironmentSatisfaction', 1, 4, int(template['EnvironmentSatisfaction'])
        ),
        'JobInvolvement': st.sidebar.slider(
            'JobInvolvement', 1, 4, int(template['JobInvolvement'])
        ),
        'JobSatisfaction': st.sidebar.slider(
            'JobSatisfaction', 1, 4, int(template['JobSatisfaction'])
        ),
        'PerformanceRating': st.sidebar.slider(
            'PerformanceRating', 1, 4, int(template['PerformanceRating'])
        ),
        'RelationshipSatisfaction': st.sidebar.slider(
            'RelationshipSatisfaction', 1, 4, int(template['RelationshipSatisfaction'])
        ),
        'WorkLifeBalance': st.sidebar.slider(
            'WorkLifeBalance', 1, 4, int(template['WorkLifeBalance'])
        ),
    }

    return input_data


def encode_input(input_data: dict, label_encoders: dict):
    encoded = input_data.copy()
    encoded['Gender'] = 0 if encoded['Gender'] == 'Male' else 1
    encoded['OverTime'] = 0 if encoded['OverTime'] == 'No' else 1
    encoded['Over18'] = 1 if encoded['Over18'] == 'Y' else 0

    for column in ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']:
        encoded[column] = int(label_encoders[column].transform([encoded[column]])[0])

    return pd.DataFrame([encoded], columns=FEATURE_COLUMNS)


def plot_target_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Attrition', data=df, palette=['green', 'red'], ax=ax)
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_title('Attrition distribution')
    ax.set_xlabel('Attrition')
    ax.set_ylabel('Count')
    return fig


def plot_attrition_by_column(df: pd.DataFrame, label_encoders: dict, column: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = label_encoders[column].inverse_transform(sorted(df[column].unique()))
    sns.countplot(
        x=df[column].map(lambda x: label_encoders[column].inverse_transform([x])[0]),
        hue=df['Attrition'].map({0: 'No', 1: 'Yes'}),
        data=df,
        ax=ax,
        palette=['green', 'red'],
    )
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.legend(title='Attrition')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title='Employee Attrition Prediction', layout='wide')
    st.title('Employee Attrition Prediction')
    st.write(
        'This app is based on the Employee Attrition notebook and allows you to explore the data, ' 
        'view model metrics, and predict attrition interactively.'
    )

    df = load_data(DATA_PATH)
    df_processed, label_encoders = preprocess_data(df)
    model, metrics = train_model(df_processed)

    st.markdown('## Data preview')
    st.dataframe(df.head(5))

    with st.expander('Dataset summary'):
        st.write('Shape:', df.shape)
        st.write(df.describe(include='all'))
        st.write('Missing values:')
        st.write(df.isnull().sum())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Attrition counts')
        st.pyplot(plot_target_distribution(df_processed))
    with col2:
        st.subheader('Attrition by Department')
        st.pyplot(plot_attrition_by_column(df_processed, label_encoders, 'Department', 'Attrition by Department'))

    st.subheader('Model metrics')
    st.metric('Accuracy', f'{metrics["accuracy"]:.2%}')
    st.metric('ROC AUC', f'{metrics["roc_auc"]:.2f}')

    st.write('#### Confusion matrix')
    cm = metrics['confusion_matrix']
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    fpr, tpr, _ = metrics['fpr_tpr']
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.plot(fpr, tpr, label=f'AUC = {metrics["roc_auc"]:.2f}')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    st.pyplot(fig2)

    template = df_processed.drop(['Attrition'], axis=1).iloc[0]
    input_data = build_sidebar_inputs(label_encoders, template)
    encoded_input = encode_input(input_data, label_encoders)

    st.subheader('Predict attrition for a new employee')
    if st.button('Predict Attrition'):
        prediction = model.predict(encoded_input)[0]
        proba = model.predict_proba(encoded_input)[0, 1]
        result = 'Yes' if prediction == 1 else 'No'
        st.markdown(
            f"### Predicted Attrition: **{result}**  \n**Probability:** {proba:.2%}"
        )

        st.write('#### Input values')
        st.json(input_data)

    st.markdown('---')
    st.write('App built from the notebook workflow: data loading, cleaning, encoding, model training, and prediction.')


if __name__ == '__main__':
    main()
