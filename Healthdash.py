# Healthcare Dashboard with personalized recommendations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
import torch
import time
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import streamlit as st
import plotly.express as px



# Load dataset
df = pd.read_csv('https://github.com/MegaaCV/AI-HEALTHCARE-DASHBOARD/blob/main/cleaned_dataset_350_rows.csv')

# Static features used for XGBoost
static_features = [
    'Patient_ID', 'Patient_Age', 'Patient_Gender', 'Blood Type',
    'Medical Condition', 'Procedure', 'Cost', 'Length_of_Stay',
    'Readmission', 'Outcome', 'Satisfaction'
]

# Drop rows with missing values in static part
df_static = df[static_features].dropna()

# Convert Patient_ID to integer
df_static['Patient_ID'] = df_static['Patient_ID'].astype(int)

# Encode categorical columns
label_encoders = {}
for col in df_static.select_dtypes(include='object'):
    le = LabelEncoder()
    df_static[col] = le.fit_transform(df_static[col])
    label_encoders[col] = le

# Define X and y
X = df_static.drop(columns=['Outcome'])
y = df_static['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = xgb_model.predict(X_test)
report = classification_report(y_test, y_pred)

print("Classification Report:\n")
print(report)
# GRU model
# Time-series columns
categorical_cols = ['Symptom_Gender', 'Fever', 'Cough', 'Fatigue',
                    'Difficulty Breathing', 'Blood Pressure', 'Cholesterol Level']
numerical_cols = ['Symptom_Age']

df = df.dropna(subset=['Patient_ID'])
df['Patient_ID'] = df['Patient_ID'].astype(int)
grouped = df.groupby('Patient_ID')

# One-hot encode categorical features
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(df[categorical_cols])

sequences, labels = [], []
for pid, group in grouped:
    cat_enc = ohe.transform(group[categorical_cols])
    num_vals = group[numerical_cols].values
    seq = np.hstack((num_vals, cat_enc))
    sequences.append(seq)
    label = 1 if group['Outcome Variable'].iloc[-1] == 'Positive' else 0
    labels.append(label)

# Pad sequences to max length
max_len = max(len(seq) for seq in sequences)
input_dim = sequences[0].shape[1]
padded_sequences = np.zeros((len(sequences), max_len, input_dim), dtype=np.float32)
for i, seq in enumerate(sequences):
    padded_sequences[i, :len(seq), :] = seq
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, stratify=labels, random_state=42
)

# PyTorch Dataset
class PatientDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = PatientDataset(X_train, y_train)
test_ds = PatientDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# GRU Model
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n.squeeze(0))
        return self.sigmoid(out)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUNet(input_size=input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "gru_model_pytorch.pt")

#CMAB model for decision making

# cmab_recommender.py

class CMABThompsonSampler:
    def __init__(self, actions):
        self.actions = actions
        self.alpha = defaultdict(lambda: np.ones(len(actions)))
        self.beta = defaultdict(lambda: np.ones(len(actions)))

    def recommend(self, context_key):
        samples = np.random.beta(self.alpha[context_key], self.beta[context_key])
        return self.actions[np.argmax(samples)]

    def update(self, context_key, action, reward):
        action_idx = self.actions.index(action)
        if reward:
            self.alpha[context_key][action_idx] += 1
        else:
            self.beta[context_key][action_idx] += 1

# Example use
actions = ['Angioplasty', 'Insulin Therapy', 'X-Ray and Splint']
cmab = CMABThompsonSampler(actions)

context_key = 'Male_Obesity'
recommended = cmab.recommend(context_key)
cmab.update(context_key, recommended, reward=1)

# Generate synthetic data

np.random.seed(42)
n = 100
synthetic_data = pd.DataFrame({
    "Patient_ID": np.arange(1001, 1001 + n),
    "Patient_Age": np.random.randint(20, 90, size=n),
    "Patient_Gender": np.random.choice(["Male", "Female"], size=n),
    "Medical Condition": np.random.choice(["Cancer", "Diabetes", "Obesity", "Heart Disease"], size=n),
    "Procedure": np.random.choice(["Angioplasty", "Insulin Therapy", "X-Ray and Splint"], size=n),
    "Cost": np.random.randint(500, 30000, size=n),
    "Satisfaction": np.random.randint(1, 6, size=n)
})

synthetic_data.to_csv("synthetic_new_patient_data.csv", index=False)

# Creating Dashboard

st.set_page_config(layout='wide', page_title="Live Patient Dashboard", page_icon="📊")

st.title("📊 Real-Time Patient Monitoring Dashboard ")

@st.cache_data
def load_data():
    return pd.read_csv('https://github.com/MegaaCV/AI-HEALTHCARE-DASHBOARD/blob/main/synthetic_new_patient_data.csv')

data = load_data()
arms = data['Procedure'].unique().tolist()
context_policies = {}

placeholder = st.empty()
for i in range(len(data)):
    row = data.iloc[i]
    context_key = f"{row['Patient_Gender']}_{row['Medical Condition']}"

    # Use our custom Thompson Sampling recommender
    if context_key not in context_policies:
        context_policies[context_key] = CMABThompsonSampler(arms)

    sampler = context_policies[context_key]
    chosen_procedure = sampler.recommend(context_key)
    reward = 1 if row['Satisfaction'] >= 4 else 0
    sampler.update(context_key, chosen_procedure, reward)

    current_data = data.iloc[:i+1]

    with placeholder.container():
        st.subheader("📈 Visual Analytics")
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.line(current_data, x=current_data.index, y='Cost', title="Cost Trend", template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.pie(current_data, names='Procedure', title='Procedure Distribution', template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = px.histogram(current_data, x='Patient_Age', nbins=10, title="Age Distribution", template="plotly_dark")
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = px.box(current_data, x='Patient_Gender', y='Cost', title='Cost by Gender', template="plotly_dark")
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("📋 Latest Patient Data")
        st.dataframe(current_data.tail(5), use_container_width=True)

        st.subheader("📊 Procedure Summary Table")
        summary_table = current_data.groupby('Procedure')[['Cost', 'Satisfaction']].mean().round(2)
        st.dataframe(summary_table, use_container_width=True)

        st.subheader("💡 Personalized Recommendation")
        st.markdown(f"**Recommended Procedure:** `{chosen_procedure}` for context `{context_key}`")

        time.sleep(2)
