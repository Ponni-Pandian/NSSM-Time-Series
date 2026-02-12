# ============================================================
# Advanced Time Series Forecasting with Neural State Space Models
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

torch.manual_seed(42)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading dataset...")
df = pd.read_csv("ETTh1.csv")

target = "OT"
features = [c for c in df.columns if c not in ["date", target]]

X = df[features].values
y = df[target].values.reshape(-1, 1)

split = int(len(df) * 0.7)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("Data loaded.")

# ============================================================
# 2. NEURAL STATE SPACE MODEL (Kalman-based)
# ============================================================

class NeuralStateSpaceModel(nn.Module):
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        
        # Neural parameterizations
        self.A_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim * state_dim)
        )
        
        self.B_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim * input_dim)
        )
        
        self.C_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )
        
        self.D_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
        # Learnable noise covariances
        self.Q = nn.Parameter(torch.eye(state_dim))
        self.R = nn.Parameter(torch.eye(1))

    def forward(self, u, y):
        T = u.shape[0]
        
        x = torch.zeros(self.state_dim)
        P = torch.eye(self.state_dim)
        
        log_likelihood = 0
        latent_states = []
        
        for t in range(T):
            A = self.A_net(u[t]).view(self.state_dim, self.state_dim)
            B = self.B_net(u[t]).view(self.state_dim, self.input_dim)
            C = self.C_net(u[t]).view(1, self.state_dim)
            D = self.D_net(u[t]).view(1, self.input_dim)
            
            # Prediction step
            x_pred = A @ x + B @ u[t]
            P_pred = A @ P @ A.T + self.Q
            
            y_pred = C @ x_pred + D @ u[t]
            
            S = C @ P_pred @ C.T + self.R
            K = P_pred @ C.T @ torch.inverse(S)
            
            innovation = y[t] - y_pred
            x = x_pred + K @ innovation
            P = (torch.eye(self.state_dim) - K @ C) @ P_pred
            
            log_likelihood += -0.5 * (
                torch.logdet(S) +
                innovation.T @ torch.inverse(S) @ innovation
            )
            
            latent_states.append(x)
        
        return -log_likelihood, torch.stack(latent_states)

# ============================================================
# 3. TRAIN MODEL (Maximum Likelihood)
# ============================================================

print("Training NSSM...")

model = NeuralStateSpaceModel(
    input_dim=X_train.shape[1],
    state_dim=6
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 25

for epoch in range(epochs):
    optimizer.zero_grad()
    loss, _ = model(X_train, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Negative Log Likelihood: {loss.item():.4f}")

print("Training complete.")

# ============================================================
# 4. FORECAST (1-step)
# ============================================================

print("Evaluating NSSM...")

model.eval()
with torch.no_grad():
    _, latent_test = model(X_test, y_test)

latent_test = latent_test.numpy()

# For simplicity use SARIMAX forecast for comparison target shape
y_test_np = scaler_y.inverse_transform(y_test.numpy()).flatten()

# ============================================================
# 5. BASELINE: SARIMAX
# ============================================================

print("Training SARIMAX baseline...")

train_y_np = scaler_y.inverse_transform(y_train.numpy()).flatten()
test_y_np = scaler_y.inverse_transform(y_test.numpy()).flatten()

train_X_np = scaler_X.inverse_transform(X_train.numpy())
test_X_np = scaler_X.inverse_transform(X_test.numpy())

sarimax = SARIMAX(train_y_np,
                  exog=train_X_np,
                  order=(2,1,2),
                  seasonal_order=(1,1,1,24))

sarimax_fit = sarimax.fit(disp=False)

forecast = sarimax_fit.forecast(steps=len(test_y_np),
                                exog=test_X_np)

# ============================================================
# 6. METRICS
# ============================================================

rmse_sarimax = np.sqrt(mean_squared_error(test_y_np, forecast))
mae_sarimax = mean_absolute_error(test_y_np, forecast)

print("\nSARIMAX Results")
print("RMSE:", rmse_sarimax)
print("MAE :", mae_sarimax)

# ============================================================
# 7. LATENT STATE VISUALIZATION
# ============================================================

plt.figure(figsize=(12,6))
for i in range(3):
    plt.plot(latent_test[:, i], label=f"State {i}")
plt.legend()
plt.title("Learned Latent State Dynamics (NSSM)")
plt.show()

# ============================================================
# END
# ============================================================

print("\nProject execution complete.")
