import torch
import torch.nn as nn

x = torch.tensor([[1.0],
                  [2.0],
                  [3.0],
                  [4.0],
                  [5.0]])

y = torch.tensor([[3.0],
                  [5.0],
                  [7.0],
                  [9.0],
                  [11.0]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)
    
    def forward(self,x):
        return self.linear(x)
model=LinearRegressionModel()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
epochs=1000
for epoch in range(epochs):
    predictions = model(x)
    loss = criterion(predictions, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test
test = torch.tensor([[10.0]])
prediction = model(test)

print(f"Prediction for x=10: {prediction.item():.2f}")