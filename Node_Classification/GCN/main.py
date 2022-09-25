import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from gcn import GCN
from shared.component_logger import component_logger as logger

dataset = Planetoid(root='/tmp/Cora', name='Cora')
model_save_path = './checkpoints/gcn.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_features = dataset.num_node_features
hidden_features = 16
num_classes = dataset.num_classes

model = GCN(input_features, hidden_features, num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(500):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    logger.log("NLL Loss: {}".format(loss.item()))
    loss.backward()
    optimizer.step()


torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}, model_save_path)



model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())


logger.log("Accuracy: {}".format(acc))


