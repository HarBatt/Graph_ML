import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from rgat import RGAT
from shared.component_logger import component_logger as logger

dataset = Entities(root='/tmp/Entities', name='AIFB')
model_save_path = './checkpoints/rgat.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_features = dataset.num_node_features
hidden_features = 16
num_classes = dataset.num_classes


data = dataset[0].to(device)
data.x = torch.randn(data.num_nodes, 16).to(device)

model = RGAT(16, 16, dataset.num_classes, dataset.num_relations).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


logger.log("Started training")

model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_type)
    #_, out = torch.max(out, 1)
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    logger.log("NLL Loss: {}".format(loss.item()))
    loss.backward()
    optimizer.step()


torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}, model_save_path)



with torch.no_grad():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
    train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
    test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    logger.log("test_acc: {}".format(test_acc))


