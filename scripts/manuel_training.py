import torch
from torch.optim import Adam
from modelforge.dataset.qm9 import QM9Dataset
from modelforge.dataset.dataset import TorchDataModule
from modelforge.dataset.qm9 import QM9Dataset
from modelforge.potential import SchNet, ANI2x, PaiNN, PhysNet
from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
from tqdm import tqdm

# Assuming your dataset and model setup remains the same
data = QM9Dataset(for_unit_testing=False)
dataset = TorchDataModule(
    data, batch_size=128, splitting_strategy=FirstComeFirstServeSplittingStrategy()
)

dataset.prepare_data(remove_self_energies=True, normalize=False)

# Initialize model
model = SchNet().to(torch.float32)
# optimizer and loss function setup
optimizer = Adam(model.parameters(), lr=1e-3)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_nr_of_molecules = 0
    loss_function = torch.nn.MSELoss()

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # Adjust according to how your data is structured
        nnp_input = batch.nnp_input
        E_true = batch.metadata.E.to(torch.float32).squeeze(1)
        E_predict = model(nnp_input).E
        loss = loss_function(E_predict, E_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        nr_of_molecules_in_batch = batch.metadata.E.shape[0]
        total_nr_of_molecules += batch.metadata.E.shape[0]
        # progress_bar.set_postfix(
        #     {
        #         "loss": f"{loss.item():.4f}",
        #         "per_mol": f"{loss.item()/nr_of_molecules_in_batch:.4f}",
        #     }
        # )
    return total_loss / total_nr_of_molecules


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_nr_of_molecules = 0
    mean_absolute_error = torch.nn.L1Loss()

    with torch.no_grad():
        for batch in dataloader:
            # Adjust according to how your data is structured
            batch = batch.to(device)
            nnp_input = batch.nnp_input
            E_true = batch.metadata.E.to(torch.float32).squeeze(1)
            E_predict = model(nnp_input).E
            loss = mean_absolute_error(E_predict, E_true)
            total_loss += loss.item()
            total_nr_of_molecules += batch.metadata.E.shape[0]
    return total_loss / total_nr_of_molecules


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Main training loop
epochs = 100  # number of epochs
progress_bar = tqdm(range(epochs), desc="Train/Val", unit="batch")
for epoch in progress_bar:
    train_loss = train_one_epoch(
        model, dataset.train_dataloader(), optimizer, device
    )
    val_loss = validate(model, dataset.val_dataloader(), device)
    progress_bar.set_postfix(
        {
            "Epoch": f"{epoch}",
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
        }
    )
