import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataset import SciDataset
from model import GradingModel
import config


def train():

    train_data = load_dataset("csv", data_files="data/phase2/train/train.csv")["train"]
    test_data  = load_dataset("csv", data_files="data/phase2/test/test.csv")["train"]

    # print("Train size:", len(train_data))
    # print("Test size:", len(test_data))

    train_dataset = SciDataset(train_data)
    test_dataset  = SciDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    model = GradingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    loss_fn = torch.nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            labels = labels.unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch+1} Training Loss: {total_loss / len(train_loader)}")

        model.eval()
        test_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)

                test_loss += loss.item()

        print(f"Epoch {epoch+1} Test Loss: {test_loss / len(test_loader)}")

    torch.save(model.state_dict(), "model.pth")
    print("\nModel saved as model.pth")


if __name__ == "__main__":
    train()