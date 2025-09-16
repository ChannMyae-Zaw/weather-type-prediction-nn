import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import yaml
import mlflow
import mlflow.pytorch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.data.etl import load_data, transform_data
from src.models.model import WeatherMLP
import joblib


def main(lr, batch_size, hidden_dim, epochs, device, seed, log_interval):
    df = load_data()
    target_col = "Weather Type"
    num_cols = df.drop(target_col, axis=1).select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.drop(target_col, axis=1).select_dtypes(include=['object']).columns.tolist()
    skew_threshold = 0.8
    skewed_cols = [col for col in num_cols if abs(df[col].skew()) > skew_threshold]

    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)["train"]

    seed = seed or config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = device or config["device"] if torch.cuda.is_available() else "cpu"

    df_train, df_temp = train_test_split(df, test_size=0.4, random_state=seed, stratify=df[target_col])
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed, stratify=df_temp[target_col])

    X_train_tensor, y_train_tensor, preprocessor = transform_data(
        df_train, num_cols, skewed_cols, cat_cols, target_col, device=device
    )
    X_val_tensor, y_val_tensor, _ = transform_data(
        df_val, num_cols, skewed_cols, cat_cols, target_col, preprocessor=preprocessor, device=device
    )
    X_test_tensor, y_test_tensor, _ = transform_data(
        df_test, num_cols, skewed_cols, cat_cols, target_col, preprocessor=preprocessor, device=device
    )
    joblib.dump(preprocessor, "src/utils/weather_preprocessor.pkl")
    print("Preprocessor saved successfully!")

    batch_size = batch_size or config["batch_size"]
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    input_dim = X_train_tensor.shape[1]
    hidden_dim = hidden_dim or config["hidden_dim"]
    output_dim = len(preprocessor.label_encoder.classes_)

    model = WeatherMLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = lr or config["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim).to(device)
    val_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim).to(device)
    test_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim).to(device)
    epochs = epochs or config["epochs"]
    log_interval = log_interval or config.get("log_interval", 10)

    with mlflow.start_run(run_name="Weather_MLP"):
        mlflow.log_params({"epochs": epochs, "hidden_dim": hidden_dim, "batch_size": batch_size,
                           "learning_rate": lr, "device": device, "random_seed": seed})

        for epoch in range(epochs):
            model.train()
            train_acc_metric.reset()
            running_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(outputs, dim=1)
                train_acc_metric(preds, y_batch)

            epoch_acc = train_acc_metric.compute().item()
            print(f"Epoch {epoch + 1}/{epochs} — Loss: {running_loss / len(train_loader.dataset):.4f}, Train Accuracy: {epoch_acc:.4f}")

            model.eval()
            val_acc_metric.reset()
            with torch.inference_mode():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    preds = torch.argmax(outputs, dim=1)
                    val_acc_metric.update(preds, y_batch)
            val_acc = val_acc_metric.compute().item()
            print(f"Epoch {epoch + 1}/{epochs} — Validation Accuracy: {val_acc:.4f}")

            if epoch % log_interval == 0:
                mlflow.log_metrics({"train_loss": running_loss, "train_acc": epoch_acc, "val_acc": val_acc}, step=epoch)

        # Final test evaluation
        model.eval()
        test_acc_metric.reset()
        with torch.inference_mode():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                test_acc_metric.update(preds, y_batch)
        test_acc = test_acc_metric.compute().item()
        mlflow.log_metric("test_acc", test_acc)
        torch.save(model, "models/model_full.pth")
        mlflow.pytorch.log_model(model, "weather_mlp_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Weather Classification Model")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    args = parser.parse_args()

    main(args.lr, args.batch_size, args.hidden_dim, args.epochs, args.device, args.seed, args.log_interval)
