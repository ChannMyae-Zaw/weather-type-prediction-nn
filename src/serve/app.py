from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
from typing import List
from src.serve.utils import load_model, preprocess_input
from src.utils.preprocessing import WeatherPreprocessor
import joblib
from fastapi.responses import HTMLResponse
from src.models.model import WeatherMLP
import os

# Replace with your run ID
# RUN_ID = "c96c3a23c6324bb68254903eb4d5e331"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Weather Type Prediction")

# Load model
torch.serialization.add_safe_globals([WeatherMLP])

# load with weights_only=False
model = torch.load(
    "models/model_full.pth",
    map_location=DEVICE,
    weights_only=False
)

model.to(DEVICE)
model.eval()

preprocessor = joblib.load("src/utils/weather_preprocessor.pkl")

# Feature schema
class InputData(BaseModel):
    temp: float
    humidity: float
    wind_speed: float
    precipitation: float
    cloud_cover: str
    atm: float
    uv_index: float
    season: str
    visibility: float
    location: str
    # Add other features as needed

# Wrapper for batch input
class BatchInput(BaseModel):
    data: List[InputData]

@app.post("/predict")
def predict(batch: BatchInput):
    # Convert list of Pydantic models -> list of dicts -> DataFrame
    df = pd.DataFrame([row.model_dump() for row in batch.data])

    # Optional: apply preprocessing pipeline if you saved it
    # X_tensor = preprocess_input(df, preprocessor, device=DEVICE)
    X_processed = preprocessor.transform_predict(df)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(DEVICE)

    with torch.inference_mode():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        labels = preprocessor.inverse_transform_target(preds)
        labels = list(labels)

    return {"predictions": labels}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
        <html>
        <head>
            <title>Weather Type Predictor</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                body {
                    background: linear-gradient(135deg, #1e3a8a, #60a5fa);
                    min-height: 100vh;
                    color: #ffffff;
                    font-family: 'Arial', sans-serif;
                }
                .card {
                    background: rgba(255, 255, 255, 0.95);
                    color: #1f2937;
                    border-radius: 1rem;
                    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
                }
                .input-field {
                    transition: all 0.3s ease;
                }
                .input-field:focus {
                    border-color: #3b82f6;
                    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
                }
                .btn-primary {
                    background: linear-gradient(to right, #3b82f6, #1d4ed8);
                    transition: transform 0.2s ease;
                }
                .btn-primary:hover {
                    transform: translateY(-2px);
                }
                .btn-remove, .btn-clear {
                    background: linear-gradient(to right, #ef4444, #b91c1c);
                }
                .btn-remove:hover, .btn-clear:hover {
                    transform: translateY(-2px);
                }
                .weather-icon {
                    font-size: 1.25rem;
                    margin-right: 0.5rem;
                }
                .table-container {
                    overflow-x: auto;
                    -webkit-overflow-scrolling: touch;
                }
                table {
                    width: 100%;
                    border-collapse: separate;
                    border-spacing: 0 0.25rem;
                    min-width: 900px;
                }
                th, td {
                    padding: 0.5rem;
                    text-align: left;
                    font-size: 0.75rem;
                    white-space: nowrap;
                }
                th {
                    background: #e5e7eb;
                    font-weight: 600;
                    font-size: 0.7rem;
                    text-transform: uppercase;
                }
                tr {
                    background: #f9fafb;
                    border-radius: 0.25rem;
                }
                td:last-child, th:last-child {
                    text-align: center;
                }
                /* Emphasize Prediction column */
                th:nth-child(11), td:nth-child(11) {
                    background: #dbeafe; /* Light blue background for visibility */
                    font-weight: bold;
                    font-size: 0.85rem; /* Slightly larger font */
                    color: #1e40af; /* Darker blue text for contrast */
                    padding: 0.75rem; /* Slightly more padding */
                }
                th:nth-child(11) {
                    font-size: 0.8rem; /* Slightly larger header font */
                }
                @media (min-width: 640px) {
                    th, td {
                        font-size: 0.875rem;
                    }
                    th {
                        font-size: 0.8rem;
                    }
                    th:nth-child(11), td:nth-child(11) {
                        font-size: 0.95rem; /* Larger font on bigger screens */
                    }
                    th:nth-child(11) {
                        font-size: 0.85rem;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container mx-auto p-4 sm:p-6 md:p-8">
                <div class="card max-w-4xl mx-auto p-6 sm:p-8">
                    <h2 class="text-2xl sm:text-3xl font-bold text-center mb-6 text-gray-800">
                        <i class="fas fa-cloud-sun weather-icon"></i> Weather Prediction
                    </h2>
                    <form id="input-form" class="space-y-4">
                        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-1">
                                    <i class="fas fa-thermometer-half weather-icon"></i> Temperature (°C)
                                </label>
                                <input type="number" step="1" min = "-50" max="50" class="input-field w-full p-2 border rounded-lg focus:outline-none" id="temp" required>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-1">
                                    <i class="fas fa-tint weather-icon"></i> Humidity (%)
                                </label>
                                <input type="number" step="1" min="0" max="100" class="input-field w-full p-2 border rounded-lg focus:outline-none" id="humidity" required>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-1">
                                    <i class="fas fa-wind weather-icon"></i> Wind Speed (m/s)
                                </label>
                                <input type="number" step="1" min="0" max="80" class="input-field w-full p-2 border rounded-lg focus:outline-none" id="wind_speed" required>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-1">
                                    <i class="fas fa-umbrella weather-icon"></i> Precipitation (mm)
                                </label>
                                <input type="number" step="1" min="0" max="100" class="input-field w-full p-2 border rounded-lg focus:outline-none" id="precipitation" required>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-1">
                                    <i class="fas fa-cloud weather-icon"></i> Cloud Cover
                                </label>
                                <select class="input-field w-full p-2 border rounded-lg focus:outline-none" id="cloud_cover" required>
                                    <option value="clear">Clear</option>
                                    <option value="partly cloudy">Partly Cloudy</option>
                                    <option value="overcast">Overcast</option>
                                    <option value="cloudy">Cloudy</option>
                                </select>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-1">
                                    <i class="fas fa-gauge weather-icon"></i> Atmospheric Pressure (hPa)
                                </label>
                                <input type="number" step="10" min="700" max="1200" class="input-field w-full p-2 border rounded-lg focus:outline-none" id="atm" required>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-1">
                                    <i class="fas fa-sun weather-icon"></i> UV Index
                                </label>
                                <input type="number" step="1" min="1" max="11" class="input-field w-full p-2 border rounded-lg focus:outline-none" id="uv_index" required>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-1">
                                    <i class="fas fa-leaf weather-icon"></i> Season
                                </label>
                                <select class="input-field w-full p-2 border rounded-lg focus:outline-none" id="season" required>
                                    <option value="Winter">Winter</option>
                                    <option value="Spring">Spring</option>
                                    <option value="Summer">Summer</option>
                                    <option value="Autumn">Autumn</option>
                                </select>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-1">
                                    <i class="fas fa-eye weather-icon"></i> Visibility (km)
                                </label>
                                <input type="number" step="0.1" min="0" max="20" class="input-field w-full p-2 border rounded-lg focus:outline-none" id="visibility" required>
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700 mb-1">
                                    <i class="fas fa-map-marker-alt weather-icon"></i> Location
                                </label>
                                <select class="input-field w-full p-2 border rounded-lg focus:outline-none" id="location" required>
                                    <option value="inland">Inland</option>
                                    <option value="mountain">Mountain</option>
                                    <option value="coastal">Coastal</option>
                                </select>
                            </div>
                        </div>
                        <button type="submit" class="btn-primary w-full py-3 rounded-lg text-white font-semibold text-lg">
                            <i class="fas fa-plus mr-2"></i> Add Data
                        </button>
                    </form>
                    <div class="mt-6">
                        <div class="flex justify-between items-center mb-4">
                            <h5 class="text-lg font-semibold text-gray-800">
                                <i class="fas fa-table weather-icon"></i> Added Data
                            </h5>
                            <button id="clear-predictions-btn" class="btn-clear py-2 px-4 rounded-lg text-white font-semibold hidden">
                                <i class="fas fa-eraser mr-2"></i> Clear Predictions
                            </button>
                        </div>
                        <div class="table-container">
                            <table id="data-table" class="hidden">
                                <thead>
                                    <tr>
                                        <th>Temp (°C)</th>
                                        <th>Humidity (%)</th>
                                        <th>Wind (m/s)</th>
                                        <th>Precip (mm)</th>
                                        <th>Cloud Cover</th>
                                        <th>Pressure (hPa)</th>
                                        <th>UV Index</th>
                                        <th>Season</th>
                                        <th>Visibility (km)</th>
                                        <th>Location</th>
                                        <th>Prediction</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody id="data-rows"></tbody>
                            </table>
                        </div>
                        <p id="no-data" class="text-gray-600 text-center">No data added yet.</p>
                        <button id="predict-btn" class="btn-primary w-full py-3 rounded-lg text-white font-semibold text-lg mt-4 hidden">
                            <i class="fas fa-cloud-sun-rain mr-2"></i> Predict Weather
                        </button>
                    </div>
                </div>
            </div>
            <script>
                let dataRows = [];

                function updateTable() {
                    const table = document.getElementById("data-table");
                    const noData = document.getElementById("no-data");
                    const predictBtn = document.getElementById("predict-btn");
                    const clearBtn = document.getElementById("clear-predictions-btn");
                    const tbody = document.getElementById("data-rows");
                    tbody.innerHTML = "";

                    if (dataRows.length === 0) {
                        table.classList.add("hidden");
                        noData.classList.remove("hidden");
                        predictBtn.classList.add("hidden");
                        clearBtn.classList.add("hidden");
                        return;
                    }

                    table.classList.remove("hidden");
                    noData.classList.add("hidden");
                    predictBtn.classList.remove("hidden");
                    if (dataRows.some(row => row.prediction)) {
                        clearBtn.classList.remove("hidden");
                    } else {
                        clearBtn.classList.add("hidden");
                    }

                    dataRows.forEach((row, index) => {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td>${row.temp}</td>
                            <td>${row.humidity}</td>
                            <td>${row.wind_speed}</td>
                            <td>${row.precipitation}</td>
                            <td>${row.cloud_cover}</td>
                            <td>${row.atm}</td>
                            <td>${row.uv_index}</td>
                            <td>${row.season}</td>
                            <td>${row.visibility}</td>
                            <td>${row.location}</td>
                            <td>${row.prediction || "Not predicted yet"}</td>
                            <td><button class="btn-remove py-1 px-2 rounded-lg text-white font-semibold" onclick="removeRow(${index})"><i class="fas fa-trash mr-1"></i> Remove</button></td>
                        `;
                        tbody.appendChild(tr);
                    });
                }

                function removeRow(index) {
                    dataRows.splice(index, 1);
                    updateTable();
                }

                function clearPredictions() {
                    dataRows.forEach(row => delete row.prediction);
                    updateTable();
                }

                document.getElementById("input-form").addEventListener("submit", (e) => {
                    e.preventDefault();

                    const row = {
                        temp: parseFloat(document.getElementById("temp").value),
                        humidity: parseFloat(document.getElementById("humidity").value),
                        wind_speed: parseFloat(document.getElementById("wind_speed").value),
                        precipitation: parseFloat(document.getElementById("precipitation").value),
                        cloud_cover: document.getElementById("cloud_cover").value,
                        atm: parseFloat(document.getElementById("atm").value),
                        uv_index: parseFloat(document.getElementById("uv_index").value),
                        season: document.getElementById("season").value,
                        visibility: parseFloat(document.getElementById("visibility").value),
                        location: document.getElementById("location").value
                    };

                    dataRows.push(row);
                    updateTable();
                    document.getElementById("input-form").reset();
                });

                document.getElementById("predict-btn").addEventListener("click", async () => {
                    const data = { data: dataRows.map(({ prediction, ...rest }) => rest) };

                    try {
                        const res = await fetch("/predict", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(data)
                        });
                        const json = await res.json();
                        if (json.predictions && Array.isArray(json.predictions)) {
                            json.predictions.forEach((pred, index) => {
                                if (dataRows[index]) {
                                    dataRows[index].prediction = pred;
                                }
                            });
                            updateTable();
                        } else {
                            document.getElementById("data-rows").innerHTML = `<tr><td colspan="12">Error: Invalid prediction response</td></tr>`;
                        }
                    } catch (error) {
                        document.getElementById("data-rows").innerHTML = `<tr><td colspan="12">Error: ${error.message}</td></tr>`;
                    }
                });

                document.getElementById("clear-predictions-btn").addEventListener("click", clearPredictions);

                updateTable();
            </script>
        </body>
        </html>
    """

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT if available
    uvicorn.run("src.serve.app:app", host="0.0.0.0", port=port, reload=True)