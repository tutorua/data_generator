from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import cast
import numpy as np
import scipy.stats as stats
import pandas as pd
import yaml
import json
from typing import Optional
from pathlib import Path

app = FastAPI()

# Load YAML config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 📦 Output directory
output_dir = Path("generated_data")
output_dir.mkdir(exist_ok=True)

# 🧪 Pydantic models for validation
class NormalParams(BaseModel):
    loc: float = Field(0)
    scale: float = Field(1, gt=0)
    size: int = Field(1000, gt=0, le=100000)

class BetaParams(BaseModel):
    a: float = Field(..., gt=0)
    b: float = Field(..., gt=0)
    size: int = Field(1000, gt=0, le=100000)

class GammaParams(BaseModel):
    a: float = Field(..., gt=0)
    scale: float = Field(1, gt=0)
    size: int = Field(1000, gt=0, le=100000)

# 🔧 Utility to generate and save data
""" def generate_and_save(dist_name: str, data: np.ndarray) -> str:
    file_path = output_dir / f"{dist_name}.json"
    with open(file_path, "w") as f:
        json.dump(data.tolist(), f)
    return str(file_path) """


def generate_and_save(dist_name: str, data: np.ndarray) -> dict:
    output_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(data, columns=[dist_name])

    json_path = output_dir / f"{dist_name}.json"
    csv_path = output_dir / f"{dist_name}.csv"

    df.to_json(json_path, orient="records")
    df.to_csv(csv_path, index=False)

    return {
        "json_file": str(json_path),
        "csv_file": str(csv_path)
    }


# 🔹 Normal endpoint
@app.get("/generate/normal")
def generate_normal(
    loc: float = Query(config["distributions"]["normal"]["params"]["loc"]),
    scale: float = Query(config["distributions"]["normal"]["params"]["scale"]),
    size: int = Query(config["distributions"]["normal"]["params"]["size"])
):
    params = NormalParams(loc=loc, scale=scale, size=size)
    data = cast(np.ndarray, stats.norm.rvs(loc=params.loc, scale=params.scale, size=params.size))
    path = generate_and_save("normal", data)
    return {"distribution": "normal", "file": path, "preview": data[:10].tolist()}

# 🔹 Beta endpoint
@app.get("/generate/beta")
def generate_beta(
    a: float = Query(config["distributions"]["beta"]["params"]["a"]),
    b: float = Query(config["distributions"]["beta"]["params"]["b"]),
    size: int = Query(config["distributions"]["beta"]["params"]["size"])
):
    params = BetaParams(a=a, b=b, size=size)
    data = cast(np.ndarray, stats.beta.rvs(a=params.a, b=params.b, size=params.size))
    path = generate_and_save("beta", data)
    return {"distribution": "beta", "file": path, "preview": data[:10].tolist()}

# 🔹 Gamma endpoint
@app.get("/generate/gamma")
def generate_gamma(
    a: float = Query(config["distributions"]["gamma"]["params"]["a"]),
    scale: float = Query(config["distributions"]["gamma"]["params"]["scale"]),
    size: int = Query(config["distributions"]["gamma"]["params"]["size"])
):
    params = GammaParams(a=a, scale=scale, size=size)
    data = cast(np.ndarray, stats.gamma.rvs(a=params.a, scale=params.scale, size=params.size))
    path = generate_and_save("gamma", data)
    return {"distribution": "gamma", "file": path, "preview": data[:10].tolist()}



@app.get("/preview/{dist_name}", response_class=HTMLResponse)
def preview_distribution(dist_name: str):
    file_path = output_dir / f"{dist_name}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"No data found for '{dist_name}'")

    with open(file_path) as f:
        data = json.load(f)

    # Extract the actual number from the dict: {'gamma': 4.12} → 4.12
    values = [row.get(dist_name, row) if isinstance(row, dict) else row for row in data[:20]]

    # Build HTML table
    html = f"<h2>Preview: {dist_name.title()}</h2><table border='1'><tr><th>Index</th><th>Value</th></tr>"
    for i, val in enumerate(values):
        html += f"<tr><td>{i}</td><td>{val}</td></tr>"
    html += "</table>"
    return html



@app.get("/download/{dist_name}.{ext}")
def download_data(dist_name: str, ext: str):
    if ext not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="Only .csv and .json supported")

    file_path = output_dir / f"{dist_name}.{ext}"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=file_path, filename=file_path.name, media_type="application/octet-stream")
