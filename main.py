import plotly.graph_objs as go
import numpy as np
import scipy.stats as stats
import pandas as pd
import yaml
import json
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import cast
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

# helper function to compute and plot both PDF and CDF
# This function uses scipy's gaussian_kde for PDF estimation and numpy for CDF calculation

def create_pdf_cdf_plot(values: list, dist_name: str) -> str:
    arr = np.array(values)
    x = np.linspace(arr.min(), arr.max(), 500)

    # Estimate PDF using KDE
    kde = stats.gaussian_kde(arr)
    pdf_y = kde(x)

    # Compute empirical CDF
    sorted_vals = np.sort(arr)
    cdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    fig = go.Figure()

    # PDF line
    fig.add_trace(go.Scatter(x=x, y=pdf_y, mode="lines", name="PDF", line=dict(color="royalblue")))

    # CDF line
    fig.add_trace(go.Scatter(x=sorted_vals, y=cdf_y, mode="lines", name="CDF", line=dict(color="orange")))

    fig.update_layout(
        title=f"{dist_name.title()} - PDF & CDF",
        xaxis_title="Value",
        yaxis_title="Probability",
        legend=dict(x=0.8, y=0.95),
        height=500
    )

    return fig.to_html(full_html=True)


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


# Normal endpoint
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


# Beta endpoint
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


# Gamma endpoint
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
def preview_distribution(dist_name: str, page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):

    file_path = output_dir / f"{dist_name}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"No data found for '{dist_name}'")

    # with open(file_path) as f:
    #     data = json.load(f)

    # Load and extract values
    data = pd.read_json(file_path)
    values = data[dist_name].tolist() if dist_name in data.columns else data.iloc[:, 0].tolist()

    # Pagination
    start = (page - 1) * size
    end = start + size
    paginated = values[start:end]

    # Metadata
    arr = np.array(values)
    stats_html = f"""
    <h3>Summary Stats</h3>
    <ul>
        <li>Count: {len(arr)}</li>
        <li>Mean: {arr.mean():.4f}</li>
        <li>Std Dev: {arr.std():.4f}</li>
        <li>Min: {arr.min():.4f}</li>
        <li>Max: {arr.max():.4f}</li>
    </ul>
    """

    # Build HTML table
    html = f"<h2>Preview: {dist_name.title()} (Page {page})</h2>" + stats_html
    html += "<table border='1'><tr><th>Index</th><th>Value</th></tr>"
    for i, val in enumerate(paginated, start=start):
        html += f"<tr><td>{i}</td><td>{val:.6f}</td></tr>"
    html += "</table>"

    # Navigation
    next_page = f"/preview/{dist_name}?page={page+1}&size={size}"
    prev_page = f"/preview/{dist_name}?page={page-1}&size={size}" if page > 1 else None
    nav_html = "<div style='margin-top: 1em;'>"
    if prev_page:
        nav_html += f"<a href='{prev_page}'>⏪ Previous</a> &nbsp;&nbsp;"
    if end < len(values):
        nav_html += f"<a href='{next_page}'>Next ⏩</a>"
    nav_html += "</div>"

    return html + nav_html


@app.get("/plot/{dist_name}", response_class=HTMLResponse)
def plot_distribution(dist_name: str):
    file_path = output_dir / f"{dist_name}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No data to plot.")

    df = pd.read_json(file_path)
    values = df[dist_name] if dist_name in df.columns else df.iloc[:, 0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=50, marker_color="royalblue", name="Histogram"))
    fig.update_layout(title=f"{dist_name.title()} Distribution", xaxis_title="Value", yaxis_title="Count")

    return fig.to_html(full_html=True)


@app.get("/plot/pdf-cdf/{dist_name}", response_class=HTMLResponse)
def plot_pdf_cdf(dist_name: str):
    file_path = output_dir / f"{dist_name}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No data to plot.")

    df = pd.read_json(file_path)
    values = df[dist_name] if dist_name in df.columns else df.iloc[:, 0]

    return create_pdf_cdf_plot(values.tolist(), dist_name)


# histogram + PDF overlay with a toggle to switch the PDF curve on or off
@app.get("/plot/histogram/{dist_name}", response_class=HTMLResponse)
def plot_histogram_with_pdf(
    dist_name: str,
    show_pdf: bool = Query(True, description="Toggle PDF overlay on/off")
):
    file_path = output_dir / f"{dist_name}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No data to plot.")

    df = pd.read_json(file_path)
    values = df[dist_name] if dist_name in df.columns else df.iloc[:, 0]
    arr = np.array(values)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=arr,
        nbinsx=50,
        name="Histogram",
        marker_color="lightsteelblue",
        opacity=0.75
    ))

    # Optional PDF overlay
    if show_pdf:
        x = np.linspace(arr.min(), arr.max(), 500)
        kde = stats.gaussian_kde(arr)
        pdf_y = kde(x)

        # Scale PDF to match histogram height
        scale_factor = len(arr) * (arr.max() - arr.min()) / 50
        pdf_y_scaled = pdf_y * scale_factor

        fig.add_trace(go.Scatter(
            x=x,
            y=pdf_y_scaled,
            mode="lines",
            name="PDF",
            line=dict(color="crimson", width=2)
        ))

    fig.update_layout(
        title=f"{dist_name.title()} - Histogram {'+ PDF' if show_pdf else ''}",
        xaxis_title="Value",
        yaxis_title="Count",
        barmode="overlay",
        height=500
    )

    return fig.to_html(full_html=True)


@app.get("/download/{dist_name}.{ext}")
def download_data(dist_name: str, ext: str):
    if ext not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="Only .csv and .json supported")

    file_path = output_dir / f"{dist_name}.{ext}"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=file_path, filename=file_path.name, media_type="application/octet-stream")
