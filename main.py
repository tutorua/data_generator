import plotly.graph_objs as go
import numpy as np
import scipy.stats as stats
import pandas as pd
import yaml
import os
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi import FastAPI, Query, HTTPException, Form
from pydantic import BaseModel, Field
from typing import cast, Optional
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
    
    # Save YAML in flow style (compact list)
    yaml_path = output_dir / f"{dist_name}.yaml"
    with open(yaml_path, "w") as f:
        # wrap the output every few (4) items for readability
        # yaml.dump(data.tolist(), f, default_flow_style=True)
        # Ensure the YAML file is saved in a single line
        yaml.dump(data.tolist(), f, default_flow_style=True, width=float("inf"))
    

    # Save CSV for consistency
    csv_path = output_dir / f"{dist_name}.csv"
    pd.DataFrame(data, columns=[dist_name]).to_csv(csv_path, index=False)

    return {
        "yaml_file": str(yaml_path),
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
def preview_distribution(dist_name: str, page: int = 1, size: int = 20):
    file_path = output_dir / f"{dist_name}.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="CSV not found")

    df = pd.read_csv(file_path)
    values = df[dist_name].tolist()

    # Pagination
    start = (page - 1) * size
    end = start + size
    paginated = values[start:end]

    # Stats
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

    # Table
    html = f"<h2>Preview: {dist_name.title()} (Page {page})</h2>" + stats_html
    html += "<table border='1'><tr><th>Index</th><th>Value</th></tr>"
    for i, val in enumerate(paginated, start=start):
        html += f"<tr><td>{i}</td><td>{val:.6f}</td></tr>"
    html += "</table>"

    # Navigation
    nav_html = "<div style='margin-top: 1em;'>"
    if start > 0:
        nav_html += f"<a href='/preview/{dist_name}?page={page-1}&size={size}'>⏪ Previous</a> &nbsp;"
    if end < len(values):
        nav_html += f"<a href='/preview/{dist_name}?page={page+1}&size={size}'>Next ⏩</a>"
    nav_html += "</div>"

    return html + nav_html


@app.get("/plot/{dist_name}", response_class=HTMLResponse)
def plot_distribution(dist_name: str):
    file_path = output_dir / f"{dist_name}.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No data to plot.")

    df = pd.read_csv(output_dir / f"{dist_name}.csv")
    values = df[dist_name].tolist()


    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=50, marker_color="royalblue", name="Histogram"))
    fig.update_layout(title=f"{dist_name.title()} Distribution", xaxis_title="Value", yaxis_title="Count")

    return fig.to_html(full_html=True)


@app.get("/plot/pdf-cdf/{dist_name}", response_class=HTMLResponse)
def plot_pdf_cdf(dist_name: str):
    file_path = output_dir / f"{dist_name}.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No data to plot.")

    df = pd.read_csv(output_dir / f"{dist_name}.csv")
    values = df[dist_name].tolist()

    return create_pdf_cdf_plot(values, dist_name)


# histogram + PDF overlay with a toggle to switch the PDF curve on or off
@app.get("/plot/histogram/{dist_name}", response_class=HTMLResponse)
def plot_histogram_with_pdf(
    dist_name: str,
    show_pdf: bool = Query(True, description="Toggle PDF overlay on/off")
):
    file_path = output_dir / f"{dist_name}.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="No data to plot.")

    df = pd.read_csv(output_dir / f"{dist_name}.csv")
    values = df[dist_name].tolist()
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
    if ext not in ("csv", "yaml"):
        raise HTTPException(status_code=400, detail="Only .csv and .yaml supported")

    file_path = output_dir / f"{dist_name}.{ext}"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=file_path, filename=file_path.name, media_type="application/octet-stream")


# Enhanced HTML dashboard template
@app.get("/", response_class=HTMLResponse)
def dashboard():
    existing = sorted({f.stem for f in output_dir.glob("*.yaml")})

    html = """
    <html>
    <head>
        <title>📊 Data Generator Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="p-4">
        <div class="container">
            <h1 class="mb-4">📊 Data Generator Dashboard</h1>
            
            <h3>🎲 Generate New Distribution</h3>
            <form action="/generate-ui" method="post" class="row g-3 mb-5">
                <div class="col-md-3">
                    <label class="form-label">Distribution</label>
                    <select name="dist" class="form-select">
                        <option value="normal">Normal</option>
                        <option value="beta">Beta</option>
                        <option value="gamma">Gamma</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <label class="form-label">Param 1 (e.g. loc / a)</label>
                    <input type="text" name="param1" class="form-control" required>
                </div>
                <div class="col-md-2">
                    <label class="form-label">Param 2 (e.g. scale / b)</label>
                    <input type="text" name="param2" class="form-control">
                </div>
                <div class="col-md-2">
                    <label class="form-label">Size</label>
                    <input type="number" name="size" class="form-control" value="1000" required>
                </div>
                <div class="col-md-2 align-self-end">
                    <button type="submit" class="btn btn-primary w-100">Generate</button>
                </div>
            </form>
    """

    html += "<h3>📁 Available Datasets</h3><ul class='list-group'>"
    for dist in existing:
        html += f"""
            <li class="list-group-item">
                <strong>{dist.title()}</strong> &nbsp;
                <a href='/preview/{dist}'>🔍 Preview</a> &nbsp;
                <a href='/plot/histogram/{dist}?show_pdf=true'>📊 Histogram + PDF</a> &nbsp;
                <a href='/plot/pdf-cdf/{dist}'>📈 PDF & CDF</a> &nbsp;
                <a href='/download/{dist}.csv'>⬇️ CSV</a> &nbsp;
                <a href='/download/{dist}.yaml'>⬇️ YAML</a>
            </li>
        """
    html += "</ul></div></body></html>"
    return html


# Bootstrap-Enhanced HTML form
@app.get("/generate-ui", response_class=HTMLResponse)
def generate_ui_form():
    return """
    <html>
    <head>
        <title>Generate Distribution</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="p-4">
        <div class="container">
            <h2 class="mb-4">🎲 Generate Distribution</h2>
            <form action="/generate-ui" method="post" class="row g-3">
                <div class="col-md-4">
                    <label class="form-label">Distribution</label>
                    <select name="dist" class="form-select">
                        <option value="normal">Normal</option>
                        <option value="beta">Beta</option>
                        <option value="gamma">Gamma</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <label class="form-label">Param 1 (e.g. loc / a)</label>
                    <input type="text" name="param1" class="form-control" required>
                </div>
                <div class="col-md-2">
                    <label class="form-label">Param 2 (e.g. scale / b)</label>
                    <input type="text" name="param2" class="form-control">
                </div>
                <div class="col-md-2">
                    <label class="form-label">Size</label>
                    <input type="number" name="size" class="form-control" value="1000" required>
                </div>
                <div class="col-md-2 align-self-end">
                    <button type="submit" class="btn btn-primary w-100">Generate</button>
                </div>
            </form>
        </div>
    </body>
    </html>
    """


# POST handler for form submission
@app.post("/generate-ui", response_class=HTMLResponse)
def handle_form_submission(
    dist: str = Form(...),
    param1: float = Form(...),
    param2: float = Form(None),
    size: int = Form(...)
):
    try:
        if dist == "normal":
            data = stats.norm.rvs(loc=param1, scale=param2 or 1.0, size=size)
        elif dist == "beta":
            data = stats.beta.rvs(a=param1, b=param2 or 1.0, size=size)
        elif dist == "gamma":
            data = stats.gamma.rvs(a=param1, scale=param2 or 1.0, size=size)
        else:
            raise ValueError("Unsupported distribution")

        paths = generate_and_save(dist, np.asarray(data))

        return f"""
        <html><body class="p-4">
        <h3>{dist.title()} distribution generated!</h3>
        <ul>
            <li><a href="/preview/{dist}">🔍 Preview</a></li>
            <li><a href="/plot/histogram/{dist}?show_pdf=true">📊 Histogram + PDF</a></li>
            <li><a href="/plot/pdf-cdf/{dist}">📈 PDF & CDF</a></li>
            <li><a href="/download/{dist}.csv">⬇️ Download CSV</a></li>
            <li><a href="/download/{dist}.yaml">⬇️ Download YAML</a></li>
        </ul>
        <a href="/generate-ui">← Back to form</a>
        </body></html>
        """
    except Exception as e:
        return f"<h3>Error: {e}</h3><a href='/generate-ui'>Try again</a>"
    