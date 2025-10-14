import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from rasterio.features import shapes
from shapely.geometry import shape
from io import BytesIO

def read_raster(file):
    with rasterio.open(file) as src:
        return src.read(1), src.meta

def detect_anomalies(temp, deltaT=5.0):
    mean_temp = np.mean(temp)
    threshold = mean_temp + deltaT
    mask = temp > threshold
    n_labels = np.sum(mask)
    return mask, n_labels, threshold

def export_polygons(mask, meta):
    results = []
    for geom, val in shapes(mask.astype(np.int16), transform=meta["transform"]):
        if val == 1:
            results.append({"geometry": geom, "properties": {"anomaly": 1}})
    return {"type": "FeatureCollection", "features": results}

def generate_preview(temp, mask, threshold):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Base image: full thermal data in grayscale
    ax.imshow(temp, cmap="gray")

    # Overlay: actual temperature values where anomalies exist
    overlay = np.ma.masked_where(mask == 0, temp)
    im = ax.imshow(overlay, cmap="inferno", alpha=0.6)

    # Colorbar for overlay
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (relative)")

    ax.set_title(f"🔥 Thermal Anomalies (T > {threshold:.2f})")
    ax.axis("off")

    st.pyplot(fig)




st.set_page_config(page_title="Neighborhood Thermal Anomaly Detection", layout="wide")
st.title("🌡️ Neighborhood Thermal Anomaly Detection")

st.markdown("Upload a thermal GeoTIFF to detect heat anomalies.")

uploaded_file = st.file_uploader("Upload a .tif file", type=["tif", "tiff"])
deltaT = st.slider("Set ΔT (Temperature threshold above mean):", 0.5, 10.0, 5.0, 0.5)

if uploaded_file is not None:
    temp, meta = read_raster(uploaded_file)
    mask, n_labels, threshold = detect_anomalies(temp, deltaT)
    st.success(f"✅ Detected {n_labels} anomalous pixels (T > {threshold:.2f})")

    st.subheader("Heatmap Preview")
    generate_preview(temp, mask, threshold)

    st.subheader("Export Results")
    geojson_data = export_polygons(mask, meta)
    geojson_bytes = BytesIO(json.dumps(geojson_data).encode())

    st.download_button("📍 Download GeoJSON", geojson_bytes, file_name="anomalies.geojson", mime="application/geo+json")
