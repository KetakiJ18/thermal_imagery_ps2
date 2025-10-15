# ------------------ Imports ------------------
import os
import torch
import numpy as np
import pandas as pd
import rasterio
import tifffile
import streamlit as st
import matplotlib.pyplot as plt
import json
from io import BytesIO
from rasterio.features import shapes
from shapely.geometry import shape
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ------------------ Helper: Safe OpenCV Import ------------------
try:
    import cv2
except ImportError:
    cv2 = None

# ------------------ Helper: Download ESRGAN weights ------------------
def get_esrgan_weights():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "RealESRGAN_x4plus.pth")

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        st.info("🔽 Downloading RealESRGAN weights (~67MB)... Please wait...")
        import urllib.request
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        urllib.request.urlretrieve(url, model_path)
        st.success("✅ RealESRGAN weights downloaded successfully!")

    return model_path


# ------------------ Feature 1: Thermal Anomaly Detection ------------------
def feature_one():
    st.title("Neighborhood Thermal Anomaly Detection")
    st.markdown("Upload a thermal GeoTIFF to detect heat anomalies.")

    uploaded_file = st.file_uploader("Upload a .tif file", type=["tif", "tiff"], key="f1_uploader")
    deltaT = st.slider("Set ΔT (Temperature threshold above mean):", 0.5, 10.0, 5.0, 0.5, key="f1_deltaT")

    if uploaded_file:
        try:
            with rasterio.open(BytesIO(uploaded_file.getvalue())) as src:
                temp = src.read(1)
                meta = src.meta
        except Exception as e:
            st.error(f"Error reading TIFF: {e}")
            return

        mean_temp = np.mean(temp)
        threshold = mean_temp + deltaT
        mask = temp > threshold
        n_labels = np.sum(mask)
        st.success(f"Detected {n_labels} anomalous pixels (T > {threshold:.2f})")

        # Preview
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(temp, cmap="gray")
        overlay = np.ma.masked_where(mask == 0, temp)
        im = ax.imshow(overlay, cmap="inferno", alpha=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Temperature (relative)")
        ax.set_title(f"Thermal Anomalies (T > {threshold:.2f})")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

        # Export as GeoJSON
        results = []
        for geom, val in shapes(mask.astype(np.int16), transform=meta["transform"]):
            if val == 1:
                results.append({"geometry": geom, "properties": {"anomaly": 1}})
        geojson_data = {"type": "FeatureCollection", "features": results}
        geojson_bytes = BytesIO(json.dumps(geojson_data).encode())
        st.download_button("Download GeoJSON", geojson_bytes, file_name="anomalies.geojson", mime="application/geo+json")


# ------------------ Feature 2: Land Cover & Temperature Analysis ------------------
def feature_two():
    st.title("Land Cover & Temperature Analysis")
    uploaded_file = st.file_uploader("Upload a .tif file (e.g., Landsat 8)", type=["tif"], key="f2_uploader")

    if uploaded_file:
        temp_file_path = "temp_input_f2.tif"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded!")

        try:
            with rasterio.open(temp_file_path) as src:
                bands = src.read()
                if bands.shape[0] < 10:
                    st.error(f"Error: Only {bands.shape[0]} bands found. Need >=10 for Landsat 8.")
                    return
                red, green, nir, swir, thermal = bands[3], bands[2], bands[4], bands[5], bands[9]

            ndvi = (nir - red) / (nir + red + 1e-6)
            ndwi = (green - nir) / (green + nir + 1e-6)
            ndbi = (swir - nir) / (swir + nir + 1e-6)

            landcover = np.zeros_like(ndvi, dtype=np.uint8)
            landcover[ndvi > 0.4] = 1
            landcover[ndwi > 0.3] = 2
            landcover[ndbi > 0.2] = 3
            landcover[landcover == 0] = 4

            colors = {1: [0, 128, 0], 2: [0, 0, 255], 3: [128, 128, 128], 4: [210, 180, 140]}
            rgb_lc = np.zeros((*landcover.shape, 3), dtype=np.uint8)
            for c, col in colors.items():
                rgb_lc[landcover == c] = col
            st.subheader("Land Cover Map")
            st.image(rgb_lc, caption="Land Cover Map", use_column_width=True)

            # Compute temperature stats
            stats = []
            classes = {1: "Vegetation", 2: "Water", 3: "Urban", 4: "Bare Soil"}
            for c, name in classes.items():
                mask = landcover == c
                temps = thermal[mask]
                if temps.size > 0:
                    stats.append({
                        "Land Cover": name,
                        "Pixel Count": int(np.sum(mask)),
                        "Mean Temp": float(np.mean(temps)),
                        "Max Temp": float(np.max(temps)),
                        "Min Temp": float(np.min(temps)),
                        "Std Dev": float(np.std(temps)),
                        "Median": float(np.median(temps))
                    })
            df = pd.DataFrame(stats)
            st.subheader("Temperature Statistics")
            if not df.empty:
                st.dataframe(df)
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv_bytes, "landcover_temp_stats.csv", "text/csv")
            else:
                st.info("No land cover detected.")

        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


# ------------------ Feature 3: ESRGAN Super-Resolution ------------------
def feature_three():
    st.title("🌈 Satellite Image Super-Resolution with ESRGAN")
    uploaded_file = st.file_uploader("Upload a .tif file", type=["tif"], key="f3_uploader")

    if uploaded_file:
        input_tif = f"/tmp/{uploaded_file.name}"
        with open(input_tif, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info(f"📁 Processing file: {uploaded_file.name}")

        img = tifffile.imread(input_tif)
        if img.ndim == 2:
            if cv2 is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img_rgb = img[:, :, :3]
        else:
            st.error("❌ Unsupported image format.")
            return

        # Normalize to 0-255 range
        img_rgb = ((img_rgb - np.min(img_rgb)) / (np.max(img_rgb) - np.min(img_rgb)) * 255).astype(np.uint8)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = get_esrgan_weights()

        # Define and load ESRGAN safely
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        state_dict = torch.load(model_path, map_location=device)
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        model.load_state_dict(state_dict, strict=False)

        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device
        )

        with st.spinner("⏳ Enhancing image..."):
            output, _ = upsampler.enhance(img_rgb, outscale=4)
        st.success("🎉 Enhancement complete!")

        output_tif = f"/tmp/enhanced_{uploaded_file.name}"
        tifffile.imwrite(output_tif, np.uint8(output))
        st.image([img_rgb, output], caption=["Original", "Enhanced"], use_column_width=True)
        st.download_button("⬇️ Download Enhanced TIFF", data=open(output_tif, "rb"),
                           file_name=f"enhanced_{uploaded_file.name}")


# ------------------ Main App ------------------
def main():
    st.set_page_config(page_title="Geospatial Analysis App", layout="wide")
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio(
        "",
        [
            "Thermal Anomaly Detection",
            "Land Cover & Temperature Analysis",
            "Satellite Image Super-Resolution",
        ],
    )

    if selection == "Thermal Anomaly Detection":
        feature_one()
    elif selection == "Land Cover & Temperature Analysis":
        feature_two()
    elif selection == "Satellite Image Super-Resolution":
        feature_three()


if __name__ == "__main__":
    main()
