# ------------------ Install Dependencies ------------------
# Make sure these are installed in your environment
# !pip uninstall realesrgan basicsr -y
# !pip install git+https://github.com/xinntao/basicsr.git
# !pip install git+https://github.com/xinntao/Real-ESRGAN.git
# !pip install tifffile opencv-python-headless imagecodecs streamlit

# ------------------ Imports ------------------
import os
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import tifffile
import matplotlib.pyplot as plt
import json
from io import BytesIO
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

# ------------------ Feature One: Thermal Anomaly Detection ------------------
def feature_one():
    st.title("Neighborhood Thermal Anomaly Detection")
    st.markdown("Upload a thermal GeoTIFF to detect heat anomalies.")

    uploaded_file = st.file_uploader("Upload a .tif file for Anomaly Detection", type=["tif", "tiff"], key="f1_uploader")
    deltaT = st.slider("Set ΔT (Temperature threshold above mean):", 0.5, 10.0, 5.0, 0.5, key="f1_deltaT")

    if uploaded_file is not None:
        # Read raster from BytesIO
        with rasterio.open(BytesIO(uploaded_file.getvalue())) as src:
            temp_data = src.read(1)
            meta = src.meta

        mean_temp = np.mean(temp_data)
        threshold = mean_temp + deltaT
        mask = temp_data > threshold
        n_labels = np.sum(mask)
        st.success(f"Detected {n_labels} anomalous pixels (T > {threshold:.2f})")

        # Plot preview
        fig, ax = plt.subplots(figsize=(8,6))
        ax.imshow(temp_data, cmap="gray")
        overlay = np.ma.masked_where(mask == 0, temp_data)
        im = ax.imshow(overlay, cmap="inferno", alpha=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Temperature (relative)")
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

        st.download_button(
            "Download GeoJSON",
            geojson_bytes,
            file_name="anomalies.geojson",
            mime="application/geo+json",
            key="f1_download_geojson"
        )

# ------------------ Feature Two: Land Cover & Temperature Analysis ------------------
def feature_two():
    st.title("Land Cover & Temperature Analysis from Satellite Imagery")
    uploaded_file = st.file_uploader("Upload a .tif file for Land Cover Analysis", type=["tif"], key="f2_uploader")

    if uploaded_file is not None:
        temp_file_path = "/tmp/temp_input_f2.tif"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        try:
            with rasterio.open(temp_file_path) as src:
                bands = src.read()
            if bands.shape[0] < 10:
                st.error(f"Uploaded .tif has only {bands.shape[0]} bands. At least 10 required.")
                return

            # Select bands (0-indexed)
            red, green, nir, swir, thermal = bands[3], bands[2], bands[4], bands[5], bands[9]

            # Compute indices
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndwi = (green - nir) / (green + nir + 1e-6)
            ndbi = (swir - nir) / (swir + nir + 1e-6)

            # Land cover classification
            landcover = np.zeros_like(ndvi, dtype=np.uint8)
            landcover[ndvi > 0.4] = 1   # Vegetation
            landcover[ndwi > 0.3] = 2   # Water
            landcover[ndbi > 0.2] = 3   # Urban
            landcover[landcover == 0] = 4  # Bare soil / other

            # Color map for visualization
            colors = {1:[0,128,0], 2:[0,0,255], 3:[128,128,128], 4:[210,180,140]}
            rgb_image = np.zeros((*landcover.shape,3), dtype=np.uint8)
            for val, color in colors.items():
                rgb_image[landcover==val] = color

            st.subheader("Land Cover Map")
            st.image(rgb_image, caption="Land Cover Map", use_column_width=True)

            # Temperature stats
            classes = {1:"Vegetation",2:"Water",3:"Urban",4:"Bare Soil"}
            stats = []
            for val, name in classes.items():
                mask = landcover==val
                temps = thermal[mask]
                pixel_count = np.sum(mask)
                if temps.size>0:
                    stats.append({
                        "Land Cover": name,
                        "Pixel Count": int(pixel_count),
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
                csv_bytes = df.to_csv(index=False).encode()
                st.download_button("Download CSV", csv_bytes, "temperature_stats.csv", "text/csv")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

# ------------------ Feature Three: ESRGAN Super-Resolution ------------------
def feature_three():
    st.title("🌈 Satellite Image Super-Resolution with ESRGAN")
    uploaded_file = st.file_uploader("Upload a .tif file", type=["tif"], key="f3_uploader")

    if uploaded_file:
        input_tif = f"/tmp/{uploaded_file.name}"
        with open(input_tif, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info(f"📁 Processing {uploaded_file.name}")

        # Load TIFF and select RGB bands
        img = tifffile.imread(input_tif)
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] >= 3:
            if img.shape[2] >= 4:
                img_rgb = img[:, :, [3,2,1]]
            else:
                img_rgb = img[:, :, :3]
        else:
            st.error("Cannot detect valid RGB bands in TIFF!")
            st.stop()

        if img_rgb.dtype != np.uint8:
            img_rgb = (255*(img_rgb-np.min(img_rgb))/(np.max(img_rgb)-np.min(img_rgb))).astype(np.uint8)

        # ESRGAN model setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = "/tmp/RealESRGAN_x4plus.pth"
        if not os.path.exists(model_path):
            st.info("🔽 Downloading RealESRGAN weights...")
            os.system(f"wget -O {model_path} https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
            st.success("✅ Download complete!")

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
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
        st.success(f"✅ Using device: {device}")

        with st.spinner("⏳ Enhancing image..."):
            output, _ = upsampler.enhance(img_rgb, outscale=4)
        st.success("🎉 Enhancement complete!")

        output_tif = f"/tmp/enhanced_{uploaded_file.name}"
        tifffile.imwrite(output_tif, np.uint8(output))

        st.image([img_rgb, output], caption=["Original","Enhanced"], use_column_width=True)
        st.download_button("⬇️ Download Enhanced TIFF", data=open(output_tif,"rb"), file_name=f"enhanced_{uploaded_file.name}")

# ------------------ Main Navigation ------------------
def main():
    st.set_page_config(page_title="Geospatial Analysis & ESRGAN App", layout="wide")
    st.sidebar.title("Navigation")
    feature = st.sidebar.radio("", ["Thermal Anomaly Detection", "Land Cover & Temperature Analysis", "ESRGAN Super-Resolution"])

    if feature=="Thermal Anomaly Detection":
        feature_one()
    elif feature=="Land Cover & Temperature Analysis":
        feature_two()
    elif feature=="ESRGAN Super-Resolution":
        feature_three()

if __name__=="__main__":
    main()
