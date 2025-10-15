import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from rasterio.features import shapes
from shapely.geometry import shape
from io import BytesIO

# --- Feature One: Thermal Anomaly Detection ---
def feature_one():
    st.title("Neighborhood Thermal Anomaly Detection")
    st.markdown("Upload a thermal GeoTIFF to detect heat anomalies.")

    # Helper functions for Feature One (moved inside the feature function or defined globally if preferred)
    def read_raster(file_path_or_buffer):
        """Reads a single band raster from a file path or buffer."""
        # rasterio.open can take a file path or a file-like object
        with rasterio.open(file_path_or_buffer) as src:
            return src.read(1), src.meta

    def detect_anomalies(temp_data, deltaT=5.0):
        """Detects thermal anomalies based on a temperature threshold."""
        mean_temp = np.mean(temp_data)
        threshold = mean_temp + deltaT
        mask = temp_data > threshold
        n_labels = np.sum(mask)
        return mask, n_labels, threshold

    def export_polygons(mask_data, meta_data):
        """Exports detected anomalies as GeoJSON polygons."""
        results = []
        # Ensure mask_data is a boolean or integer array for shapes function
        for geom, val in shapes(mask_data.astype(np.int16), transform=meta_data["transform"]):
            if val == 1: # Assuming '1' indicates an anomaly
                results.append({"geometry": geom, "properties": {"anomaly": 1}})
        return {"type": "FeatureCollection", "features": results}

    def generate_preview(temp_data, mask_data, threshold_val):
        """Generates a matplotlib preview of thermal anomalies."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Base image: full thermal data in grayscale
        ax.imshow(temp_data, cmap="gray")

        # Overlay: actual temperature values where anomalies exist
        overlay = np.ma.masked_where(mask_data == 0, temp_data)
        im = ax.imshow(overlay, cmap="inferno", alpha=0.6)

        # Colorbar for overlay
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Temperature (relative)")

        ax.set_title(f"Thermal Anomalies (T > {threshold_val:.2f})")
        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig) # Close the figure to free up memory

    # Streamlit UI for Feature One
    uploaded_file_f1 = st.file_uploader("Upload a .tif file for Anomaly Detection", type=["tif", "tiff"], key="f1_uploader")
    deltaT = st.slider("Set ΔT (Temperature threshold above mean):", 0.5, 10.0, 5.0, 0.5, key="f1_deltaT")

    if uploaded_file_f1 is not None:
        # rasterio can read directly from the BytesIO object
        temp, meta = read_raster(BytesIO(uploaded_file_f1.getvalue()))
        mask, n_labels, threshold = detect_anomalies(temp, deltaT)
        st.success(f"Detected {n_labels} anomalous pixels (T > {threshold:.2f})")

        st.subheader("Heatmap Preview")
        generate_preview(temp, mask, threshold)

        st.subheader("Export Results")
        geojson_data = export_polygons(mask, meta)
        geojson_bytes = BytesIO(json.dumps(geojson_data).encode())

        st.download_button(
            "Download GeoJSON",
            geojson_bytes,
            file_name="anomalies.geojson",
            mime="application/geo+json",
            key="f1_download_geojson"
        )

# --- Feature Two: Land Cover & Temperature Analysis ---
def feature_two():
    st.title("Land Cover & Temperature Analysis from Satellite Imagery")
    st.write("Upload a .tif file (e.g., Landsat 8 imagery) to analyze land cover types and associated temperature statistics.")

    uploaded_file_f2 = st.file_uploader("Choose a .tif file for Land Cover Analysis...", type=["tif"], key="f2_uploader")

    if uploaded_file_f2 is not None:
        # Save the uploaded file temporarily for rasterio to read by path
        # rasterio can also read from BytesIO, but your original code was set up for path
        temp_file_path = "temp_input_f2.tif"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file_f2.getbuffer())

        st.success(f"File '{uploaded_file_f2.name}' uploaded successfully!")

        st.subheader("Processing Imagery...")
        try:
            with rasterio.open(temp_file_path) as src:
                bands = src.read()

                # Ensure enough bands exist
                # Your original code used bands[4-1] etc., implying 1-based indexing in comments
                # Streamlit app needs 0-based indexing.
                # Assuming Landsat 8 typical band order where thermal is often band 10/11
                # Adjusted for 0-based: B1=0, B2=1, B3=2, B4=3, B5=4, B6=5, B7=6, B8=7, B9=8, B10=9, B11=10
                # If your TIFF only has fewer bands and Thermal is say band 6, adjust thermal_idx accordingly
                if bands.shape[0] < 10:
                    st.error(f"Error: The uploaded .tif file has only {bands.shape[0]} bands. At least 10 bands are typically required for this analysis (e.g., Landsat 8).")
                    return

                # Assign bands (0-indexed for Python)
                red = bands[3]      # Band 4
                nir = bands[4]      # Band 5
                green = bands[2]    # Band 3
                swir = bands[5]     # Band 6
                thermal = bands[9]  # Band 10

            st.write("Calculating Indices (NDVI, NDWI, NDBI)...")
            # Compute indices
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndwi = (green - nir) / (green + nir + 1e-6)
            ndbi = (swir - nir) / (swir + nir + 1e-6)

            st.write("Classifying Land Cover...")
            # Create land-cover map (dynamic thresholds)
            landcover = np.zeros_like(ndvi, dtype=np.uint8)
            landcover[ndvi > 0.4] = 1   # Vegetation
            landcover[ndwi > 0.3] = 2   # Water
            landcover[ndbi > 0.2] = 3   # Urban
            landcover[(landcover == 0)] = 4  # Bare Soil / other

            # Define colors for visualization
            colors = {
                1: [0, 128, 0],   # Vegetation (Dark Green)
                2: [0, 0, 255],   # Water (Blue)
                3: [128, 128, 128], # Urban (Gray)
                4: [210, 180, 140]  # Bare Soil (Tan)
            }

            # Create an RGB image for visualization
            normalized_landcover = np.zeros((*landcover.shape, 3), dtype=np.uint8)
            for class_val, color in colors.items():
                normalized_landcover[landcover == class_val] = color

            st.subheader("Land Cover Map Visualization")
            st.image(normalized_landcover, caption="Generated Land Cover Map", use_container_width=True)
            st.markdown(
                """
                **Legend:**
                - <span style="color:darkgreen;">■</span> Vegetation
                - <span style="color:blue;">■</span> Water
                - <span style="color:gray;">■</span> Urban
                - <span style="color:#D2B48C;">■</span> Bare Soil / Other
                """,
                unsafe_allow_html=True
            )

            st.write("Calculating Temperature Statistics per Land Cover Class...")
            # Compute statistics per class
            classes = {1: "Vegetation", 2: "Water", 3: "Urban", 4: "Bare Soil"}
            stats = []

            for class_val, class_name in classes.items():
                mask = (landcover == class_val)
                temps = thermal[mask]
                pixel_count = np.sum(mask)  # Number of pixels in this class
                if temps.size > 0:
                    stats.append({
                        "Land Cover": class_name,
                        "Pixel Count": int(pixel_count),
                        "Mean Temp": float(np.mean(temps)),
                        "Max Temp": float(np.max(temps)),
                        "Min Temp": float(np.min(temps)),
                        "Std Dev": float(np.std(temps)),
                        "Median": float(np.median(temps))
                    })

            df = pd.DataFrame(stats)

            st.subheader("Temperature Statistics by Land Cover Type")
            if not df.empty:
                st.dataframe(df)

                # Provide CSV download
                csv_output = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Temperature Statistics as CSV",
                    data=csv_output,
                    file_name="landcover_temperature_statistics.csv",
                    mime="text/csv",
                    key="f2_download_csv"
                )
            else:
                st.info("No land cover classes detected or no statistics could be computed.")

        except rasterio.errors.RasterioIOError:
            st.error("Error: Could not open the .tif file. It might be corrupted or not a valid GeoTIFF.")
        except IndexError as e:
            st.error(f"Error: Band index out of range. Please ensure your .tif file has enough bands for the selected analysis. Details: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

# --- Main App Logic (Navigation) ---
def main():
    st.set_page_config(page_title="Geospatial Analysis App", layout="wide") # Set config once here
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio(
        "",
        ["Thermal Anomaly Detection", "Land Cover & Temperature Analysis"]
    )

    if selection == "Thermal Anomaly Detection":
        feature_one()
    elif selection == "Land Cover & Temperature Analysis":
        feature_two()

if __name__ == "__main__":
    main()