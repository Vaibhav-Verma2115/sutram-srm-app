# HF Spaces no longer offers a Streamlit SDK, so we run it under the Docker SDK.
# python:3.12-slim matches the project's requires-python >=3.12.
FROM python:3.12-slim

# rasterio ships GDAL inside its wheel, but that GDAL still links against the
# system libexpat, which python:*-slim does not include -- without this, the
# app builds cleanly and then dies at `import rasterio`. libgomp is the OpenMP
# runtime torch and scikit-image expect.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000; writing as root would leave the app
# unable to write its output GeoTIFFs.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Fail the build rather than the first visitor if a native library is missing.
RUN python -c "import rasterio, torch, skimage, sen2sr, mlstac; print('imports OK', rasterio.__version__, torch.__version__)"

EXPOSE 7860

# XSRF protection is disabled because the Space is served inside an iframe,
# which otherwise breaks the file uploader.
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableXsrfProtection=false", \
     "--server.maxUploadSize=400", \
     "--browser.gatherUsageStats=false"]
