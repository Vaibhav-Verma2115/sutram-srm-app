# HF Spaces no longer offers a Streamlit SDK, so we run it under the Docker SDK.
# python:3.12-slim matches the project's requires-python >=3.12; rasterio and
# torch both ship manylinux wheels, so no GDAL or build toolchain is needed.
FROM python:3.12-slim

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
