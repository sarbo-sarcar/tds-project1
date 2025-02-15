FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates npm

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /app

# Copy your FastAPI app code into the container
COPY main.py /app

# Run the FastAPI server using the uv command (assumed to be the correct command)
CMD ["uv", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
