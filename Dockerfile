FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY . .

EXPOSE 8080

CMD ["scripts/run_simulation.sh"]
