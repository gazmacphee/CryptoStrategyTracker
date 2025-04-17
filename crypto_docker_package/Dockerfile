FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy application files
COPY . .

# Create the .streamlit directory if it doesn't exist
RUN mkdir -p /app/.streamlit

# Ensure config.toml exists in .streamlit directory
RUN if [ ! -f /app/.streamlit/config.toml ]; then \
    echo '[server]' > /app/.streamlit/config.toml && \
    echo 'headless = true' >> /app/.streamlit/config.toml && \
    echo 'address = "0.0.0.0"' >> /app/.streamlit/config.toml && \
    echo 'port = 5000' >> /app/.streamlit/config.toml; \
    fi

# Expose the port Streamlit runs on
EXPOSE 5000

# Create a volume for persistent database data
VOLUME ["/app/data"]

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Wait for PostgreSQL\n\
echo "Waiting for PostgreSQL connection..."\n\
max_attempts=30\n\
attempt=0\n\
db_ready=false\n\
\n\
while [ $attempt -lt $max_attempts ] && [ "$db_ready" = false ]; do\n\
  attempt=$((attempt+1))\n\
  echo "Attempt $attempt/$max_attempts to connect to database..."\n\
  if python -c "import psycopg2; conn = psycopg2.connect(host=\"${PGHOST:-db}\", port=\"${PGPORT:-5432}\", user=\"${PGUSER:-postgres}\", password=\"${PGPASSWORD:-postgres}\", dbname=\"${PGDATABASE:-crypto}\"); conn.close()" 2>/dev/null; then\n\
    echo "Database connection successful!"\n\
    db_ready=true\n\
  else\n\
    echo "Database not ready yet..."\n\
    sleep $((attempt < 5 ? 2 : 5))\n\
  fi\n\
done\n\
\n\
if [ "$db_ready" = false ]; then\n\
  echo "ERROR: Could not connect to database after $max_attempts attempts! The application will still attempt to start."\n\
fi\n\
\n\
# Run database initialization if requested\n\
if [ "$RESET_DB" = "true" ]; then\n\
  echo "Resetting database..."\n\
  python reset_database.py\n\
fi\n\
\n\
# Start application\n\
if [ "$BACKFILL_ON_START" = "true" ]; then\n\
  echo "Starting application with background data backfill..."\n\
  python backfill_database.py --continuous --interval=15 &\n\
else\n\
  echo "Starting application without background data backfill..."\n\
fi\n\
\n\
# Start Streamlit app\n\
streamlit run app.py --server.port 5000\n\
' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]