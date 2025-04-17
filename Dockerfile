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
\n\
# Function to check PostgreSQL connection\n\
check_postgres() {\n\
  python -c "import psycopg2; conn = psycopg2.connect(\"$DATABASE_URL\"); conn.close()" 2>/dev/null\n\
  return $?\n\
}\n\
\n\
# Wait for PostgreSQL to be ready\n\
wait_for_postgres() {\n\
  echo "Waiting for PostgreSQL to be ready..."\n\
  for i in {1..30}; do\n\
    if check_postgres; then\n\
      echo "PostgreSQL is ready!"\n\
      return 0\n\
    fi\n\
    echo "Waiting for PostgreSQL... attempt $i/30"\n\
    sleep 2\n\
  done\n\
  echo "Timed out waiting for PostgreSQL"\n\
  return 1\n\
}\n\
\n\
# Wait for PostgreSQL\n\
wait_for_postgres\n\
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
  python backfill_database.py &\n\
else\n\
  echo "Starting application without background data backfill..."\n\
fi\n\
\n\
# Start Streamlit app\n\
streamlit run app.py --server.port 5000\n\
' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]