FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file - use a more flexible approach
COPY pyproject.toml* requirements.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    elif [ -f pyproject.toml ]; then \
        pip install --no-cache-dir -e .; \
    else \
        echo "No requirements found, installing basic dependencies..."; \
        pip install streamlit pandas psycopg2-binary plotly numpy requests; \
    fi

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
# Wait for PostgreSQL if using external database\n\
if [ ! -z "$DATABASE_URL" ]; then\n\
  echo "Waiting for PostgreSQL..."\n\
  python -c "import time, psycopg2; time.sleep(5); print(\"PostgreSQL ready\")"\n\
fi\n\
\n\
# Run database initialization if requested\n\
if [ "$RESET_DB" = "true" ]; then\n\
  echo "Resetting database..."\n\
  python app/reset_database.py\n\
fi\n\
\n\
# Start application\n\
if [ "$BACKFILL_ON_START" = "true" ]; then\n\
  echo "Starting application with background data backfill..."\n\
  python app/backfill_database.py --continuous --interval=15 &\n\
else\n\
  echo "Starting application without background data backfill..."\n\
fi\n\
\n\
# Start Streamlit app\n\
streamlit run app/app.py --server.port 5000\n\
' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]