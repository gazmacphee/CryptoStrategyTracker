# Cryptocurrency Trading Analysis Platform - Deployment Guide

This document explains how to deploy the Cryptocurrency Trading Analysis Platform using Docker.

## Prerequisites

- Docker and Docker Compose installed on your system
- A PostgreSQL database (can be local or cloud-hosted)
- Binance API keys (optional but recommended for real data)
- OpenAI API key (for news digest and sentiment analysis features)

## Deployment Steps

### 1. Extract the Package

Extract the tar.gz file to a directory of your choice:

```bash
tar -xzvf crypto_trading_platform_docker.tar.gz -C /path/to/destination
cd /path/to/destination
```

### 2. Configure Environment Variables

Create a `.env` file based on the `.env.example` provided:

```bash
cp .env.example .env
```

Edit the `.env` file and update the following:

- Database connection details (PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE, DATABASE_URL)
- Binance API credentials (BINANCE_API_KEY, BINANCE_API_SECRET)
- OpenAI API key (OPENAI_API_KEY)
- Application settings (RESET_DB, BACKFILL_ON_START)

### 3. Build and Start the Docker Container

Build and start the Docker container using Docker Compose:

```bash
docker-compose up -d
```

This will:
- Build the Docker image if it doesn't exist
- Start the crypto application container
- Run the container in detached mode (-d flag)

### 4. Monitor the Application

Check if the application is running properly:

```bash
docker-compose logs -f
```

### 5. Access the Application

Once the containers are running, you can access your cryptocurrency trading platform at:

```
http://localhost:5000
```

## Docker Commands Reference

- To stop the containers:
  ```bash
  docker-compose down
  ```

- To view the logs:
  ```bash
  docker-compose logs
  ```

- To view logs specifically for the app (and follow them in real-time):
  ```bash
  docker-compose logs -f crypto-app
  ```

- To restart the containers:
  ```bash
  docker-compose restart
  ```

- To rebuild the image and restart containers (if you made changes to the code):
  ```bash
  docker-compose up -d --build
  ```

## Customization

You can modify the following files to customize your deployment:

- `docker-compose.yml`: Container configuration
- `.env`: Environment variables and API keys
- `start.sh`: Application startup script

## Troubleshooting

If you encounter any issues:

1. Check the container logs: `docker-compose logs`
2. Verify database connection in your `.env` file
3. Ensure your API keys are valid
4. Make sure the required ports (5000) are not being used by other applications

## Data Management

- Historical data is stored in your PostgreSQL database
- The application will automatically backfill data on startup if BACKFILL_ON_START=true
- You can reset the database by setting RESET_DB=true in your .env file (warning: this will delete all data)