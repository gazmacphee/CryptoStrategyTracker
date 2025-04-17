# Docker Deployment Guide

This guide will help you deploy the Cryptocurrency Trading Analysis Platform using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- Git (to clone the repository)

## Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd cryptocurrency-trading-platform
```

2. Set up environment variables:

Copy the .env.example file to .env and modify as needed:

```bash
cp .env.example .env
```

Edit the .env file to include your API keys:

```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
OPENAI_API_KEY=your_openai_api_key
```

## Deployment

1. **Build and start the containers with the updated configuration files:**

```bash
# Use the updated Docker Compose file 
docker-compose -f docker-compose.updated.yml up -d --build
```

2. **Access the application:**

Once deployed, the application will be available at:
http://localhost:5001

## Troubleshooting

If you encounter issues with the Docker deployment, try the following steps:

### 1. Check container status

```bash
docker-compose -f docker-compose.updated.yml ps
```

### 2. View container logs

```bash
# View application logs
docker-compose -f docker-compose.updated.yml logs crypto-app

# View database logs
docker-compose -f docker-compose.updated.yml logs db
```

### 3. Database connection issues

If the application cannot connect to the database:

```bash
# Restart the containers
docker-compose -f docker-compose.updated.yml down
docker-compose -f docker-compose.updated.yml up -d
```

### 4. Reset the database

If you need to reset the database completely:

```bash
# Stop all containers
docker-compose -f docker-compose.updated.yml down

# Remove the volume
docker volume rm cryptocurrency-trading-platform_postgres_data

# Start containers again
docker-compose -f docker-compose.updated.yml up -d
```

### 5. Container shell access

To access the shell inside a container:

```bash
# Access application container
docker-compose -f docker-compose.updated.yml exec crypto-app bash

# Access database container
docker-compose -f docker-compose.updated.yml exec db bash
```

## Customization

You can modify the following environment variables in docker-compose.updated.yml:

- `RESET_DB`: Set to "true" to reset the database on startup
- `BACKFILL_ON_START`: Set to "true" to start backfilling data on startup
- Port mappings: Change the port from 5001 if needed

## Data Persistence

The PostgreSQL data is stored in a Docker volume named `postgres_data`. This ensures your data persists between container restarts.

## Monitoring and Maintenance

### Checking database status

```bash
docker-compose -f docker-compose.updated.yml exec db psql -U postgres -d crypto -c "SELECT COUNT(*) FROM historical_data;"
```

### Updating the application

To update to the latest version:

```bash
git pull
docker-compose -f docker-compose.updated.yml up -d --build
```