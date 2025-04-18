# Local PostgreSQL Database Setup Guide

This guide explains how to set up and use your local PostgreSQL database with the Crypto Trading Analysis Platform.

## Prerequisites

- PostgreSQL installed on your local machine
- Username: postgres
- Password: 2212
- Default port: 5432

## Using the Database Setup Script

I've created a script to automatically check for and set up the database on your local PostgreSQL server.

1. Run the database setup script:

```bash
python check_local_db.py
```

This script will:
- Check if the `crypto` database exists, and create it if needed
- Create all required tables with proper indexes
- Grant necessary permissions to the database user
- Update your `.env` file with the correct database connection settings

## How the Local Database Connection Works

The application will now try to connect to your local PostgreSQL database in the following ways:

1. **Automatic Local Database Detection**:
   - When it detects a Docker-style database URL (db:5432/crypto), it will automatically try to connect to your local PostgreSQL server.
   - This happens both during application startup and in the database connection function.

2. **Connection Priority**:
   - Local PostgreSQL connection (localhost:5432) with your credentials
   - If local connection fails, it will fall back to the Neon cloud database

3. **Customization**: 
   - All your database settings are stored in the `.env` file
   - The local connection credentials are used by default

## Database Migration and Backfill

After setting up your local database, you may want to populate it with cryptocurrency data:

1. **Reset the database** (optional, if you want to start fresh):
   ```bash
   python reset_database.py
   ```

2. **Run the data backfill**:
   ```bash
   python backfill_database.py
   ```

3. **Alternatively**, you can set `BACKFILL_ON_START=true` in your `.env` file to automatically backfill when you start the application.

## Troubleshooting

If you encounter database connection issues:

1. **Check PostgreSQL Service**:
   - Make sure your PostgreSQL server is running
   - Verify you can connect with: `psql -U postgres -h localhost -p 5432`

2. **Check Database Existence**:
   - Connect to PostgreSQL and check if the database exists:
   ```sql
   \l
   ```

3. **Verify Credentials**:
   - Ensure the username (postgres) and password (2212) are correct
   - If you need to use different credentials, update them in the `.env` file

4. **Check Database Tables**:
   - Connect to the crypto database and verify tables exist:
   ```sql
   \c crypto
   \dt
   ```

5. **Manual Connection Using Python**:
   ```python
   import psycopg2
   conn = psycopg2.connect(
       host="localhost",
       port="5432",
       database="crypto", 
       user="postgres",
       password="2212"
   )
   ```

## Switching Between Local and Remote Databases

The application will automatically try to use your local PostgreSQL database when running locally. If you want to explicitly use a specific database:

1. **To use local database**:
   - Ensure the `.env` file contains the correct local PostgreSQL connection info

2. **To use the remote Neon database**:
   - Update the DATABASE_URL in `.env` to use the cloud-hosted PostgreSQL instance

With this setup, you should be able to seamlessly run the Crypto Trading Analysis Platform with your local PostgreSQL database.