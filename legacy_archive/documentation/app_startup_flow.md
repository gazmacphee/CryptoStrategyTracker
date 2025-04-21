# Application Startup Flow Diagram

This document provides a visual representation of what happens when `streamlit run app.py` is executed, showing which processes are run and in which order.

## Main Application Flow

```mermaid
graph TD
    A[User starts application<br>'streamlit run app.py'] --> B[reset_and_start.py is called by workflow]
    B --> C{Database exists?}
    C -->|No| D[Create fresh database tables]
    C -->|Yes| E{Reset Database?}
    E -->|Yes| F[Drop all tables]
    F --> D
    E -->|No| G[Keep existing data]
    D --> H[Start background backfill process]
    G --> H
    H --> I[Start Streamlit UI]
    I --> J[Initialize app.py components]
    J --> K[Load and display UI components]
    K --> L[Background processes continue running]
```

## Database Initialization Process

```mermaid
graph TD
    A[Database initialization] --> B{Check DATABASE_URL environment variable}
    B --> C[Connect to PostgreSQL]
    C --> D[Create tables if they don't exist]
    D --> E[historical_data table]
    D --> F[technical_indicators table]
    D --> G[trades table]
    D --> H[portfolio table]
    D --> I[sentiment_data table]
```

## Background Data Processes

```mermaid
graph TD
    A[Background Data Processes] --> B[start_improved_backfill.py]
    B --> C[Process each symbol/interval pair]
    C --> D[Get file listings from Binance]
    D --> E{File listings available?}
    E -->|Yes| F[Download data month by month]
    E -->|No| G[Use fallback time range]
    G --> F
    F --> H{Monthly file available?}
    H -->|Yes| I[Process monthly file]
    H -->|No| J[Try daily files]
    I --> K[Save to database]
    J --> K
    K --> L[Calculate technical indicators]
    L --> M[Run gap detection]
    M --> N[Fill any data gaps]
    N --> O[Report unprocessed files]
    O --> P{Continuous mode?}
    P -->|Yes| Q[Sleep for 15 minutes]
    Q --> B
    P -->|No| R[End process]
```

## Unprocessed Files Tracking

```mermaid
graph TD
    A[Unprocessed Files Tracking] --> B{File download attempt}
    B -->|Success| C[Process file content]
    B -->|Failure| D[Log to unprocessed_files.log]
    C -->|Success| E[Save to database]
    C -->|Failure| D
    D --> F[Report in Data Gaps tab]
    F --> G{User action}
    G -->|View details| H[Display file details]
    G -->|Restart data collection| I[Kill existing processes]
    I --> J[Remove lock file]
    J --> K[Start new backfill process]
    G -->|Clear log| L[Reset unprocessed_files.log]
```

## HTTP Request Flow for Data

```mermaid
graph TD
    A[Data Request Flow] --> B[User selects symbol/interval]
    B --> C[get_data function called]
    C --> D{Data in cache?}
    D -->|Yes| E[Return cached data]
    D -->|No| F{Data in database?}
    F -->|Yes| G[Get from database]
    F -->|No| H[Try Binance API]
    H -->|Success| I[Save to database]
    H -->|Failure| J[Try CSV files]
    J -->|Success| I
    J -->|Failure| K[Report missing data]
    G --> L[Return data to UI]
    I --> L
```

## Gap Detection and Filling Process

```mermaid
graph TD
    A[Gap Detection and Filling] --> B[Analyze time series data]
    B --> C{Gaps detected?}
    C -->|Yes| D[Log gaps to gap_filler.log]
    C -->|No| E[Record complete status]
    D --> F[For each gap]
    F --> G[Attempt to fill from Binance API]
    G -->|Success| H[Update database]
    G -->|Failure| I[Try daily files]
    I -->|Success| H
    I -->|Failure| J[Mark as failed in log]
    H --> K[Update gap statistics]
    J --> K
    K --> L[Report in Data Gaps tab]
```

This diagram represents the flow of execution when the application starts. It covers the main application startup, database initialization, background data processes, unprocessed files tracking, data request flow, and gap detection/filling process.