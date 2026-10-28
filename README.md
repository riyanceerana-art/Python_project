# OmniSight BI 

OmniSight BI is a lightweight business intelligence workspace for exploring tabular data. Upload a CSV or delimited text file to profile its columns, review summary statistics, browse and filter records, and build interactive charts by selecting dimensions, measures, and aggregation methods. Charts can be exported as PNG files, and filtered records can be downloaded as CSV.

The application uses a Flask backend with Pandas and NumPy for data processing, and SQLite for persistent dataset storage. Its browser interface uses Chart.js. On first startup, it creates a sample enterprise dataset if the database is empty.

## Features

- Upload CSV and plain delimited text files (up to 32 MB).
- Automatically classify columns as dimensions or measures and calculate column statistics.
- Explore, search, sort, and paginate dataset records.
- Create aggregated visualizations and export them as PNG.
- Export filtered dataset records as CSV.
- Store datasets, profiles, and records in a local SQLite database.

## Run locally

Requirements: Python 3 and `pip`.

From the project directory, install the Python dependencies and start the server:

```powershell
python -m pip install Flask numpy pandas
python app.py
```

Open [http://localhost:3000](http://localhost:3000) in your browser. The SQLite database (`analytics.db`) is created in the project directory when the application starts.
