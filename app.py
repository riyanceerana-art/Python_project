"""
OmniSight BI Analytics Studio - Core Backend Service
Architecture: Python / Flask / SQLite3 / Pandas
Thread-Safe Context-Managed Database Operations
"""

import os
import io
import json
import sqlite3
import contextlib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, Response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "analytics.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload limit


# ==========================================
# THREAD-SAFE SQLITE CONTEXT MANAGER
# ==========================================
@contextlib.contextmanager
def get_db():
    """
    Context manager providing safe SQLite connections with row dict factory
    and explicit transaction commits / rollbacks.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initializes tables for datasets, column profiles, and cached rows."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Primary datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                row_count INTEGER NOT NULL,
                col_count INTEGER NOT NULL
            )
        """)
        
        # Column metrics & data profiling table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS column_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                column_name TEXT NOT NULL,
                data_type TEXT NOT NULL,
                missing_values INTEGER NOT NULL,
                mean REAL,
                median REAL,
                min_val REAL,
                max_val REAL,
                unique_count INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            )
        """)
        
        # Raw tabular records store for paginated querying
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                row_idx INTEGER NOT NULL,
                data_json TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_dataset ON dataset_records(dataset_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_dataset ON column_metrics(dataset_id)")


def seed_default_dataset_if_empty():
    """Populates a realistic enterprise dataset if no datasets exist."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM datasets")
        if cursor.fetchone()['count'] > 0:
            return

    # Generate rich enterprise dataset
    np.random.seed(42)
    regions = ['North America', 'EMEA', 'APAC', 'LATAM']
    segments = ['Enterprise', 'Mid-Market', 'Strategic Accounts', 'Commercial']
    product_lines = ['Cloud Infrastructure', 'Cybersecurity Suite', 'Data Lake Platform', 'Workflow Automation', 'Cognitive AI Engine']
    priorities = ['Critical', 'High', 'Medium', 'Standard']
    channels = ['Direct Sales', 'Partner Network', 'Online Portal', 'Field Solutions']
    
    records = []
    n_rows = 240
    
    for i in range(n_rows):
        region = np.random.choice(regions, p=[0.40, 0.28, 0.22, 0.10])
        segment = np.random.choice(segments, p=[0.35, 0.30, 0.20, 0.15])
        prod = np.random.choice(product_lines)
        priority = np.random.choice(priorities, p=[0.15, 0.35, 0.35, 0.15])
        channel = np.random.choice(channels)
        
        base_revenue = float(np.random.gamma(shape=5.0, scale=3500.0))
        discount_rate = float(np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.25, 0.30, 0.25, 0.15, 0.05]))
        net_revenue = round(base_revenue * (1.0 - discount_rate), 2)
        margin_pct = float(np.random.uniform(0.18, 0.58))
        operating_profit = round(net_revenue * margin_pct, 2)
        unit_cost = round(float(np.random.uniform(120.0, 1850.0)), 2)
        customer_satisfaction = round(float(np.clip(np.random.normal(86, 7), 55, 100)), 1)
        cycle_days = int(np.random.randint(12, 95))
        active_seats = int(np.random.randint(25, 650))
        
        records.append({
            "Region": region,
            "Segment": segment,
            "Product_Line": prod,
            "Order_Priority": priority,
            "Sales_Channel": channel,
            "Net_Revenue": net_revenue,
            "Operating_Profit": operating_profit,
            "Discount_Rate": discount_rate,
            "Unit_Cost": unit_cost,
            "Active_Seats": active_seats,
            "Satisfaction_Score": customer_satisfaction,
            "Sales_Cycle_Days": cycle_days
        })
        
    df = pd.DataFrame(records)
    process_and_store_dataframe(df, "enterprise_commercial_benchmark_2026.csv")


def process_and_store_dataframe(df: pd.DataFrame, filename: str) -> int:
    """Analyzes schema, calculates summary metrics, and persists into SQLite."""
    # Sanitize column names: strip whitespace
    df.columns = [str(c).strip() for c in df.columns]
    
    row_count = int(len(df))
    col_count = int(len(df.columns))
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. Insert dataset record
        cursor.execute(
            "INSERT INTO datasets (filename, upload_time, row_count, col_count) VALUES (?, datetime('now'), ?, ?)",
            (filename, row_count, col_count)
        )
        dataset_id = cursor.lastrowid
        
        # 2. Analyze columns
        for col_name in df.columns:
            series = df[col_name]
            missing_vals = int(series.isna().sum())
            unique_count = int(series.nunique(dropna=True))
            
            # Determine if column is measure or dimension
            is_numeric = pd.api.types.is_numeric_dtype(series)
            
            # In cases where numeric column has only 2-3 discrete values, consider if measure
            data_type = "Measure" if is_numeric else "Dimension"
            
            mean_val = None
            median_val = None
            min_val = None
            max_val = None
            
            if is_numeric:
                clean_s = series.dropna()
                if len(clean_s) > 0:
                    mean_val = float(clean_s.mean())
                    median_val = float(clean_s.median())
                    min_val = float(clean_s.min())
                    max_val = float(clean_s.max())
            
            cursor.execute("""
                INSERT INTO column_metrics (
                    dataset_id, column_name, data_type, missing_values,
                    mean, median, min_val, max_val, unique_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id, col_name, data_type, missing_vals,
                mean_val, median_val, min_val, max_val, unique_count
            ))
            
        # 3. Store row data serialized as JSON for quick pagination
        records_to_insert = []
        # Replace NaN with None for valid JSON serialization
        clean_df = df.where(pd.notnull(df), None)
        dict_records = clean_df.to_dict(orient='records')
        
        for idx, row_dict in enumerate(dict_records):
            records_to_insert.append((
                dataset_id,
                idx,
                json.dumps(row_dict, default=str)
            ))
            
        cursor.executemany(
            "INSERT INTO dataset_records (dataset_id, row_idx, data_json) VALUES (?, ?, ?)",
            records_to_insert
        )
        
    return dataset_id


# ==========================================
# APPLICATION ROUTES
# ==========================================

@app.route("/")
def index():
    """Serves the primary Business Intelligence desktop workspace layout."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Service health and database diagnostic check."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as ds_count FROM datasets")
            ds_count = cursor.fetchone()['ds_count']
        return jsonify({
            "status": "healthy",
            "service": "OmniSight BI Analytics Studio Engine",
            "database": "analytics.db (SQLite3)",
            "datasets_available": ds_count
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    """Returns all registered datasets with metadata."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, upload_time, row_count, col_count FROM datasets ORDER BY id DESC")
        rows = [dict(r) for r in cursor.fetchall()]
    return jsonify(rows)


@app.route("/api/datasets/<int:dataset_id>", methods=["GET"])
def get_dataset(dataset_id):
    """Retrieves full profile, dimensions, measures, and statistics for a dataset."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        dataset = cursor.fetchone()
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404
        
        cursor.execute("SELECT * FROM column_metrics WHERE dataset_id = ?", (dataset_id,))
        columns = [dict(r) for r in cursor.fetchall()]
        
    # Split into dimensions and measures
    dimensions = [c for c in columns if c["data_type"] == "Dimension"]
    measures = [c for c in columns if c["data_type"] == "Measure"]
    
    # Calculate global missing value ratio
    total_cells = dataset["row_count"] * dataset["col_count"]
    total_missing = sum(c["missing_values"] for c in columns)
    missing_ratio = round((total_missing / total_cells * 100) if total_cells > 0 else 0, 2)
    
    return jsonify({
        "dataset": dict(dataset),
        "columns": columns,
        "dimensions": dimensions,
        "measures": measures,
        "summary": {
            "total_rows": dataset["row_count"],
            "total_columns": dataset["col_count"],
            "dimension_count": len(dimensions),
            "measure_count": len(measures),
            "total_missing_cells": total_missing,
            "missing_ratio_pct": missing_ratio
        }
    })


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Secure endpoint to parse, profile, and store an uploaded CSV file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename provided"}), 400
        
    if not file.filename.lower().endswith(('.csv', '.txt')):
        return jsonify({"error": "Only CSV and plain delimited text files are supported"}), 400
        
    filename = os.path.basename(file.filename)
    
    try:
        # Read the uploaded stream using Pandas
        try:
            df = pd.read_csv(file.stream, encoding='utf-8')
        except UnicodeDecodeError:
            file.stream.seek(0)
            df = pd.read_csv(file.stream, encoding='latin1')
            
        if df.empty:
            return jsonify({"error": "The uploaded CSV file is empty"}), 400
            
        # Process and store in SQLite
        dataset_id = process_and_store_dataframe(df, filename)
        
        return jsonify({
            "success": True,
            "message": f"Successfully loaded and profiled {filename}",
            "dataset_id": dataset_id
        }), 201
        
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 500


@app.route("/api/datasets/<int:dataset_id>/data", methods=["GET"])
def get_dataset_data(dataset_id):
    """
    Returns paginated raw table data with optional text search and column sorting.
    """
    page = max(1, int(request.args.get("page", 1)))
    page_size = max(5, min(200, int(request.args.get("page_size", 25))))
    offset = (page - 1) * page_size
    search_query = request.args.get("search", "").strip().lower()
    sort_col = request.args.get("sort_col", "")
    sort_dir = request.args.get("sort_dir", "asc").lower()

    with get_db() as conn:
        cursor = conn.cursor()
        
        # Verify dataset exists
        cursor.execute("SELECT id, row_count, col_count FROM datasets WHERE id = ?", (dataset_id,))
        ds = cursor.fetchone()
        if not ds:
            return jsonify({"error": "Dataset not found"}), 404
            
        # Fetch records
        cursor.execute(
            "SELECT data_json FROM dataset_records WHERE dataset_id = ? ORDER BY row_idx ASC",
            (dataset_id,)
        )
        all_raw = cursor.fetchall()

    # Parse JSON records
    records = [json.loads(r['data_json']) for r in all_raw]
    
    # Optional search filtering across all cell values
    if search_query:
        filtered = []
        for r in records:
            matched = any(search_query in str(v).lower() for v in r.values() if v is not None)
            if matched:
                filtered.append(r)
        records = filtered
        
    # Optional sorting
    if sort_col:
        reverse = (sort_dir == "desc")
        def sort_key(item):
            val = item.get(sort_col)
            if val is None:
                return (1, "")
            return (0, val)
        records.sort(key=sort_key, reverse=reverse)
        
    total_matched = len(records)
    paginated_slice = records[offset:offset + page_size]
    
    return jsonify({
        "page": page,
        "page_size": page_size,
        "total_records": total_matched,
        "total_pages": max(1, (total_matched + page_size - 1) // page_size),
        "data": paginated_slice
    })


@app.route("/api/datasets/<int:dataset_id>/export-csv", methods=["GET"])
def export_dataset_csv(dataset_id):
    """
    Exports the currently filtered and sorted dataset view as a downloadable CSV.
    """
    search_query = request.args.get("search", "").strip().lower()
    sort_col = request.args.get("sort_col", "")
    sort_dir = request.args.get("sort_dir", "asc").lower()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM datasets WHERE id = ?", (dataset_id,))
        ds = cursor.fetchone()
        if not ds:
            return jsonify({"error": "Dataset not found"}), 404
        orig_filename = ds['filename']
        
        cursor.execute(
            "SELECT data_json FROM dataset_records WHERE dataset_id = ? ORDER BY row_idx ASC",
            (dataset_id,)
        )
        all_raw = cursor.fetchall()

    records = [json.loads(r['data_json']) for r in all_raw]

    if search_query:
        filtered = []
        for r in records:
            matched = any(search_query in str(v).lower() for v in r.values() if v is not None)
            if matched:
                filtered.append(r)
        records = filtered

    if sort_col:
        reverse = (sort_dir == "desc")
        def sort_key(item):
            val = item.get(sort_col)
            if val is None:
                return (1, "")
            return (0, val)
        records.sort(key=sort_key, reverse=reverse)

    df = pd.DataFrame(records)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    base_name = os.path.splitext(orig_filename)[0]
    safe_base = "".join(c for c in base_name if c.isalnum() or c in ('_', '-')) or "dataset"
    download_filename = f"{safe_base}_filtered_export.csv"
    
    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=\"{download_filename}\"",
            "Content-Type": "text/csv; charset=utf-8"
        }
    )


@app.route("/api/datasets/<int:dataset_id>/aggregate", methods=["POST"])
def aggregate_data(dataset_id):
    """
    Performs dynamic aggregation for Chart.js visual canvas.
    Parameters in JSON:
      x_axis: Column name for categorizing / X dimension
      y_axis: Column name for metrics / Y measure
      agg: 'SUM' | 'AVG' | 'COUNT' | 'MIN' | 'MAX'
      sort: 'desc' | 'asc' | 'alpha'
      top_n: integer (e.g. 10, 25, 50) or None
    """
    payload = request.get_json(silent=True) or {}
    x_axis = payload.get("x_axis")
    y_axis = payload.get("y_axis")
    agg_func = payload.get("agg", "SUM").upper()
    sort_order = payload.get("sort", "desc").lower()
    top_n = payload.get("top_n")
    if top_n is not None:
        try:
            top_n = int(top_n)
        except (ValueError, TypeError):
            top_n = None

    if not x_axis or not y_axis:
        return jsonify({"error": "Both x_axis and y_axis must be selected"}), 400
        
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT data_json FROM dataset_records WHERE dataset_id = ?", (dataset_id,))
        rows = cursor.fetchall()
        if not rows:
            return jsonify({"error": "Dataset records not found"}), 404
            
    records = [json.loads(r['data_json']) for r in rows]
    df = pd.DataFrame(records)
    
    if x_axis not in df.columns or y_axis not in df.columns:
        return jsonify({"error": "Selected columns do not exist in dataset"}), 400

    # Ensure y_axis is numeric for mathematical aggregations
    df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
    
    # Fill NAs in grouping dimension
    df[x_axis] = df[x_axis].fillna("(Unspecified)")
    
    # Aggregation computation
    try:
        if agg_func == "SUM":
            grouped = df.groupby(x_axis, as_index=False)[y_axis].sum()
        elif agg_func in ("AVG", "MEAN"):
            grouped = df.groupby(x_axis, as_index=False)[y_axis].mean()
        elif agg_func == "COUNT":
            grouped = df.groupby(x_axis, as_index=False)[y_axis].count()
        elif agg_func == "MIN":
            grouped = df.groupby(x_axis, as_index=False)[y_axis].min()
        elif agg_func == "MAX":
            grouped = df.groupby(x_axis, as_index=False)[y_axis].max()
        else:
            grouped = df.groupby(x_axis, as_index=False)[y_axis].sum()
    except Exception as e:
        return jsonify({"error": f"Aggregation error: {str(e)}"}), 500

    # Rename aggregation output column
    grouped.columns = ['label', 'value']
    
    # Handle NaN values in aggregated metrics
    grouped['value'] = grouped['value'].fillna(0.0)
    
    # Apply Sorting
    if sort_order == "desc":
        grouped = grouped.sort_values(by='value', ascending=False)
    elif sort_order == "asc":
        grouped = grouped.sort_values(by='value', ascending=True)
    elif sort_order == "alpha":
        grouped = grouped.sort_values(by='label', ascending=True)
        
    # Top N limit
    if top_n and top_n > 0:
        grouped = grouped.head(top_n)
        
    labels = [str(v) for v in grouped['label'].tolist()]
    values = [round(float(v), 2) for v in grouped['value'].tolist()]
    
    # Compute summary benchmarks
    total_agg = round(float(sum(values)), 2)
    avg_agg = round(float(np.mean(values)) if values else 0.0, 2)
    max_agg = round(float(max(values)) if values else 0.0, 2)
    min_agg = round(float(min(values)) if values else 0.0, 2)
    
    return jsonify({
        "x_axis": x_axis,
        "y_axis": y_axis,
        "aggregation": agg_func,
        "labels": labels,
        "values": values,
        "count": len(labels),
        "summary": {
            "total": total_agg,
            "average": avg_agg,
            "max": max_agg,
            "min": min_agg
        }
    })


@app.route("/api/sample-dataset", methods=["POST"])
def reload_sample():
    """Generates and re-seeds the default enterprise benchmark dataset."""
    try:
        seed_default_dataset_if_empty()
        # Find latest dataset
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM datasets ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            latest_id = row['id'] if row else None
        return jsonify({"success": True, "dataset_id": latest_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/datasets/<int:dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id):
    """Deletes a dataset and all associated records and metrics."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    return jsonify({"success": True, "message": "Dataset removed successfully"})


# Initialize database upon module import
init_db()
seed_default_dataset_if_empty()


if __name__ == "__main__":
    # Dev server binds to 0.0.0.0 on port 3000 as required by environment
    app.run(host="0.0.0.0", port=3000, debug=True)
