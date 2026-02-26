# Jungle Dashboard

Simple Streamlit app that connects to Supabase to display monthly net sales per studio.

## Setup

1. Install deps: `pip install -r requirements.txt`
2. Create `.streamlit/secrets.toml` with:
   ```toml
   [general]
   SUPABASE_DB_URL="postgresql://..."
   ```
3. Run locally: `streamlit run app.py`
4. Studio daily dashboard (occupancy + FW dashboard tabs): `streamlit run app_studio_daily.py`

### Database expectations

- Legacy dashboard (`app.py`): Supabase table `public."Jun"` with columns `studio`, `date`, `netsales`, and `weekday`.
- Studio daily dashboard (`app_studio_daily.py`): Supabase table `public.studio_daily_metrics` with columns `studio`, `date`, and `net_sales` (weekday is derived from the date in the app).

## Loading New Studio Metrics

1. Drop the source CSV in `data/JFWTest.csv` (or pass a custom path with `--csv`).
2. Ensure `SUPABASE_DB_URL` is set to the Supabase/Postgres connection string.
3. Run `python scripts/import_jfw_metrics.py --dry-run` to verify parsing.
4. Load the data with `python scripts/import_jfw_metrics.py` (adds/updates rows in `public.studio_daily_metrics`).

### Scraping Jungle Studio Fitness (Marianatek) daily metrics

Use `scripts/scrape_jungle_site.py` to log into <https://junglestudiofitness.marianatek.com/> and push the table output straight into Supabase. The script only scaffolds the flow—you may need to tweak the CSS selector or login fields if Marianatek changes their markup.

```
export SUPABASE_DB_URL="postgresql://..."
export JUNGLE_DASHBOARD_EMAIL="sierra@exxircapital.com"
export JUNGLE_DASHBOARD_PASSWORD="ChangeMe123"

python scripts/scrape_jungle_site.py \
  --metrics-path /reports/studio_daily \
  --table-selector 'table#studio-daily-metrics' \
  --output latest_scrape.csv --dry-run
```

- Drop `--dry-run` to load into `public.studio_daily_metrics`.
- If the login form needs extra hidden fields (e.g., authenticity tokens), pass them via repeated `--extra-login-field key=value` flags.
- Schedule the script via cron/systemd to refresh data each morning.

### Loading `junalldb.csv`

Legacy exports that live in OneDrive can be written to a separate Supabase table with:

```
python scripts/import_junalldb.py --csv \
  "/Users/mnaz/Library/CloudStorage/OneDrive-SharedLibraries-ExxirCapital/Exxir Concepts - Documents/Jungle/Dashboards/Databases/junalldb.csv" \
  --table-name junalldb_metrics
```

- Use `--dry-run` to validate parsing without writing.
- The script normalizes the column names (`class_mat`, `cp_sales_mat`, etc.) and upserts into `public.junalldb_metrics` by default, creating the table if needed.

## Deploy

Push to `main` (repo linked to Streamlit Cloud) with `.streamlit/config.toml` + secrets configured in the Streamlit dashboard.
