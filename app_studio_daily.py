from calendar import monthrange
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import create_engine, text


def ensure_date(value, fallback):
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return fallback


def normalize_range(selection, fallback: Tuple[date, date]) -> Tuple[date, date]:
    start, end = fallback
    seq: Sequence[object] = ()

    if isinstance(selection, (list, tuple)):
        seq = selection
    elif selection is not None:
        return (ensure_date(selection, start), ensure_date(selection, end))

    if len(seq) >= 2:
        start = ensure_date(seq[0], start)
        end = ensure_date(seq[1], end)
    elif len(seq) == 1:
        start = end = ensure_date(seq[0], start)

    if start > end:
        start, end = end, start

    return (start, end)


def clamp_date(value: date, lower: date, upper: date) -> date:
    return max(min(value, upper), lower)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    if "date" in view.columns:
        view["date"] = view["date"].dt.strftime("%m-%d-%y")
    if "weekday" in view.columns:
        view["weekday"] = view["weekday"].fillna("").str[:3]
    if "netsales" in view.columns:
        view["netsales"] = view["netsales"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    return view


def safe_sum(df: pd.DataFrame, column: str) -> Optional[float]:
    if column not in df.columns or df.empty:
        return None
    total = df[column].sum()
    return float(total) if pd.notna(total) else None


def closest_timestamp(index: pd.DatetimeIndex, candidate: pd.Timestamp) -> pd.Timestamp:
    if len(index) == 0:
        return candidate
    first_ts = pd.Timestamp(index[0])  # type: ignore[index]
    last_ts = pd.Timestamp(index[-1])  # type: ignore[index]
    if candidate <= first_ts:
        return first_ts
    if candidate >= last_ts:
        return last_ts
    pos = index.get_indexer([candidate], method="nearest")[0]
    return pd.Timestamp(index[pos])  # type: ignore[index]


def build_chart_data(df: pd.DataFrame, series_label: str, range_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            "netsales": pd.Series(dtype=float),
            "series": pd.Series(dtype="string"),
            "weekday": pd.Series(dtype="string"),
            "range_label": pd.Series(dtype="string"),
        })
    grouped = (
        df.groupby("date")["netsales"].sum().reset_index()
    )
    grouped["series"] = series_label
    grouped["weekday"] = grouped["date"].dt.strftime("%a")
    grouped["range_label"] = range_label
    return grouped


def align_date_to_weekday(target_date: date, weekday_index_map: dict[int, pd.DatetimeIndex], history_index: pd.DatetimeIndex) -> date:
    if len(history_index) == 0:
        return target_date
    target_ts = cast(pd.Timestamp, pd.Timestamp(target_date))
    weekday = int(target_ts.dayofweek)
    candidates = weekday_index_map.get(weekday)
    if candidates is not None and len(candidates) > 0:
        earlier = candidates[candidates <= target_ts]
        if len(earlier) > 0:
            return pd.Timestamp(earlier[-1]).date()  # type: ignore[index]
        return pd.Timestamp(candidates[0]).date()  # type: ignore[index]
    earlier_any = history_index[history_index <= target_ts]
    if len(earlier_any) > 0:
        return pd.Timestamp(earlier_any[-1]).date()  # type: ignore[index]
    return pd.Timestamp(history_index[0]).date()  # type: ignore[index]


def compute_current_dates(horizon: str, min_date: date, max_date: date) -> Tuple[date, date]:
    if horizon == "Daily":
        start = end = max_date
    elif horizon == "Weekly":
        end = max_date
        start = max_date - timedelta(days=6)
    elif horizon == "Monthly Estimate":
        start = max_date.replace(day=1)
        end_day = monthrange(max_date.year, max_date.month)[1]
        end = max_date.replace(day=end_day)
    else:
        start = max_date.replace(day=1)
        end = max_date
    if start < min_date:
        start = min_date
    if end < start:
        end = start
    return start, end


def compute_comparison_dates(
    horizon: str,
    current_start: date,
    current_end: date,
    min_date: date,
    max_date: date,
    weekday_index_map: dict[int, pd.DatetimeIndex],
    history_index: pd.DatetimeIndex,
    oldest_month_start: date,
) -> Tuple[date, date]:
    period_length = current_end - current_start

    if horizon in ("Daily", "Weekly"):
        shift = timedelta(weeks=52)
        candidate_start = current_start - shift
        candidate_end = current_end - shift
        comp_start = align_date_to_weekday(candidate_start, weekday_index_map, history_index)
        comp_end = align_date_to_weekday(candidate_end, weekday_index_map, history_index)
    else:
        candidate_start = current_start - timedelta(days=365)
        comp_start = candidate_start
        comp_end = candidate_start + period_length

    if len(history_index) == 0 or comp_start < min_date:
        comp_start = oldest_month_start
        comp_end = min(comp_start + period_length, max_date)

    if comp_end < comp_start:
        comp_end = comp_start

    return comp_start, comp_end


st.set_page_config(layout="wide")
header_html = (
    "<style>"
    ".primary-header {font-size: 2rem; font-weight: 700; margin: 0;}"
    ".header-divider {border-bottom: 1px solid #1e2438; margin: 0.15rem 0 0.35rem;}"
    "</style>"
    "<div class=\"primary-header\">Jungle Studio Dashboard</div>"
    "<div class=\"header-divider\"></div>"
)
st.markdown(header_html, unsafe_allow_html=True)

STUDIO_PICKER_CSS = """
<style>
div[data-baseweb="select"] > div {
    background-color: #0c0f1f;
    border: 1px solid #2c314f;
    border-radius: 12px;
    min-height: auto;
    padding: 0.25rem 0.3rem;
}
div[data-baseweb="tag"] {
    background-color: #5c5feb;
    border-radius: 10px;
    color: #fff;
    font-weight: 600;
}
div[data-baseweb="tag"] span {
    color: #fff !important;
}
div[data-baseweb="select"] svg {
    color: #9ea4da;
}
.selector-card {
    background: #0b1124;
    border: 1px solid #2a3154;
    border-radius: 18px;
    padding: 0.5rem 0.9rem 0.7rem;
    margin: 0;
}
div[data-baseweb="radio"] {
    padding: 0;
    margin: 0;
}
div[data-baseweb="radio"] > div {
    display: flex;
    gap: 0.3rem;
    flex-wrap: wrap;
}
</style>
"""
st.markdown(STUDIO_PICKER_CSS, unsafe_allow_html=True)

engine = create_engine(
    st.secrets["SUPABASE_DB_URL"],
    connect_args={"sslmode": "require"}
)


@st.cache_data(ttl=60)
def load_data():
    query = text("""
        SELECT
            "studio",
            "date",
            "net_sales",
            "total_visits",
            "capacity",
            "classes",
            "first_time"
        FROM public.studio_daily_metrics
        WHERE "net_sales" IS NOT NULL
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"net_sales": "netsales"})
    df["weekday"] = df["date"].dt.strftime("%A")
    for column in ("total_visits", "capacity", "classes", "first_time"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


df = load_data()

# --- Studio Selector ---
studios = sorted(df["studio"].unique())
default_selection = studios[:1]
st.markdown('<div class="selector-title">Studios</div>', unsafe_allow_html=True)
selected_studios = st.multiselect(
    "Studios",
    studios,
    default=default_selection,
    label_visibility="collapsed",
)

if not selected_studios:
    st.info("Select at least one studio to continue.")
    st.stop()

selection_label = ", ".join(selected_studios)

studio_df = df[df["studio"].isin(selected_studios)].copy()

if studio_df.empty:
    st.warning("No data available for the selected studios.")
    st.stop()

studio_df = studio_df.sort_values("date")  # type: ignore[arg-type]

min_date = studio_df["date"].min().date()
max_date = studio_df["date"].max().date()
oldest_month_start = min_date.replace(day=1)

history_series = studio_df.groupby("date")["netsales"].sum().sort_index()
history_index: pd.DatetimeIndex = pd.DatetimeIndex(history_series.index)
weekday_index_map = {}
if len(history_index) > 0:
    history_weekday_series = pd.Series(history_index, index=history_index).dt.weekday
    for weekday in range(7):
        mask = history_weekday_series == weekday
        if mask.any():
            weekday_index_map[weekday] = history_weekday_series.index[mask]

horizon = st.radio(
    "Select horizon",
    ["Daily", "Weekly", "Monthly", "Monthly Estimate"],
    horizontal=True,
    label_visibility="collapsed",
)

start_date, end_date = compute_current_dates(horizon, min_date, max_date)
comp_start_date, comp_end_date = compute_comparison_dates(
    horizon,
    start_date,
    end_date,
    min_date,
    max_date,
    weekday_index_map,
    history_index,
    oldest_month_start,
)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)
comp_start_ts = pd.Timestamp(comp_start_date)
comp_end_ts = pd.Timestamp(comp_end_date)

actual_end_ts = cast(pd.Timestamp, end_ts)
if horizon == "Monthly Estimate":
    actual_end_ts = cast(pd.Timestamp, pd.Timestamp(max_date))

filtered_selection = studio_df[
    (studio_df["date"] >= start_ts) &
    (studio_df["date"] <= actual_end_ts)
]
filtered_df = pd.DataFrame(filtered_selection).copy()

if filtered_df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()

range_sales = filtered_df["netsales"].sum()

comparison_selection_df = studio_df[
    (studio_df["date"] >= comp_start_ts) &
    (studio_df["date"] <= comp_end_ts)
]
comparison_df: pd.DataFrame = pd.DataFrame(comparison_selection_df).copy()
comparison_sales = comparison_df["netsales"].sum() if not comparison_df.empty else 0.0
yoy_multiplier = 1.0
comparison_delta_pct = None
if comparison_df.empty or comparison_sales == 0:
    comparison_delta_pct = None
else:
    diff_pct = ((range_sales - comparison_sales) / comparison_sales) * 100
    comparison_delta_pct = f"{diff_pct:+.1f}%"
    yoy_multiplier = range_sales / comparison_sales if comparison_sales else 1.0

forecast_extra_total = 0.0
estimated_rows: List[Dict[str, Any]] = []
range_sales_display = range_sales

forecast_values: List[float] = []

if horizon == "Monthly Estimate":
    month_start_ts = pd.Timestamp(start_date)
    month_end_ts = pd.Timestamp(end_date)

    actual_range = studio_df[
        (studio_df["date"] >= month_start_ts) &
        (studio_df["date"] <= actual_end_ts)
    ]
    actual_total = actual_range["netsales"].sum()

    remaining_dates = pd.date_range(start=actual_end_ts + timedelta(days=1), end=month_end_ts)
    if not remaining_dates.empty:
        forecast_rows = []
        for ts in remaining_dates:
            target_ts = cast(pd.Timestamp, pd.Timestamp(ts))
            candidate = cast(pd.Timestamp, target_ts - pd.DateOffset(years=1))
            weekday = int(target_ts.dayofweek)
            weekday_history = weekday_index_map.get(weekday)

            if weekday_history is not None and len(weekday_history) > 0:
                source_timestamp = closest_timestamp(weekday_history, candidate)
            else:
                source_timestamp = closest_timestamp(history_index, candidate)

            base_value = float(history_series.loc[source_timestamp])
            projected = base_value * yoy_multiplier
            forecast_rows.append(projected)

        forecast_extra_total = float(sum(forecast_rows))
        estimated_rows = [
            {
                "date": date,
                "netsales": value,
                "estimated": True,
            }
            for date, value in zip(remaining_dates, forecast_rows)
        ]
        range_sales_display = actual_total + forecast_extra_total
        range_sales = actual_total

    if (not comparison_df.empty) and comparison_sales:
        diff_pct = ((range_sales_display - comparison_sales) / comparison_sales) * 100
        comparison_delta_pct = f"{diff_pct:+.1f}%"

comparison_period_label = f"{comp_start_date:%b %d, %Y} – {comp_end_date:%b %d, %Y}"


def sum_sales_between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    window = df[(df["date"] >= start) & (df["date"] <= end)]
    if window.empty:
        return 0.0
    total = window["netsales"].sum()
    return float(total) if pd.notna(total) else 0.0

month_reference_ts = cast(pd.Timestamp, actual_end_ts)

month_start_ts = cast(pd.Timestamp, pd.Timestamp(month_reference_ts).replace(day=1))
month_to_date_df = studio_df[(studio_df["date"] >= month_start_ts) & (studio_df["date"] <= month_reference_ts)]
month_sales_to_date = float(month_to_date_df["netsales"].sum()) if not month_to_date_df.empty else 0.0
month_sales_estimate = range_sales_display if horizon == "Monthly Estimate" else month_sales_to_date
month_label_td = (
    f"Sales MTD: {month_start_ts:%b %d} – {month_reference_ts:%b %d}"
    if month_sales_to_date
    else "Sales MTD: No data"
)
month_label_est = f"Sales Est: {start_date:%b %d} – {end_date:%b %d}" if horizon == "Monthly Estimate" else f"Sales Est: {month_start_ts:%b %d} – {month_reference_ts:%b %d}"

month_td_span = month_reference_ts - month_start_ts
month_td_comp_start = cast(pd.Timestamp, comp_start_ts)
month_td_comp_end = cast(pd.Timestamp, min(comp_start_ts + month_td_span, comp_end_ts))
month_sales_to_date_comp = sum_sales_between(comparison_df, month_td_comp_start, month_td_comp_end)
month_sales_estimate_comp = sum_sales_between(comparison_df, cast(pd.Timestamp, comp_start_ts), cast(pd.Timestamp, comp_end_ts))
month_label_td_comp = f"{month_td_comp_start:%b %d} – {month_td_comp_end:%b %d}"
month_label_est_comp = f"{comp_start_date:%b %d} – {comp_end_date:%b %d}"

st.markdown(
    (
        "<div style='margin-top:-0.05rem;margin-bottom:0;color:#aeb3d1;font-size:0.9rem;'>"
        f"<span style='color:#f5c746;'>Current: {start_date:%b %d, %Y} – {end_date:%b %d, %Y}</span> | "
        f"Comparison: {comp_start_date:%b %d, %Y} – {comp_end_date:%b %d, %Y}"
        "</div>"
    ),
    unsafe_allow_html=True,
)

# --- Layout ---
tab_snap, tab_sales, tab_chart, tab_visits, tab_forecast, tab_occupancy, tab_fw_dashboard, tab_sales_money = st.tabs(["Snap", "Current", "Chart", "Visits", "Forecast", "Occupancy", "Summary", "Sales $"])

with tab_sales:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric(
            label="Net sales (selected range)",
            value=f"${range_sales_display:,.0f}"
        )

    with col2:
        st.metric(
            label="Comparison net sales",
            value=f"${comparison_sales:,.0f}",
            delta=comparison_delta_pct
        )

    st.subheader("Selected Range Details")
    current_table_df = filtered_df.sort_values("date", ascending=False)
    if estimated_rows:
        add_df = pd.DataFrame(estimated_rows)
        current_table_df = pd.concat([current_table_df, add_df], ignore_index=True)
    st.dataframe(format_table(current_table_df))
    if horizon == "Monthly Estimate" and forecast_extra_total > 0:
        st.markdown(
            f"<div style='color:#f5b342;font-weight:600;margin-top:0.3rem;'>Monthly estimate projection adds ${forecast_extra_total:,.0f} beyond actual MTD.</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Comparison Range Details")
    if comparison_df.empty:
        st.info("No data available for the comparison range.")
    else:
        comparison_view = comparison_df.sort_values("date", ascending=False)
        st.dataframe(format_table(comparison_view))

with tab_chart:
    selected_label = f"{start_date:%m-%d-%y} – {end_date:%m-%d-%y}"
    comparison_label = f"{comp_start_date:%m-%d-%y} – {comp_end_date:%m-%d-%y}"

    selected_series_label = f"Current {selected_label}"
    comparison_series_label = f"Comparison {comparison_label}"

    selected_chart_df = build_chart_data(filtered_df, selected_series_label, selected_label)
    comparison_chart_df = build_chart_data(comparison_df, comparison_series_label, comparison_label)
    if horizon in ("Daily", "Weekly"):
        selected_chart_df["x_axis"] = selected_chart_df["date"].dt.strftime("%a")
        comparison_chart_df["x_axis"] = comparison_chart_df["date"].dt.strftime("%a")
        x_title = "Weekday"
    else:
        selected_chart_df["x_axis"] = selected_chart_df["date"].dt.strftime("%m")
        comparison_chart_df["x_axis"] = comparison_chart_df["date"].dt.strftime("%m")
        x_title = "Month"
    chart_frames = [selected_chart_df, comparison_chart_df]
    legend_order = [selected_series_label, comparison_series_label]

    chart_df = pd.concat(chart_frames, ignore_index=True)

    if chart_df.empty:
        st.info("Not enough data to render the chart.")
    else:
        weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "x_axis:N",
                    title=x_title,
                ),
                y=alt.Y("netsales:Q", title="Net sales"),
                color=alt.Color(
                    "series:N",
                    title="",
                    scale=alt.Scale(domain=legend_order),
                    legend=alt.Legend(
                        orient="bottom",
                        direction="horizontal",
                        labelLimit=0,
                        symbolType="circle"
                    )
                ),
                tooltip=[
                    alt.Tooltip("series:N", title="Range"),
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("weekday:N", title="Weekday"),
                    alt.Tooltip("netsales:Q", title="Net sales", format="$.0f"),
                    alt.Tooltip("range_label:N", title="Date range")
                ]
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

with tab_visits:
    selected_visits = filtered_df.copy()
    comparison_visits = comparison_df.copy()

    for frame in (selected_visits, comparison_visits):
        for col in ("mt_visits", "cp_visits"):
            if col not in frame.columns:
                if "total_visits" in frame.columns:
                    frame[col] = frame["total_visits"] / 2
                else:
                    frame[col] = 0
        frame["visits"] = frame[["mt_visits", "cp_visits"]].fillna(0).sum(axis=1)

    total_visits = selected_visits["visits"].sum()
    comparison_total_visits = comparison_visits["visits"].sum() if not comparison_visits.empty else 0.0
    visits_delta = None
    if comparison_total_visits:
        visits_delta = ((total_visits - comparison_total_visits) / comparison_total_visits) * 100

    visit_cols = st.columns(2)
    visit_cols[0].metric("Visits (selected)", f"{total_visits:,.0f}", f"{visits_delta:+.1f}%" if visits_delta is not None else None)
    visit_cols[1].metric("Visits (comparison)", f"{comparison_total_visits:,.0f}")

    visit_chart_df = pd.concat([
        selected_visits.assign(series="Selected"),
        comparison_visits.assign(series="Comparison"),
    ])

    if visit_chart_df.empty:
        st.info("No visit data to chart.")
    else:
        visit_chart = (
            alt.Chart(visit_chart_df)
            .mark_line(point=True)
            .encode(
                x="date:T",
                y="visits:Q",
                color="series:N",
                tooltip=["series", "date", "visits"]
            )
        )
        st.altair_chart(visit_chart, use_container_width=True)

    st.subheader("Visits (Selected Range)")
    st.dataframe(format_table(selected_visits))

    st.subheader("Visits (Comparison Range)")
    if comparison_visits.empty:
        st.info("No comparison data available for visits.")
    else:
        st.dataframe(format_table(comparison_visits))

with tab_snap:
    st.markdown(
        """
        <style>
        .snap-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.8rem;margin-top:0.5rem;}
        .snap-card {background:#10121a;border:1px solid #2c2f38;border-radius:12px;padding:0.7rem 0.9rem;position:relative;}
        .snap-card[data-tab-target] {cursor:pointer;}
        .snap-card[data-tab-target]::after {content:""; position:absolute; top:6px; right:6px; width:20px; height:20px; background-image:url('data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 24 24%22 fill=%22none%22%3E%3Cpath fill=%22%23f5c746%22 fill-rule=%22evenodd%22 d=%22M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25Zm4.28 10.28a.75.75 0 0 0 0-1.06l-3-3a.75.75 0 1 0-1.06 1.06l1.72 1.72H8.25a.75.75 0 0 0 0 1.5h5.69l-1.72 1.72a.75.75 0 1 0 1.06 1.06l3-3Z%22 clip-rule=%22evenodd%22/%3E%3C/svg%3E'); background-size:100%; background-repeat:no-repeat; pointer-events:none;}
        .snap-card[data-tooltip]::before {content:attr(data-tooltip); position:absolute; top:-10px; left:50%; transform:translate(-50%, 6px); opacity:0; pointer-events:none; transition:opacity 0.12s ease, transform 0.12s ease; background:#0f1324; color:#f5c746; padding:0.35rem 0.6rem; border-radius:8px; font-size:0.75rem; border:1px solid #f5c746; white-space:nowrap; box-shadow:0 4px 18px rgba(0,0,0,0.35);}
        .snap-card[data-tooltip]:hover::before,
        .snap-card[data-tooltip]:focus-visible::before {opacity:1; transform:translate(-50%, -6px);}
        .snap-label {font-size:0.9rem;color:#fdfdfd;font-weight:700;letter-spacing:0.05em;}
        .snap-main {display:flex;justify-content:space-between;align-items:center;margin-top:0.15rem;}
        .snap-value {font-size:1.4rem;font-weight:600;color:#f5c746;}
        .snap-delta {font-size:0.9rem;font-weight:600;}
        .snap-sub {font-size:0.8rem;color:#a8aec6;margin-top:0.25rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    def fmt_value(value: Optional[float], kind: str) -> str:
        if value is None:
            return "—"
        if kind == "currency":
            return f"${value:,.0f}"
        if kind == "percent":
            return f"{value * 100:.0f}%"
        if kind == "number2":
            return f"{value:,.2f}"
        return f"{value:,.0f}"

    def yoy_delta(current: Optional[float], comparison: Optional[float]) -> Optional[float]:
        if current is None or comparison in (None, 0):
            return None
        return (current / comparison) - 1

    def occ_ratio(df: pd.DataFrame) -> Optional[float]:
        if df.empty or "total_visits" not in df.columns:
            return None
        if {"capacity", "classes"}.issubset(df.columns):
            denom = (df["capacity"] * df["classes"]).replace({0: pd.NA}).sum()
        else:
            denom = 0
        if denom in (None, 0):
            return None
        numer = df["total_visits"].fillna(0).sum()
        return numer / denom if denom else None

    def ratio_from_columns(df: pd.DataFrame, numer: str, denom: str) -> Optional[float]:
        num = safe_sum(df, numer)
        den = safe_sum(df, denom)
        if num is None or den in (None, 0):
            return None
        return num / den

    selected_visits_total = safe_sum(filtered_df, "total_visits") or 0.0
    comparison_visits_total = safe_sum(comparison_df, "total_visits") or 0.0
    selected_occ = occ_ratio(filtered_df)
    comparison_occ = occ_ratio(comparison_df)
    selected_mat = ratio_from_columns(filtered_df, "mt_visits", "total_visits")
    comparison_mat = ratio_from_columns(comparison_df, "mt_visits", "total_visits")
    selected_cp = ratio_from_columns(filtered_df, "cp_visits", "total_visits")
    comparison_cp = ratio_from_columns(comparison_df, "cp_visits", "total_visits")
    selected_per_visit = (range_sales_display / selected_visits_total) if selected_visits_total else None
    comparison_per_visit = (comparison_sales / comparison_visits_total) if comparison_visits_total else None
    selected_ft = safe_sum(filtered_df, "first_time")
    comparison_ft = safe_sum(comparison_df, "first_time")

    def snap_card_html(label: str, current: Optional[float], comparison: Optional[float], kind: str, target: Optional[str] = None) -> str:
        current_str = fmt_value(current, kind)
        comparison_str = fmt_value(comparison, kind)
        delta = yoy_delta(current, comparison)
        if delta is None:
            delta_str = "<span class='snap-delta'>—</span>"
        else:
            color = "#19c37d" if delta >= 0 else "#ff4b4b"
            delta_str = f"<span class='snap-delta' style='color:{color};'>{delta*100:+.1f}%</span>"
        target_attr = f"data-tab-target='{target}'" if target else ""
        tooltip_attr = f"data-tooltip='{comparison_period_label}: {comparison_str}'"
        return (
            f"<div class='snap-card' {target_attr} {tooltip_attr}>"
            f"<div class='snap-label'>{label}</div>"
            f"<div class='snap-main'><span class='snap-value'>{current_str}</span>{delta_str}</div>"
            f"<div class='snap-sub'>LP {comparison_str}</div>"
            f"</div>"
        )

    cards = [
        ("Sales", range_sales_display, comparison_sales, "currency", "Sales $"),
        ("Occ %", selected_occ, comparison_occ, "percent", "Occupancy"),
        ("Mat %", selected_mat, comparison_mat, "percent", None),
        ("$ / Visit", selected_per_visit, comparison_per_visit, "number2", None),
        ("FT Visit", selected_ft, comparison_ft, "number", None),
        ("Visits", selected_visits_total, comparison_visits_total, "number", None),
        ("CP %", selected_cp, comparison_cp, "percent", None),
    ]

    snap_html = "<div class='snap-grid'>" + "".join(snap_card_html(*card) for card in cards) + "</div>"
    st.markdown(snap_html, unsafe_allow_html=True)

    components.html(
        """
        <script>
        const cards = window.parent.document.querySelectorAll('div.snap-card[data-tab-target]');
        cards.forEach(card => {
            if(card.dataset.bound === 'true') return;
            card.dataset.bound = 'true';
            card.addEventListener('click', () => {
                const target = card.getAttribute('data-tab-target');
                if(!target) return;
                const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                tabs.forEach(btn => {
                    if(btn.innerText.trim() === target) {
                        btn.click();
                    }
                });
            });
        });
        </script>
        """,
        height=0,
    )

with tab_forecast:
    if history_series.empty:
        st.info("Not enough historical data to project future sales.")
    else:
        future_min = end_date + timedelta(days=1)
        future_max = max_date + timedelta(days=365)
        if future_min > future_max:
            st.info("Extend your dataset to enable future projections.")
        else:
            default_end = min(future_max, future_min + timedelta(days=13))
            forecast_range_input = st.date_input(
                "Forecast range",
                value=(future_min, default_end),
                min_value=future_min,
                max_value=future_max,
                help="Select future dates to view projected net sales"
            )

            normalized_forecast_range = normalize_range(
                forecast_range_input,
                (future_min, default_end)
            )
            forecast_start = clamp_date(normalized_forecast_range[0], future_min, future_max)
            forecast_end = clamp_date(normalized_forecast_range[1], future_min, future_max)

            forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq="D")
            forecast_rows = []

            for ts in forecast_dates:
                target_ts = cast(pd.Timestamp, pd.Timestamp(ts))
                candidate = target_ts - pd.DateOffset(years=1)
                target_weekday = int(target_ts.dayofweek)
                weekday_history = weekday_index_map.get(target_weekday)

                if weekday_history is not None and len(weekday_history) > 0:
                    source_timestamp = closest_timestamp(weekday_history, candidate)
                else:
                    source_timestamp = closest_timestamp(history_index, candidate)

                base_value = float(history_series.loc[source_timestamp])
                projected = base_value * yoy_multiplier

                forecast_rows.append(
                    {
                        "date": target_ts,
                        "weekday": target_ts.strftime("%a"),
                        "netsales": projected,
                        "studio": selection_label,
                        "source_date": source_timestamp.date()
                    }
                )

            forecast_view = pd.DataFrame(forecast_rows)
            if forecast_view.empty:
                st.info("No forecast data for the selected range.")
            else:
                st.dataframe(format_table(forecast_view))


def calculate_occupancy_ratio(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    capacity_classes = (df["capacity"] * df["classes"]).fillna(0)
    denominator = capacity_classes.sum()
    if denominator == 0:
        return None
    numerator = df["total_visits"].fillna(0).sum()
    return numerator / denominator if denominator else None


with tab_occupancy:
    st.subheader("Occupancy Percentage")
    current_occ = calculate_occupancy_ratio(filtered_df)
    comparison_occ = calculate_occupancy_ratio(comparison_df)

    occ_col1, occ_col2 = st.columns(2)
    occ_col1.metric(
        "Selected range occupancy",
        value=f"{current_occ:.1%}" if current_occ is not None else "N/A"
    )
    occ_col2.metric(
        "Comparison occupancy",
        value=f"{comparison_occ:.1%}" if comparison_occ is not None else "N/A",
        delta=(
            f"{((current_occ - comparison_occ) / comparison_occ * 100):+.1f}%"
            if (current_occ is not None and comparison_occ not in (None, 0))
            else None
        )
    )

    def build_occupancy_table(df: pd.DataFrame) -> pd.DataFrame:
        table = df.copy()
        denom = (table["capacity"] * table["classes"]).replace({0: pd.NA})
        table["occupancy_pct"] = table["total_visits"] / denom
        table["occupancy_pct"] = table["occupancy_pct"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else ""
        )
        return table

    st.markdown("### Selected Range Occupancy Detail")
    st.dataframe(build_occupancy_table(filtered_df))

    st.markdown("### Comparison Range Occupancy Detail")
    if comparison_df.empty:
        st.info("No comparison data available to compute occupancy.")
    else:
        st.dataframe(build_occupancy_table(comparison_df))


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"${value:,.0f}"


def format_number(value: Optional[float], decimals: int = 0, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}{suffix}"


def format_percent(value: Optional[float], decimals: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{decimals}f}%"


def yoy_ratio(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous in (None, 0):
        return None
    return current / previous


def ratio_badge(ratio: Optional[float]) -> str:
    if ratio is None:
        return "<span class=\"fw-secondary\">—</span>"
    color = "#19c37d" if ratio >= 1 else "#ff4b4b"
    return f"<span style='color:{color}'>{ratio:.0%}</span>"


def render_fw_card(label: str, value: str, comparison_label: str, ratio_html: str) -> str:
    return f"""
    <div class='fw-card'>
        <div class='fw-label'>{label}</div>
        <div class='fw-value'>{value}</div>
        <div class='fw-sub'>{comparison_label}</div>
        <div class='fw-ratio'>{ratio_html}</div>
    </div>
    """


def render_fw_row(title: str, value: str, subtitle: str, ratio_html: str) -> str:
    return f"""
    <div class='fw-row'>
        <div class='fw-row-title'>{title}</div>
        <div class='fw-row-value'>{value}</div>
        <div class='fw-row-sub'>{subtitle}</div>
        <div class='fw-row-ratio'>{ratio_html}</div>
    </div>
    """


def render_summary_content():
    st.markdown(
        """
        <style>
        .fw-card, .fw-row {
            background: #1f1f1f;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
            font-family: "Inter", "Segoe UI", sans-serif;
            color: #f5f5f5;
        }
        .fw-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05rem;
            color: #bdbdbd;
        }
        .fw-value {
            font-size: 1.6rem;
            font-weight: 600;
            color: #f4b400;
        }
        .fw-sub, .fw-row-sub {
            font-size: 0.8rem;
            color: #aaaaaa;
        }
        .fw-ratio, .fw-row-ratio {
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.2rem;
        }
        .fw-row-title {
            font-size: 1rem;
            font-weight: 600;
        }
        .fw-row-value {
            font-size: 1.2rem;
            color: #f4b400;
            font-weight: 600;
        }
        .fw-section-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .fw-secondary {
            color: #7b7b7b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    studio_fw_df = cast(pd.DataFrame, studio_df.copy())
    date_series = pd.to_datetime(studio_fw_df["date"], errors="coerce")
    studio_fw_df["date"] = date_series
    studio_fw_df["week_start"] = date_series.dt.to_period("W-SUN").dt.start_time

    month_start_ts = cast(pd.Timestamp, pd.Timestamp(end_date.replace(day=1)))
    month_df = studio_fw_df.loc[
        (studio_fw_df["date"] >= month_start_ts) & (studio_fw_df["date"] <= end_ts)
    ].copy()
    days_covered = (end_ts - month_start_ts).days
    prev_month_start = cast(pd.Timestamp, month_start_ts - pd.DateOffset(years=1))
    prev_month_end = cast(pd.Timestamp, prev_month_start + pd.Timedelta(days=days_covered))
    prev_month_df = studio_fw_df.loc[
        (studio_fw_df["date"] >= prev_month_start) & (studio_fw_df["date"] <= prev_month_end)
    ].copy()

    def sum_optional(df: pd.DataFrame, column: str) -> Optional[float]:
        if column not in df.columns or df.empty:
            return None
        total = df[column].sum()
        return float(total) if pd.notna(total) else None

    month_est_sales = sum_optional(month_df, "est_sales")
    prev_est_sales = sum_optional(prev_month_df, "est_sales")

    month_est_visits = sum_optional(month_df, "est_visits")
    prev_est_visits = sum_optional(prev_month_df, "est_visits")

    def mat_pct(df: pd.DataFrame) -> Optional[float]:
        total = sum_optional(df, "total_visits")
        mt = sum_optional(df, "mt_visits")
        return (mt / total) if (mt is not None and total not in (None, 0)) else None

    month_mat = mat_pct(month_df)
    prev_mat = mat_pct(prev_month_df)

    def occ_pct(df: pd.DataFrame) -> Optional[float]:
        numer = sum_optional(df, "total_visits")
        denom = sum_optional(df, "capacity")
        classes = sum_optional(df, "classes")
        if denom in (None, 0) or classes in (None, 0):
            return None
        slots = sum_optional(df, "slots")
        if slots:
            return numer / slots if (numer is not None and slots not in (None, 0)) else None
        return None

    month_occ = calculate_occupancy_ratio(month_df)
    prev_occ = calculate_occupancy_ratio(prev_month_df)

    def per_visit(df: pd.DataFrame) -> Optional[float]:
        total_visits = sum_optional(df, "total_visits")
        sales = sum_optional(df, "netsales")
        return (sales / total_visits) if (sales is not None and total_visits not in (None, 0)) else None

    month_per_visit = per_visit(month_df)
    prev_per_visit = per_visit(prev_month_df)

    month_ft = sum_optional(month_df, "first_time")
    prev_ft = sum_optional(prev_month_df, "first_time")

    prev_month_label = prev_month_end.strftime("%m/%d/%y") if not prev_month_df.empty else ""

    month_cards = [
        (
            "Est Sales",
            format_currency(range_sales_display),
            prev_month_label,
            ratio_badge(yoy_ratio(month_est_sales, prev_est_sales)),
        ),
        (
            "Est Visits",
            format_number(month_est_visits, 0),
            prev_month_label,
            ratio_badge(yoy_ratio(month_est_visits, prev_est_visits)),
        ),
        (
            "Occ %",
            format_percent(month_occ),
            prev_month_label,
            ratio_badge(yoy_ratio(month_occ, prev_occ)),
        ),
        (
            "$ / Visit",
            format_number(month_per_visit, 2),
            prev_month_label,
            ratio_badge(yoy_ratio(month_per_visit, prev_per_visit)),
        ),
        (
            "FT Visit",
            format_number(month_ft, 0),
            prev_month_label,
            ratio_badge(yoy_ratio(month_ft, prev_ft)),
        ),
    ]

    weekly_totals = studio_fw_df.groupby("week_start")["netsales"].sum().sort_index(ascending=False)
    weekly_rows = weekly_totals.head(6).reset_index()

    daily_totals = studio_fw_df.groupby("date")["netsales"].sum().sort_index(ascending=False)
    daily_rows = daily_totals.head(6).reset_index()

    col_month, col_week, col_day = st.columns([1.2, 1, 1])

    with col_month:
        st.markdown("<div class='fw-section-title'>MTD</div>", unsafe_allow_html=True)
        month_cards_html = "".join(
            render_fw_card(label, value, subtitle, ratio) for label, value, subtitle, ratio in month_cards
        )
        st.markdown(month_cards_html, unsafe_allow_html=True)

    with col_week:
        st.markdown("<div class='fw-section-title'>Weekly Sales</div>", unsafe_allow_html=True)
        weekly_html_parts = []
        for row in weekly_rows.to_dict("records"):
            week_start = cast(pd.Timestamp, pd.Timestamp(row["week_start"]))
            week_value = float(row["netsales"])
            prev_week = cast(pd.Timestamp, week_start - pd.Timedelta(weeks=52))
            prev_value = weekly_totals.get(prev_week)
            weekly_html_parts.append(
                render_fw_row(
                    week_start.strftime("%m/%d/%y"),
                    format_currency(week_value),
                    prev_week.strftime("%m/%d/%y"),
                    ratio_badge(yoy_ratio(week_value, prev_value)),
                )
            )
        st.markdown("".join(weekly_html_parts), unsafe_allow_html=True)

    with col_day:
        st.markdown("<div class='fw-section-title'>Daily Sales</div>", unsafe_allow_html=True)
        daily_html_parts = []
        for row in daily_rows.to_dict("records"):
            day = cast(pd.Timestamp, pd.Timestamp(row["date"]))
            day_value = float(row["netsales"])
            prev_day = cast(pd.Timestamp, day - pd.DateOffset(years=1))
            prev_day_value = daily_totals.get(prev_day)
            daily_html_parts.append(
                render_fw_row(
                    day.strftime("%m/%d/%y"),
                    format_currency(day_value),
                    prev_day.strftime("%m/%d/%y"),
                    ratio_badge(yoy_ratio(day_value, prev_day_value)),
                )
            )
        st.markdown("".join(daily_html_parts), unsafe_allow_html=True)


with tab_fw_dashboard:
    render_summary_content()

with tab_sales_money:
    st.markdown(
        """
        <style>
        .sales-dollar-card {background:#0b1124;border:1px solid #2a3154;border-radius:16px;padding:0.85rem 1.1rem;margin-bottom:0.85rem;}
        .sales-dollar-card-label {font-size:0.8rem;color:#aeb3d1;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;}
        .sales-dollar-card-main {display:flex;justify-content:space-between;align-items:center;}
        .sales-dollar-card-value {font-size:1.6rem;font-weight:600;color:#f5c746;}
        .sales-dollar-card-delta {font-size:0.95rem;font-weight:600;}
        .sales-dollar-card-sub {font-size:0.75rem;color:#aeb3d1;margin-top:0.2rem;}
        .sales-entry-card {background:#10121a;border:1px solid #252d47;border-radius:14px;padding:0.7rem 0.9rem;margin-bottom:0.6rem;}
        .sales-entry-header {display:flex;justify-content:space-between;align-items:center;}
        .sales-entry-title {font-size:0.95rem;font-weight:600;color:#f5f5ff;}
        .sales-entry-value {font-size:1.2rem;font-weight:600;color:#f5c746;}
        .sales-entry-meta {font-size:0.75rem;color:#aeb3d1;margin-top:0.15rem;}
        .sales-dollar-card[data-tooltip], .sales-entry-card[data-tooltip] {position:relative;}
        .sales-dollar-card[data-tooltip]::before, .sales-entry-card[data-tooltip]::before {
            content:attr(data-tooltip);
            position:absolute;
            top:-8px;
            left:50%;
            transform:translate(-50%, -100%);
            background:#0f1324;
            color:#f5c746;
            border:1px solid #f5c746;
            border-radius:8px;
            padding:0.3rem 0.6rem;
            font-size:0.75rem;
            white-space:nowrap;
            pointer-events:none;
            opacity:0;
            transition:opacity 0.12s ease, transform 0.12s ease;
            box-shadow:0 6px 20px rgba(0,0,0,0.35);
            z-index:20;
        }
        .sales-dollar-card[data-tooltip]:hover::before, .sales-entry-card[data-tooltip]:hover::before {
            opacity:1;
            transform:translate(-50%, calc(-100% - 6px));
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def sales_card_delta(current: float, comparison: Optional[float]) -> str:
        if comparison in (None, 0.0):
            return "<span class='sales-dollar-card-delta'>—</span>"
        delta_pct = ((current - comparison) / comparison) * 100 if comparison else 0.0
        color = "#19c37d" if delta_pct >= 0 else "#ff4b4b"
        return f"<span class='sales-dollar-card-delta' style='color:{color};'>{delta_pct:+.1f}%</span>"

    def render_sales_card(label: str, amount: float, current_label: str, comparison_value: Optional[float], comparison_label: str) -> str:
        tooltip = "Comparison: —"
        if comparison_value not in (None, 0.0):
            tooltip = f"{comparison_label}: ${comparison_value:,.0f}"
        return (
            f"<div class='sales-dollar-card' data-tooltip='{tooltip}'>"
            f"<div class='sales-dollar-card-label'>{label}</div>"
            f"<div class='sales-dollar-card-main'><span class='sales-dollar-card-value'>${amount:,.0f}</span>{sales_card_delta(amount, comparison_value)}</div>"
            f"<div class='sales-dollar-card-sub'>{current_label}</div>"
            f"<div class='sales-dollar-card-sub'>Comparison: {comparison_label}</div>"
            "</div>"
        )

    def render_sales_entry_card(title: str, amount: float, comparison_label: str, comparison_value: Optional[float]) -> str:
        delta_html = sales_card_delta(amount, comparison_value)
        comparison_text = comparison_label if comparison_label else "—"
        tooltip = "Comparison: —"
        if comparison_value not in (None, 0.0):
            tooltip = f"{comparison_text}: ${comparison_value:,.0f}"
        return (
            f"<div class='sales-entry-card' data-tooltip='{tooltip}'>"
            f"<div class='sales-entry-header'><span class='sales-entry-title'>{title}</span>{delta_html}</div>"
            f"<div class='sales-entry-value'>${amount:,.0f}</div>"
            f"<div class='sales-entry-meta'>Comparison: {comparison_text}</div>"
            "</div>"
        )

    st.markdown("<div class='fw-section-title'>Sales Breakdown</div>", unsafe_allow_html=True)
    chart_cols = st.columns([1.4, 1])
    with chart_cols[0]:
        chart_data_current = build_chart_data(filtered_df, "Current", "Current").sort_values("date")
        chart_data_comparison = build_chart_data(comparison_df, "Comparison", "Comparison").sort_values("date")
        if chart_data_current.empty:
            st.info("Not enough data to display the snapshot chart.")
        else:
            chart_data_current["x_axis"] = chart_data_current["date"].dt.strftime("%b %d")
            chart_data_current["display_label"] = chart_data_current["date"].dt.strftime("%b %d, %Y")
            chart_data_current["comparison_label"] = chart_data_current["display_label"]

            comparison_trimmed = chart_data_comparison.head(len(chart_data_current)).copy()
            comparison_trimmed["x_axis"] = chart_data_current["x_axis"].values[:len(comparison_trimmed)]
            comparison_trimmed["display_label"] = chart_data_current["display_label"].values[:len(comparison_trimmed)]
            comparison_trimmed["comparison_label"] = comparison_trimmed["date"].dt.strftime("%b %d, %Y")

            chart_df = pd.concat([chart_data_current, comparison_trimmed], ignore_index=True)

            current_range = f"{start_date:%b %d} – {end_date:%b %d, %Y}"
            comparison_range = f"{comp_start_date:%b %d} – {comp_end_date:%b %d, %Y}"
            spacer = "&nbsp;" * 6
            current_delta = sales_card_delta(range_sales_display, comparison_sales)
            comparison_delta = sales_card_delta(comparison_sales, range_sales_display)
            header_html = (
                "<div class='sales-bar-container'>"
                "<div class='sales-bar-legend'>"
                f"<span><span class='sales-bar-dot' style='background:#545b78;'></span>Current: {current_range} <span style='color:#f5c746;'>${range_sales_display:,.0f}</span> {current_delta}</span>"
                f"<span style='margin-left:0.75rem;'>{spacer}</span>"
                f"<span><span class='sales-bar-dot' style='background:#f0652a;'></span>Comparison: {comparison_range} ${comparison_sales:,.0f} {comparison_delta}</span>"
                "</div>"
            )
            st.markdown(header_html, unsafe_allow_html=True)

            bar_chart = (
                alt.Chart(chart_df)
                .mark_bar(width=18, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("x_axis:N", title="", axis=alt.Axis(labelColor="#aeb3d1", labelPadding=8, labelAngle=0)),
                    xOffset="series:N",
                    y=alt.Y("netsales:Q", title="Net Sales", axis=alt.Axis(labelColor="#aeb3d1")),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(range=["#f0652a", "#545b78"]),
                        title="",
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("series:N", title="Series"),
                        alt.Tooltip("display_label:N", title="Current Date"),
                        alt.Tooltip("netsales:Q", title="Net Sales", format="$,.0f"),
                        alt.Tooltip("comparison_label:N", title="Comparison Date"),
                    ],
                )
                .properties(width=600)
            )
            st.altair_chart(bar_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    sales_cols = st.columns(2)

    summary_df = cast(pd.DataFrame, studio_df.copy())
    summary_df["date"] = pd.to_datetime(summary_df["date"], errors="coerce")
    summary_df["week_start"] = summary_df["date"].dt.to_period("W-SUN").dt.start_time

    comparison_summary_df = cast(pd.DataFrame, comparison_df.copy())
    comparison_summary_df["date"] = pd.to_datetime(comparison_summary_df["date"], errors="coerce")
    comparison_summary_df["week_start"] = comparison_summary_df["date"].dt.to_period("W-SUN").dt.start_time
    comparison_daily_totals = comparison_summary_df.groupby("date")["netsales"].sum().sort_index(ascending=False)
    comparison_weekly_totals = comparison_summary_df.groupby("week_start")["netsales"].sum().sort_index(ascending=False)

    with sales_cols[0]:
        st.markdown(
            render_sales_card(
                "Sales TD",
                month_sales_to_date,
                month_label_td,
                month_sales_to_date_comp,
                month_label_td_comp,
            ),
            unsafe_allow_html=True,
        )
        st.markdown("<div class='fw-section-title'>Daily Sales</div>", unsafe_allow_html=True)
        daily_totals = summary_df.groupby("date")["netsales"].sum().sort_index(ascending=False)
        daily_rows = daily_totals.head(6).reset_index()
        daily_html_parts = []
        comparison_daily_rows = comparison_daily_totals.head(len(daily_rows)).reset_index()
        for idx, row in enumerate(daily_rows.to_dict("records")):
            day = cast(pd.Timestamp, pd.Timestamp(row["date"]))
            day_value = float(row["netsales"])
            comp_label = "—"
            comp_value: Optional[float] = None
            if idx < len(comparison_daily_rows):
                comp_row = comparison_daily_rows.iloc[idx]
                comp_day = cast(pd.Timestamp, pd.Timestamp(comp_row["date"]))
                comp_value = float(comp_row["netsales"])
                comp_label = comp_day.strftime("%b %d, %Y")
            daily_html_parts.append(
                render_sales_entry_card(
                    day.strftime("%b %d, %Y"),
                    day_value,
                    comp_label,
                    comp_value,
                )
            )
        st.markdown("".join(daily_html_parts), unsafe_allow_html=True)

    with sales_cols[1]:
        st.markdown(
            render_sales_card(
                "Sales Est",
                month_sales_estimate,
                month_label_est,
                month_sales_estimate_comp,
                month_label_est_comp,
            ),
            unsafe_allow_html=True,
        )
        st.markdown("<div class='fw-section-title'>Weekly Sales</div>", unsafe_allow_html=True)
        weekly_totals = summary_df.groupby("week_start")["netsales"].sum().sort_index(ascending=False)
        weekly_rows = weekly_totals.head(6).reset_index()
        weekly_html_parts = []
        comparison_weekly_rows = comparison_weekly_totals.head(len(weekly_rows)).reset_index()
        for idx, row in enumerate(weekly_rows.to_dict("records")):
            week_start = cast(pd.Timestamp, pd.Timestamp(row["week_start"]))
            week_end = week_start + pd.Timedelta(days=6)
            week_value = float(row["netsales"])
            comp_label = "—"
            comp_value: Optional[float] = None
            if idx < len(comparison_weekly_rows):
                comp_row = comparison_weekly_rows.iloc[idx]
                comp_week = cast(pd.Timestamp, pd.Timestamp(comp_row["week_start"]))
                comp_week_end = comp_week + pd.Timedelta(days=6)
                comp_value = float(comp_row["netsales"])
                comp_label = f"{comp_week:%b %d} - {comp_week_end:%b %d, %Y}"
            weekly_html_parts.append(
                render_sales_entry_card(
                    f"{week_start:%b %d} - {week_end:%b %d, %Y}",
                    week_value,
                    comp_label,
                    comp_value,
                )
            )
        st.markdown("".join(weekly_html_parts), unsafe_allow_html=True)
