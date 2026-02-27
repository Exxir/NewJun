from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable, cast

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


def sum_or_zero(df: pd.DataFrame, column: str) -> float:
    total = safe_sum(df, column)
    return total if total is not None else 0.0


def combined_occupancy_ratio(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    mat_visits = sum_or_zero(df, "total_visits")
    mt_ref_visits = sum_or_zero(df, "mt_visits_ref")
    cp_ref_visits = sum_or_zero(df, "cp_visits_ref")
    numer = mat_visits + mt_ref_visits + cp_ref_visits
    def column_or_zero(name: str) -> pd.Series:
        if name in df.columns:
            series = cast(pd.Series, df[name])
            return series.fillna(0)
        return pd.Series(0.0, index=df.index)

    capacity_mat = column_or_zero("capacity_mat")
    classes_mat = column_or_zero("classes")
    capacity_ref = column_or_zero("capacity_ref")
    classes_ref = column_or_zero("class_ref")
    mat_slots = float((capacity_mat * classes_mat).sum())
    ref_slots = float((capacity_ref * classes_ref).sum())
    denom = mat_slots + ref_slots
    if denom == 0 or numer == 0:
        return None
    return numer / denom


def ratio_from_columns(df: pd.DataFrame, numer: str, denom: str) -> Optional[float]:
    num = safe_sum(df, numer)
    den = safe_sum(df, denom)
    if num is None or den in (None, 0):
        return None
    return num / den


def _series_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return cast(pd.Series, df[column]).fillna(0)
    return pd.Series(0.0, index=df.index)


def mat_occupancy(df: pd.DataFrame) -> Optional[float]:
    numer = safe_sum(df, "total_visits")
    if numer in (None, 0):
        return None
    capacity = _series_or_zero(df, "capacity_mat")
    classes = _series_or_zero(df, "classes")
    slots = float((capacity * classes).sum())
    if slots == 0:
        return None
    return numer / slots


def reformer_occupancy(df: pd.DataFrame) -> Optional[float]:
    mt_ref = safe_sum(df, "mt_visits_ref") or 0.0
    cp_ref = safe_sum(df, "cp_visits_ref") or 0.0
    numer = mt_ref + cp_ref
    if numer == 0:
        return None
    capacity = _series_or_zero(df, "capacity_ref")
    classes = _series_or_zero(df, "class_ref")
    slots = float((capacity * classes).sum())
    if slots == 0:
        return None
    return numer / slots


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


def build_chart_data(df: pd.DataFrame, series_label: str, range_label: str, column: str = "netsales") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            column: pd.Series(dtype=float),
            "series": pd.Series(dtype="string"),
            "weekday": pd.Series(dtype="string"),
            "range_label": pd.Series(dtype="string"),
        })
    grouped = (
        df.groupby("date")[column].sum().reset_index()
    )
    grouped["series"] = series_label
    grouped["weekday"] = grouped["date"].dt.strftime("%a")
    grouped["range_label"] = range_label
    return grouped.rename(columns={column: "netsales"})


def build_weekday_map(index: pd.DatetimeIndex) -> dict[int, pd.DatetimeIndex]:
    lookup: dict[int, pd.DatetimeIndex] = {}
    if len(index) == 0:
        return lookup
    weekday_series = pd.Series(index, index=index).dt.weekday
    for weekday in range(7):
        mask = weekday_series == weekday
        if mask.any():
            lookup[weekday] = weekday_series.index[mask]
    return lookup


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
    elif horizon in ("Monthly Estimate", "Estimate"):
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
    ".primary-header {font-size: 2.3rem; font-weight: 700; margin: 0;}"
    ".header-divider {border-bottom: 1px solid #1e2438; margin: 0.1rem 0 0.2rem;}"
    "div.block-container {padding-top: 0.5rem !important;}"
    "</style>"
    "<div class=\"primary-header\">Jungle Studios Dashboard</div>"
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
            "class_mat",
            "class_ref",
            "total_visits_mat",
            "mt_visits_ref",
            "cp_visits_mat",
            "cp_visit_ref",
            "ft_mat",
            "ft_ref",
            "cp_sales_mat",
            "cp_sales_ref",
            "mt_sales_mat",
            "mt_sales_ref",
            "mt_sales_total",
            "capacity",
            "capacity_ref"
        FROM public.studio_daily_metrics
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df["date"] = pd.to_datetime(df["date"])
    rename_map = {
        "class_mat": "classes",
        "total_visits_mat": "total_visits",
        "cp_visits_mat": "cp_visits",
        "ft_mat": "first_time",
        "cp_visit_ref": "cp_visits_ref",
    }
    df = df.rename(columns=rename_map)
    numeric_columns = (
        "classes",
        "class_ref",
        "total_visits",
        "mt_visits_ref",
        "cp_visits",
        "cp_visits_ref",
        "first_time",
        "ft_ref",
        "cp_sales_mat",
        "cp_sales_ref",
        "mt_sales_mat",
        "mt_sales_ref",
        "mt_sales_total",
        "capacity",
        "capacity_ref",
    )
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    def as_numeric_series(series: Optional[pd.Series]) -> pd.Series:
        if series is None:
            return pd.Series(pd.NA, index=df.index)
        return pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)

    df["capacity_mat"] = as_numeric_series(df.get("capacity"))
    df["capacity_ref"] = as_numeric_series(df.get("capacity_ref"))
    mat_classes = as_numeric_series(df.get("classes"))
    reformer_classes = as_numeric_series(df.get("class_ref"))
    mat_capacity = df["capacity_mat"].fillna(0)
    reformer_capacity = df["capacity_ref"].fillna(0)
    combined_capacity = mat_capacity + reformer_capacity
    combined_classes = mat_classes.fillna(0) + reformer_classes.fillna(0)
    df["capacity"] = combined_capacity
    df["classes_total"] = combined_classes

    def safe_sales_series(column: str) -> pd.Series:
        if column in df.columns:
            series = cast(pd.Series, df[column])
            return series.fillna(0)
        return pd.Series(0, index=df.index, dtype=float)

    sales_components = [
        safe_sales_series("cp_sales_mat"),
        safe_sales_series("cp_sales_ref"),
        safe_sales_series("mt_sales_mat"),
        safe_sales_series("mt_sales_ref"),
    ]
    df["netsales"] = sum(sales_components)
    cp_visits_series = df.get("cp_visits")
    if cp_visits_series is None:
        cp_visits_series = pd.Series(0, index=df.index, dtype=float)
    cp_visits_series = cp_visits_series.fillna(0)
    total_visits_series = df.get("total_visits")
    if total_visits_series is None:
        total_visits_series = cp_visits_series.copy()
    total_visits_series = total_visits_series.fillna(cp_visits_series)
    df["cp_visits"] = cp_visits_series
    df["total_visits"] = total_visits_series
    mt_visits_series = df.get("mt_visits")
    fallback_series = total_visits_series - cp_visits_series
    if mt_visits_series is None:
        mt_visits_series = fallback_series
    else:
        mt_visits_series = mt_visits_series.fillna(fallback_series)
    df["mt_visits"] = mt_visits_series.clip(lower=0)
    df["weekday"] = df["date"].dt.strftime("%A")
    if "capacity" not in df.columns:
        df["capacity"] = pd.NA
    return df


df = load_data()

# --- Studio Selector ---
studios = sorted(df["studio"].unique())
default_selection = studios[:1]
st.markdown('<div class="selector-title" style="font-weight:700;font-size:1.15em;">Jungle Dashboard</div>', unsafe_allow_html=True)
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

history_visits_series = studio_df.groupby("date")["total_visits"].sum().sort_index()
history_visits_index: pd.DatetimeIndex = pd.DatetimeIndex(history_visits_series.index)
weekday_index_map_visits: dict[int, pd.DatetimeIndex] = {}
if len(history_visits_index) > 0:
    visits_weekday_series = pd.Series(history_visits_index, index=history_visits_index).dt.weekday
    for weekday in range(7):
        mask = visits_weekday_series == weekday
        if mask.any():
            weekday_index_map_visits[weekday] = visits_weekday_series.index[mask]


horizon_options = ["Monthly", "Weekly", "Daily", "Custom"]
horizon_default = st.session_state.get("selected_horizon", "Monthly")
if horizon_default not in horizon_options:
    horizon_default = "Monthly"
default_index = horizon_options.index(horizon_default)
horizon = st.radio(
    "Select horizon",
    horizon_options,
    index=default_index,
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state["selected_horizon"] = horizon

if horizon == "Custom":
    default_start = max_date - timedelta(days=6)
    if default_start < min_date:
        default_start = min_date
    custom_range_input = st.date_input(
        "Custom range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    normalized_custom_range = normalize_range(custom_range_input, (default_start, max_date))
    start_date = clamp_date(normalized_custom_range[0], min_date, max_date)
    end_date = clamp_date(normalized_custom_range[1], min_date, max_date)
else:
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

if horizon == "Custom":
    custom_comp_enabled = st.checkbox(
        "Customize comparison period",
        value=st.session_state.get("custom_comp_enabled", False),
    )
    st.session_state["custom_comp_enabled"] = custom_comp_enabled
    if custom_comp_enabled:
        custom_comp_input = st.date_input(
            "Comparison range",
            value=(comp_start_date, comp_end_date),
            min_value=min_date,
            max_value=max_date,
            key="custom_comparison_range",
        )
        normalized_comp_range = normalize_range(custom_comp_input, (comp_start_date, comp_end_date))
        comp_start_date = clamp_date(normalized_comp_range[0], min_date, max_date)
        comp_end_date = clamp_date(normalized_comp_range[1], min_date, max_date)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)
comp_start_ts = pd.Timestamp(comp_start_date)
comp_end_ts = pd.Timestamp(comp_end_date)

actual_end_ts = cast(pd.Timestamp, end_ts)
if False:
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
range_trips = filtered_df["total_visits"].sum()
range_trips_display = range_trips

comparison_selection_df = studio_df[
    (studio_df["date"] >= comp_start_ts) &
    (studio_df["date"] <= comp_end_ts)
]
comparison_df: pd.DataFrame = pd.DataFrame(comparison_selection_df).copy()
comparison_sales = comparison_df["netsales"].sum() if not comparison_df.empty else 0.0
yoy_multiplier = 1.0
yoy_visits_multiplier = 1.0
comparison_trips = comparison_df["total_visits"].sum() if not comparison_df.empty else 0.0
comparison_delta_pct = None
if comparison_df.empty or comparison_sales == 0:
    comparison_delta_pct = None
else:
    diff_pct = ((range_sales - comparison_sales) / comparison_sales) * 100
    comparison_delta_pct = f"{diff_pct:+.1f}%"
    yoy_multiplier = range_sales / comparison_sales if comparison_sales else 1.0

if comparison_trips == 0:
    yoy_visits_multiplier = 1.0
else:
    yoy_visits_multiplier = range_trips / comparison_trips if comparison_trips else 1.0


def project_sales_for_dates(date_sequence: Sequence[pd.Timestamp]) -> List[Tuple[float, Optional[pd.Timestamp]]]:
    projections: List[Tuple[float, Optional[pd.Timestamp]]] = []
    if len(history_series) == 0:
        return [(0.0, None) for _ in date_sequence]
    for date_item in date_sequence:
        target_ts = cast(pd.Timestamp, pd.Timestamp(date_item))
        weekday = int(target_ts.dayofweek)
        weekday_history = weekday_index_map.get(weekday)
        history_pool = weekday_history if (weekday_history is not None and len(weekday_history) > 0) else history_index
        if history_pool is None or len(history_pool) == 0:
            projections.append((0.0, None))
            continue
        history_pool = cast(pd.DatetimeIndex, history_pool)
        candidate = target_ts - pd.DateOffset(years=1)
        if candidate <= history_pool[0]:
            source_timestamp = cast(pd.Timestamp, history_pool[0])
        elif candidate >= history_pool[-1]:
            source_timestamp = cast(pd.Timestamp, history_pool[-1])
        else:
            source_timestamp = closest_timestamp(history_pool, candidate)
        base_value = float(history_series.loc[source_timestamp])
        projected = base_value * yoy_multiplier
        projections.append((projected, source_timestamp))
    return projections


def project_visits_for_dates(date_sequence: Sequence[pd.Timestamp]) -> List[Tuple[float, Optional[pd.Timestamp]]]:
    projections: List[Tuple[float, Optional[pd.Timestamp]]] = []
    if len(history_visits_series) == 0:
        return [(0.0, None) for _ in date_sequence]
    for date_item in date_sequence:
        target_ts = cast(pd.Timestamp, pd.Timestamp(date_item))
        weekday = int(target_ts.dayofweek)
        weekday_history = weekday_index_map_visits.get(weekday)
        history_pool = weekday_history if (weekday_history is not None and len(weekday_history) > 0) else history_visits_index
        if history_pool is None or len(history_pool) == 0:
            projections.append((0.0, None))
            continue
        history_pool = cast(pd.DatetimeIndex, history_pool)
        candidate = target_ts - pd.DateOffset(years=1)
        if candidate <= history_pool[0]:
            source_timestamp = cast(pd.Timestamp, history_pool[0])
        elif candidate >= history_pool[-1]:
            source_timestamp = cast(pd.Timestamp, history_pool[-1])
        else:
            source_timestamp = closest_timestamp(history_pool, candidate)
        base_value = float(history_visits_series.loc[source_timestamp])
        projected = base_value * yoy_visits_multiplier
        projections.append((projected, source_timestamp))
    return projections

forecast_extra_total = 0.0
estimated_rows: List[Dict[str, Any]] = []
range_sales_display = range_sales

forecast_values: List[float] = []

month_start_ts = cast(pd.Timestamp, pd.Timestamp(start_date))
month_end_ts = cast(pd.Timestamp, pd.Timestamp(end_date))

if horizon == "Estimate":

    actual_range = studio_df[
        (studio_df["date"] >= month_start_ts) &
        (studio_df["date"] <= actual_end_ts)
    ]
    actual_total = actual_range["netsales"].sum()

    remaining_dates = pd.date_range(start=actual_end_ts + timedelta(days=1), end=month_end_ts)
    if not remaining_dates.empty:
        projected_values = project_sales_for_dates(list(remaining_dates))
        forecast_rows = [value for value, _ in projected_values]
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
forecast_increment = max(range_sales_display - range_sales, 0.0)


def sum_sales_between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, column: str = "netsales") -> float:
    window = df[(df["date"] >= start) & (df["date"] <= end)]
    if window.empty or column not in window.columns:
        return 0.0
    total = window[column].sum()
    return float(total) if pd.notna(total) else 0.0

month_reference_ts = cast(pd.Timestamp, min(pd.Timestamp(actual_end_ts), pd.Timestamp(end_date)))

month_start_ts = cast(pd.Timestamp, pd.Timestamp(month_reference_ts).replace(day=1))
month_last_day = monthrange(month_start_ts.year, month_start_ts.month)[1]
full_month_end_ts = cast(pd.Timestamp, month_start_ts.replace(day=month_last_day))
actual_month_end = cast(pd.Timestamp, min(month_reference_ts, full_month_end_ts))
month_end_ts = full_month_end_ts

month_to_date_df = cast(pd.DataFrame, studio_df[(studio_df["date"] >= month_start_ts) & (studio_df["date"] <= actual_month_end)])
month_sales_to_date = float(month_to_date_df["netsales"].sum()) if not month_to_date_df.empty else 0.0

monthly_projection_remaining_total = 0.0
remaining_month_dates = pd.date_range(start=actual_month_end + timedelta(days=1), end=full_month_end_ts)
if not remaining_month_dates.empty:
    monthly_projection_remaining_total = sum(value for value, _ in project_sales_for_dates(list(remaining_month_dates)))

full_month_estimate_total = month_sales_to_date + monthly_projection_remaining_total

month_sales_estimate = full_month_estimate_total
month_sales_to_date_display = month_sales_to_date
month_label_td = (
    f"Sales MTD: {month_start_ts:%b %d} – {actual_month_end:%b %d}"
    if month_sales_to_date
    else "Sales MTD: No data"
)
month_label_est = f"Sales Est: {month_start_ts:%b %d} – {full_month_end_ts:%b %d, %Y}"

month_td_span = month_reference_ts - month_start_ts
month_td_comp_start = cast(pd.Timestamp, comp_start_ts)
month_td_comp_end = cast(pd.Timestamp, min(comp_start_ts + month_td_span, comp_end_ts))
month_sales_to_date_comp = sum_sales_between(comparison_df, month_td_comp_start, month_td_comp_end)
comparison_month_start = cast(pd.Timestamp, pd.Timestamp(comp_start_date).replace(day=1))
comparison_month_last_day = monthrange(comparison_month_start.year, comparison_month_start.month)[1]
comparison_month_end = cast(pd.Timestamp, comparison_month_start.replace(day=comparison_month_last_day))
month_sales_estimate_comp = sum_sales_between(studio_df, comparison_month_start, comparison_month_end)
month_label_td_comp = f"{month_td_comp_start:%b %d} – {month_td_comp_end:%b %d}"
month_label_est_comp = f"{comparison_month_start:%b %d} – {comparison_month_end:%b %d, %Y}"
month_visits_to_date = sum_or_zero(month_to_date_df, "total_visits")
monthly_projection_remaining_visits = 0.0
if not remaining_month_dates.empty:
    monthly_projection_remaining_visits = sum(value for value, _ in project_visits_for_dates(list(remaining_month_dates)))
full_month_visits_estimate_total = month_visits_to_date + monthly_projection_remaining_visits
month_visits_estimate_comp = sum_sales_between(studio_df, comparison_month_start, comparison_month_end, column="total_visits")
month_visits_estimate_delta_pct = None
if month_visits_estimate_comp not in (None, 0):
    month_visits_estimate_delta_pct = ((full_month_visits_estimate_total - month_visits_estimate_comp) / month_visits_estimate_comp) * 100
month_visits_to_date_comp = sum_sales_between(comparison_df, month_td_comp_start, month_td_comp_end, column="total_visits")
month_standard_to_date = sum_or_zero(month_to_date_df, "mt_visits")
month_classpass_to_date = sum_or_zero(month_to_date_df, "cp_visits")
month_standard_comp = sum_sales_between(comparison_df, month_td_comp_start, month_td_comp_end, column="mt_visits")
month_classpass_comp = sum_sales_between(comparison_df, month_td_comp_start, month_td_comp_end, column="cp_visits")
month_sales_estimate_delta_pct = None
if month_sales_estimate_comp not in (None, 0):
    month_sales_estimate_delta_pct = ((month_sales_estimate - month_sales_estimate_comp) / month_sales_estimate_comp) * 100

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
tab_snap, tab_sales_money, tab_trips, tab_occ_percent, tab_sales, tab_chart, tab_visits, tab_forecast, tab_occupancy, tab_fw_dashboard = st.tabs(["Snap", "Sales", "Visits", "Occ %", "Current", "Chart", "Clients", "Forecast", "Occupancy", "Summary"])

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

selected_occ = combined_occupancy_ratio(filtered_df)
comparison_occ = combined_occupancy_ratio(comparison_df)
selected_cp = ratio_from_columns(filtered_df, "cp_visits", "total_visits")
comparison_cp = ratio_from_columns(comparison_df, "cp_visits", "total_visits")
selected_mat_occ = mat_occupancy(filtered_df)
comparison_mat_occ = mat_occupancy(comparison_df)
selected_reformer_occ = reformer_occupancy(filtered_df)
comparison_reformer_occ = reformer_occupancy(comparison_df)


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

with tab_occ_percent:
    range_label = f"{start_date:%b %d} – {end_date:%b %d, %Y}"
    comparison_label = f"{comp_start_date:%b %d} – {comp_end_date:%b %d, %Y}"

    metric_definitions: dict[str, dict[str, Any]] = {
        "occupancy": {
            "label": "Occupancy %",
            "compute": combined_occupancy_ratio,
        },
        "classpass": {
            "label": "Classpass %",
            "compute": lambda df: ratio_from_columns(df, "cp_visits", "total_visits"),
        },
        "mat_occ": {
            "label": "Mat Occ %",
            "compute": mat_occupancy,
        },
        "reformer_occ": {
            "label": "Reformer Occ %",
            "compute": reformer_occupancy,
        },
    }

    metric_keys = list(metric_definitions.keys())
    metric_labels = {
        "occupancy": "Occ %",
        "classpass": "CP %",
        "mat_occ": "Mat %",
        "reformer_occ": "Ref %",
    }
    default_metric = st.session_state.get("occ_chart_metric", "occupancy")
    if default_metric not in metric_keys:
        default_metric = metric_keys[0]
    default_index = metric_keys.index(default_metric)
    active_metric_key = st.radio(
        "Occ Metric",
        metric_keys,
        index=default_index,
        key="occ_metric_radio",
        label_visibility="collapsed",
        format_func=lambda key: metric_labels.get(key, key.title()),
        horizontal=True,
    )
    st.session_state["occ_chart_metric"] = active_metric_key
    active_metric = metric_definitions[active_metric_key]
    st.markdown(
        "<style>[data-testid='stRadio'][aria-label='Occ Metric']{display:none !important;}.occ-toggle-links{display:flex;gap:0.6rem;align-items:center;margin-top:0.4rem;margin-bottom:0.6rem;font-size:0.85rem;color:#aeb3d1;}.occ-toggle-link{background:#0b1124;border:1px solid #2a3154;color:#f5c746;padding:0.2rem 0.75rem;border-radius:999px;font-size:0.8rem;cursor:pointer;transition:background 0.15s ease,color 0.15s ease;border-color:#2a3154;}.occ-toggle-link[data-active='true']{background:#f5c746;color:#0b1124;border-color:#f5c746;font-weight:600;} .occ-toggle-link:hover{opacity:0.85;}</style>",
        unsafe_allow_html=True,
    )

    def format_occ_percent(value: Optional[float]) -> str:
        if value is None or pd.isna(value):
            return "—"
        return f"{value * 100:.1f}%"

    def occ_card_delta(current: Optional[float], comparison: Optional[float]) -> str:
        if current in (None, 0) or comparison in (None, 0):
            return "—"
        delta_pct = ((current - comparison) / comparison) * 100
        color = "#19c37d" if delta_pct >= 0 else "#ff4b4b"
        return f"<span style='color:{color};font-weight:600;'>{delta_pct:+.1f}%</span>"

    def render_occ_card(metric_key: str, value: Optional[float], current_text: str, comparison_value: Optional[float], comparison_text: str) -> str:
        tooltip = "Comparison: —"
        if comparison_value not in (None, 0):
            tooltip = f"{comparison_text}: {format_occ_percent(comparison_value)}"
        delta_html = occ_card_delta(value, comparison_value)
        display_value = format_occ_percent(value)
        return (
            f"<div class='sales-dollar-card occ-card' data-tooltip='{tooltip}' data-occ-target='{metric_key}'>"
            f"<div class='sales-dollar-card-main'><span class='sales-dollar-card-value'>{display_value}</span>{delta_html}</div>"
            f"<div class='sales-dollar-card-sub' style='color:#f5c746;'>{current_text}</div>"
            f"<div class='sales-dollar-card-sub'>Comparison: {comparison_text}</div>"
            "</div>"
        )

    occ_main_cols = st.columns([1, 1])
    with occ_main_cols[0]:
        st.markdown("<div class='fw-section-title'>Occupancy %</div>", unsafe_allow_html=True)
        st.markdown(
            render_occ_card("occupancy", selected_occ, f"Occupancy: {range_label}", comparison_occ, comparison_label),
            unsafe_allow_html=True,
        )

    with occ_main_cols[1]:
        st.markdown("<div class='fw-section-title'>Classpass %</div>", unsafe_allow_html=True)
        st.markdown(
            render_occ_card("classpass", selected_cp, f"Classpass: {range_label}", comparison_cp, comparison_label),
            unsafe_allow_html=True,
        )

    occ_mix_cols = st.columns([1, 1])
    with occ_mix_cols[0]:
        st.markdown("<div class='fw-section-title'>Mat Occ %</div>", unsafe_allow_html=True)
        st.markdown(
            render_occ_card("mat_occ", selected_mat_occ, f"Mat Occ: {range_label}", comparison_mat_occ, comparison_label),
            unsafe_allow_html=True,
        )

    with occ_mix_cols[1]:
        st.markdown("<div class='fw-section-title'>Reformer Occ %</div>", unsafe_allow_html=True)
        st.markdown(
            render_occ_card("reformer_occ", selected_reformer_occ, f"Reformer Occ: {range_label}", comparison_reformer_occ, comparison_label),
            unsafe_allow_html=True,
        )

    def render_toggle_links(active_key: str) -> None:
        link_html = "".join(
            f"<button type='button' class='occ-toggle-link' data-occ-target='{key}' data-active={'true' if key == active_key else 'false'}>{metric_labels.get(key, key.title())}</button>"
            for key in metric_keys
        )
        st.markdown(f"<div class='occ-toggle-links'>{link_html}</div>", unsafe_allow_html=True)


    def build_occ_chart_df(df: pd.DataFrame, label: str, compute_fn: Callable[[pd.DataFrame], Optional[float]]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=pd.Index(["date", "value", "series"]))
        rows = []
        for date_value, group in df.groupby("date"):
            occ_value = compute_fn(group)
            if occ_value is None or pd.isna(occ_value):
                continue
            rows.append({
                "date": date_value,
                "value": occ_value,
                "series": label,
            })
        return pd.DataFrame(rows)

    metric_compute = cast(Callable[[pd.DataFrame], Optional[float]], active_metric["compute"])
    active_metric_label = active_metric["label"]
    st.markdown("<div class='fw-section-title'>Occupancy Breakdown</div>", unsafe_allow_html=True)
    current_occ_chart = build_occ_chart_df(filtered_df, "Current", metric_compute)
    comparison_occ_chart = build_occ_chart_df(comparison_df, "Comparison", metric_compute)
    if current_occ_chart.empty and comparison_occ_chart.empty:
        st.info("Not enough data to display the occupancy breakdown chart.")
    else:
        def enrich_chart_df(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame
            enriched = frame.sort_values("date").copy()
            enriched["x_axis"] = enriched["date"].dt.strftime("%b %d")
            enriched["display_label"] = enriched["date"].dt.strftime("%b %d, %Y")
            enriched["comparison_label"] = enriched["display_label"]
            return enriched

        current_occ_chart = enrich_chart_df(current_occ_chart)
        comparison_occ_chart = enrich_chart_df(comparison_occ_chart)

        if current_occ_chart.empty:
            occ_chart_df = comparison_occ_chart
        elif comparison_occ_chart.empty:
            occ_chart_df = current_occ_chart
        else:
            comparison_trimmed = comparison_occ_chart.head(len(current_occ_chart)).copy()
            comparison_trimmed["x_axis"] = current_occ_chart["x_axis"].values[:len(comparison_trimmed)]
            occ_chart_df = pd.concat([current_occ_chart, comparison_trimmed], ignore_index=True)

        metric_value_current = active_metric["compute"](filtered_df)
        metric_value_comparison = active_metric["compute"](comparison_df)
        current_delta_html = occ_card_delta(metric_value_current, metric_value_comparison)
        comparison_delta_html = occ_card_delta(metric_value_comparison, metric_value_current)
        header_html = (
            "<div class='sales-bar-container legend-dual'>"
            "<div class='legend-row'>"
            f"<span class='legend-entry'><span class='legend-swatch' style='background:#cda643;'></span><span class='legend-label'>Current</span><span class='legend-value' style='color:#cda643;'>{format_occ_percent(metric_value_current)}</span><span class='legend-delta'>{current_delta_html}</span></span>"
            f"<span class='legend-entry'><span class='legend-swatch' style='background:#3f4a78;'></span><span class='legend-label'>Comparison</span><span class='legend-value' style='color:#3f4a78;'>{format_occ_percent(metric_value_comparison)}</span><span class='legend-delta'>{comparison_delta_html}</span></span>"
            "</div>"
            "</div>"
        )
        st.markdown(header_html, unsafe_allow_html=True)

        occ_chart = (
            alt.Chart(occ_chart_df)
            .mark_bar(width=14, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("x_axis:N", title="", axis=alt.Axis(labelColor="#aeb3d1", labelPadding=8, labelAngle=0)),
                xOffset="series:N",
                y=alt.Y("value:Q", title=active_metric_label, axis=alt.Axis(format="%", labelColor="#aeb3d1")),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(range=["#3f4a78", "#cda643"], domain=["Comparison", "Current"]),
                    title="",
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("display_label:N", title="Date"),
                    alt.Tooltip("value:Q", title="Occupancy %", format=".1%"),
                ],
            )
            .properties(width=1187, height=240)
        )
        st.altair_chart(occ_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    render_toggle_links(active_metric_key)

    components.html(
        """
        <script>
        const doc = window.parent.document;
        const cards = doc.querySelectorAll('.occ-card[data-occ-target]');
        const radioRoot = doc.querySelector('[data-testid="stRadio"][aria-label="Occ Metric"]');
        const links = doc.querySelectorAll('.occ-toggle-link[data-occ-target]');
        const triggerSelection = (target) => {
            if(!radioRoot) return;
            const inputs = radioRoot.querySelectorAll('input[type="radio"]');
            inputs.forEach(input => {
                if(input.value === target) {
                    input.click();
                }
            });
        };
        const bindElement = (el) => {
            if(el.dataset.bound === 'true') return;
            el.dataset.bound = 'true';
            el.addEventListener('click', () => {
                const target = el.dataset.occTarget;
                if(!target) return;
                triggerSelection(target);
            });
        };
        cards.forEach(bindElement);
        links.forEach(bindElement);
        </script>
        """,
        height=0,
    )


    def occupancy_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=pd.Index(["period", "value"]))
        view = df.copy()
        view["date"] = pd.to_datetime(view["date"], errors="coerce")
        if period == "weekly":
            view["period"] = view["date"].dt.to_period("W-SUN").dt.start_time
        else:
            view["period"] = view["date"]
        rows = []
        for period_value, group in view.groupby("period"):
            ratio = combined_occupancy_ratio(group)
            if ratio is None or pd.isna(ratio):
                continue
            rows.append({"period": period_value, "value": ratio})
        return pd.DataFrame(rows).sort_values("period", ascending=False)

    def render_occ_entry_card(title: str, value: Optional[float], comparison_label: str, comparison_value: Optional[float]) -> str:
        tooltip = "Comparison: —"
        if comparison_value not in (None, 0):
            tooltip = f"{comparison_label}: {format_occ_percent(comparison_value)}"
        delta_html = occ_card_delta(value, comparison_value)
        display_value = format_occ_percent(value)
        return (
            f"<div class='sales-entry-card' data-tooltip='{tooltip}'>"
            f"<div class='sales-entry-header'><span class='sales-entry-title'>{title}</span>{delta_html}</div>"
            f"<div class='sales-entry-value'>{display_value}</div>"
            f"<div class='sales-entry-meta'>Comparison: {comparison_label}</div>"
            "</div>"
        )

    occ_summary_cols = st.columns(2)

    with occ_summary_cols[0]:
        st.markdown("<div class='fw-section-title'>Daily Occupancy</div>", unsafe_allow_html=True)
        daily_occ = occupancy_by_period(filtered_df, "daily").head(6)
        comparison_daily_occ = occupancy_by_period(comparison_df, "daily").head(len(daily_occ))
        if daily_occ.empty:
            st.info("No daily occupancy data available.")
        else:
            cards = []
            for idx, row in enumerate(daily_occ.to_dict("records")):
                period = cast(pd.Timestamp, row["period"])
                comparison_label = "—"
                comparison_value = None
                if idx < len(comparison_daily_occ):
                    comp_row = comparison_daily_occ.iloc[idx]
                    comparison_value = comp_row["value"]
                    comparison_label = cast(pd.Timestamp, comp_row["period"]).strftime("%b %d, %Y")
                cards.append(
                    render_occ_entry_card(
                        period.strftime("%b %d, %Y"),
                        row["value"],
                        comparison_label,
                        comparison_value,
                    )
                )
            st.markdown("".join(cards), unsafe_allow_html=True)

    with occ_summary_cols[1]:
        st.markdown("<div class='fw-section-title'>Weekly Occupancy</div>", unsafe_allow_html=True)
        weekly_occ = occupancy_by_period(filtered_df, "weekly").head(6)
        comparison_weekly_occ = occupancy_by_period(comparison_df, "weekly").head(len(weekly_occ))
        if weekly_occ.empty:
            st.info("No weekly occupancy data available.")
        else:
            cards = []
            for idx, row in enumerate(weekly_occ.to_dict("records")):
                week_start = cast(pd.Timestamp, row["period"])
                week_end = week_start + pd.Timedelta(days=6)
                comparison_label = "—"
                comparison_value = None
                if idx < len(comparison_weekly_occ):
                    comp_row = comparison_weekly_occ.iloc[idx]
                    comparison_value = comp_row["value"]
                    comp_week_start = cast(pd.Timestamp, comp_row["period"])
                    comparison_label = f"{comp_week_start:%b %d} – {(comp_week_start + pd.Timedelta(days=6)):%b %d}"
                cards.append(
                    render_occ_entry_card(
                        f"{week_start:%b %d} – {week_end:%b %d}",
                        row["value"],
                        comparison_label,
                        comparison_value,
                    )
                )
            st.markdown("".join(cards), unsafe_allow_html=True)


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

    selected_visits_total = safe_sum(filtered_df, "total_visits") or 0.0
    comparison_visits_total = safe_sum(comparison_df, "total_visits") or 0.0
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
            prefer_green = (label == "Classpass %")
            if prefer_green:
                color = "#19c37d"
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
        ("Sales", range_sales_display, comparison_sales, "currency", "Sales"),
        ("Visits", selected_visits_total, comparison_visits_total, "number", "Visits"),
        ("Occupancy %", selected_occ, comparison_occ, "percent", "Occupancy"),
        ("Classpass %", selected_cp, comparison_cp, "percent", None),
        ("Mat Occ %", selected_mat_occ, comparison_mat_occ, "percent", None),
        ("Reformer Occ %", selected_reformer_occ, comparison_reformer_occ, "percent", None),
        ("$ / Visit", selected_per_visit, comparison_per_visit, "number2", None),
        ("FT Visit", selected_ft, comparison_ft, "number", None),
    ]

    def render_snap_cards():
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

    render_snap_cards()
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
            projected_points = project_sales_for_dates(list(forecast_dates))
            forecast_rows = []

            for ts, (projected, source_timestamp) in zip(forecast_dates, projected_points):
                target_ts = cast(pd.Timestamp, pd.Timestamp(ts))
                forecast_rows.append(
                    {
                        "date": target_ts,
                        "weekday": target_ts.strftime("%a"),
                        "netsales": projected,
                        "studio": selection_label,
                        "source_date": source_timestamp.date() if source_timestamp is not None else None,
                    }
                )

            forecast_view = pd.DataFrame(forecast_rows)
            if forecast_view.empty:
                st.info("No forecast data for the selected range.")
            else:
                st.dataframe(format_table(forecast_view))

        # Monthly estimate summary
        month_container = st.container()
        with month_container:
            monthly_current_total = full_month_estimate_total
            prev_year_period_start = cast(pd.Timestamp, month_start_ts - pd.DateOffset(years=1))
            prev_year_period_end = cast(pd.Timestamp, full_month_end_ts - pd.DateOffset(years=1))
            previous_year_total = sum_sales_between(studio_df, prev_year_period_start, prev_year_period_end)
            delta_pct = None
            if previous_year_total not in (None, 0):
                delta_pct = ((monthly_current_total - previous_year_total) / previous_year_total) * 100
            comparison_label = f"Prior Year: {prev_year_period_start:%b %d} – {prev_year_period_end:%b %d, %Y}"
            delta_str = "—" if delta_pct is None else f"{delta_pct:+.1f}%"
            st.markdown(
                f"<div style='background:#0b1124;border:1px solid #2a3154;border-radius:16px;padding:0.8rem 1rem;margin-bottom:0.75rem;'>"
                f"<div style='font-size:0.85rem;color:#aeb3d1;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;'>Monthly Estimate</div>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='font-size:1.6rem;font-weight:600;color:#f5c746;'>${monthly_current_total:,.0f}</span>"
                f"<span style='font-size:0.95rem;font-weight:600;color:#19c37d;'>{delta_str}</span>"
                f"</div>"
                f"<div style='font-size:0.75rem;color:#aeb3d1;margin-top:0.25rem;'>"
                f"Current: {month_start_ts:%b %d} – {month_end_ts:%b %d, %Y}</div>"
                f"<div style='font-size:0.75rem;color:#aeb3d1;'>" + comparison_label + "</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def calculate_occupancy_ratio(df: pd.DataFrame) -> Optional[float]:
    return combined_occupancy_ratio(df)


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


def format_timestamp_label(ts: Optional[pd.Timestamp], fmt: str = "%b %d, %Y") -> str:
    if ts is None or cast(bool, pd.isna(ts)):
        return "—"
    ts_clean = cast(pd.Timestamp, ts)
    dt_value = date(int(ts_clean.year), int(ts_clean.month), int(ts_clean.day))
    return dt_value.strftime(fmt)


def format_week_range(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or cast(bool, pd.isna(ts)):
        return "—"
    ts_clean = cast(pd.Timestamp, ts)
    end_ts = ts_clean + pd.Timedelta(days=6)
    start_dt = date(int(ts_clean.year), int(ts_clean.month), int(ts_clean.day))
    end_dt = date(int(end_ts.year), int(end_ts.month), int(end_ts.day))
    start_str = start_dt.strftime("%b %d")
    end_str = end_dt.strftime("%b %d, %Y")
    return f"{start_str} - {end_str}"


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
            week_end = week_start + pd.Timedelta(days=6)
            week_value = float(row["netsales"])
            prev_year_week_start = align_date_to_weekday(week_start.date(), weekday_index_map, history_index)
            if prev_year_week_start is not None:
                prev_week_ts = pd.Timestamp(prev_year_week_start)
                prev_week_val = weekly_totals.get(prev_week_ts)
                weekly_label = prev_year_week_start.strftime("%m/%d/%y")
            else:
                prev_week_ts = None
                prev_week_val = None
                weekly_label = "—"
            weekly_html_parts.append(
                render_fw_row(
                    f"{week_start:%m/%d} - {week_end:%m/%d/%y}",
                    format_currency(week_value),
                    weekly_label,
                    ratio_badge(yoy_ratio(week_value, prev_week_val)),
                )
            )
        st.markdown("".join(weekly_html_parts), unsafe_allow_html=True)

    with col_day:
        st.markdown("<div class='fw-section-title'>Daily Sales</div>", unsafe_allow_html=True)
        daily_html_parts = []
        for row in daily_rows.to_dict("records"):
            day = cast(pd.Timestamp, pd.Timestamp(row["date"]))
            day_value = float(row["netsales"])
            aligned = align_date_to_weekday(day.date(), weekday_index_map, history_index)
            if aligned is not None:
                aligned_ts = pd.Timestamp(aligned)
                aligned_value = daily_totals.get(aligned_ts)
                aligned_label = aligned.strftime("%m/%d/%y")
            else:
                aligned_ts = None
                aligned_value = None
                aligned_label = "—"
            daily_html_parts.append(
                render_fw_row(
                    day.strftime("%m/%d/%y"),
                    format_currency(day_value),
                    aligned_label,
                    ratio_badge(yoy_ratio(day_value, aligned_value)),
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
        .legend-dual {margin-bottom:1rem;}
        .legend-row {display:flex;justify-content:center;flex-wrap:wrap;gap:1.5rem;font-size:0.8rem;color:#aeb3d1;}
        .legend-entry {display:flex;align-items:center;gap:0.35rem;}
        .legend-swatch {display:inline-block;width:10px;height:10px;border-radius:2px;}
        .legend-label {font-weight:600;text-transform:uppercase;letter-spacing:0.05em;}
        .legend-value {font-weight:600;color:#f5c746;}
        .legend-delta {font-weight:600;margin-left:0.2rem;}
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
            f"<div class='sales-dollar-card-main'><span class='sales-dollar-card-value'>${amount:,.0f}</span>{sales_card_delta(amount, comparison_value)}</div>"
            f"<div class='sales-dollar-card-sub' style='color:#f5c746;'>{current_label}</div>"
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


    def render_trips_card(amount: float, current_label: str, comparison_value: Optional[float], comparison_label: str) -> str:
        tooltip = "Comparison: —"
        if comparison_value not in (None, 0.0):
            tooltip = f"{comparison_label}: {format_number(comparison_value, 0)}"
        return (
            f"<div class='sales-dollar-card' data-tooltip='{tooltip}'>"
            f"<div class='sales-dollar-card-main'><span class='sales-dollar-card-value'>{format_number(amount, 0)}</span>{sales_card_delta(amount, comparison_value)}</div>"
            f"<div class='sales-dollar-card-sub' style='color:#f5c746;'>{current_label}</div>"
            f"<div class='sales-dollar-card-sub'>Comparison: {comparison_label}</div>"
            "</div>"
        )


    def render_trips_entry_card(title: str, amount: float, comparison_label: str, comparison_value: Optional[float]) -> str:
        delta_html = sales_card_delta(amount, comparison_value)
        comparison_text = comparison_label if comparison_label else "—"
        tooltip = "Comparison: —"
        if comparison_value not in (None, 0.0):
            tooltip = f"{comparison_text}: {format_number(comparison_value, 0)}"
        return (
            f"<div class='sales-entry-card' data-tooltip='{tooltip}'>"
            f"<div class='sales-entry-header'><span class='sales-entry-title'>{title}</span>{delta_html}</div>"
            f"<div class='sales-entry-value'>{format_number(amount, 0)}</div>"
            f"<div class='sales-entry-meta'>Comparison: {comparison_text}</div>"
            "</div>"
        )

    monthly_cols = st.columns([1, 1])

    with monthly_cols[0]:
        st.markdown("<div class='fw-section-title'>Monthly Sales To Date</div>", unsafe_allow_html=True)
        st.markdown(
            render_sales_card(
                "",
                month_sales_to_date_display,
                month_label_td,
                month_sales_to_date_comp,
                month_label_td_comp,
            ),
            unsafe_allow_html=True,
        )
    with monthly_cols[1]:
        st.markdown("<div class='fw-section-title'>Monthly Sales Estimate</div>", unsafe_allow_html=True)
        est_delta_label = (
            f"{month_sales_estimate_delta_pct:+.1f}%"
            if month_sales_estimate_delta_pct is not None
            else "—"
        )
        st.markdown(
            render_sales_card(
                "",
                month_sales_estimate,
                f"Sales Est: {month_start_ts:%b %d} – {full_month_end_ts:%b %d, %Y} <span style='color:#19c37d;font-weight:600;margin-left:0.35rem;'>{est_delta_label}</span>",
                month_sales_estimate_comp,
                f"Prior Year: {month_label_est_comp}",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='fw-section-title'>Sales Breakdown</div>", unsafe_allow_html=True)
    chart_cols = st.columns([1.25, 0.35])
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
            comparison_color = "#3f4a78"
            current_bar_color = "#cda643"
            header_html = (
                "<div class='sales-bar-container legend-dual'>"
                "<div class='legend-row'>"
                f"<span class='legend-entry'><span class='legend-swatch' style='background:{current_bar_color};'></span><span class='legend-label'>Current</span><span class='legend-value' style='color:{current_bar_color};'>${range_sales_display:,.0f}</span><span class='legend-period'>{current_range}</span><span class='legend-delta'>{current_delta}</span></span>"
                f"<span class='legend-entry'><span class='legend-swatch' style='background:{comparison_color};'></span><span class='legend-label'>Comparison</span><span class='legend-value' style='color:{comparison_color};'>${comparison_sales:,.0f}</span><span class='legend-period'>{comparison_range}</span><span class='legend-delta'>{comparison_delta}</span></span>"
                "</div>"
                "</div>"
            )
            st.markdown(header_html, unsafe_allow_html=True)

            bar_chart = (
                alt.Chart(chart_df)
                .mark_bar(width=14, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("x_axis:N", title="", axis=alt.Axis(labelColor="#aeb3d1", labelPadding=8, labelAngle=0)),
                    xOffset="series:N",
                    y=alt.Y("netsales:Q", title="Net Sales", axis=alt.Axis(labelColor="#aeb3d1")),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(range=[comparison_color, current_bar_color], domain=["Comparison", "Current"]),
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
                .properties(width=1187, height=240)
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
    comparison_daily_index = pd.DatetimeIndex(comparison_daily_totals.index)
    comparison_daily_map = build_weekday_map(cast(pd.DatetimeIndex, comparison_daily_index))
    comparison_weekly_index = pd.DatetimeIndex(comparison_weekly_totals.index)
    comparison_weekly_map = build_weekday_map(cast(pd.DatetimeIndex, comparison_weekly_index))

    with sales_cols[0]:
        st.markdown("<div class='fw-section-title'>Daily Sales</div>", unsafe_allow_html=True)
        daily_totals = summary_df.groupby("date")["netsales"].sum().sort_index(ascending=False)
        daily_rows = daily_totals.head(6).reset_index()
        daily_html_parts = []
        for row in daily_rows.to_dict("records"):
            day = cast(pd.Timestamp, pd.Timestamp(row["date"]))
            day_value = float(row["netsales"])
            comp_label = "—"
            comp_value: Optional[float] = None
            if len(comparison_daily_index) > 0:
                candidate = cast(date, (day - pd.Timedelta(weeks=52)).date())
                aligned_date = align_date_to_weekday(candidate, comparison_daily_map, comparison_daily_index)
                comp_ts = pd.Timestamp(aligned_date)
                if not bool(pd.isna(comp_ts)):
                    comp_ts_clean = cast(pd.Timestamp, comp_ts)
                    comp_raw = comparison_daily_totals.get(comp_ts)
                    if comp_raw is not None and not pd.isna(comp_raw):
                        comp_value = float(comp_raw)
                        comp_label = format_timestamp_label(comp_ts_clean)
            daily_html_parts.append(
                render_sales_entry_card(
                    format_timestamp_label(day),
                    day_value,
                    comp_label,
                    comp_value,
                )
            )
        st.markdown("".join(daily_html_parts), unsafe_allow_html=True)

    with sales_cols[1]:
        st.markdown("<div class='fw-section-title'>Weekly Sales</div>", unsafe_allow_html=True)
        weekly_totals = summary_df.groupby("week_start")["netsales"].sum().sort_index(ascending=False)
        weekly_rows = weekly_totals.head(6).reset_index()
        weekly_html_parts = []
        for row in weekly_rows.to_dict("records"):
            week_start = cast(pd.Timestamp, pd.Timestamp(row["week_start"]))
            week_end = week_start + pd.Timedelta(days=6)
            week_value = float(row["netsales"])
            comp_label = "—"
            comp_value: Optional[float] = None
            if len(comparison_weekly_index) > 0:
                candidate = cast(date, (week_start - pd.Timedelta(weeks=52)).date())
                aligned_week_start = align_date_to_weekday(candidate, comparison_weekly_map, comparison_weekly_index)
                aligned_week_ts = pd.Timestamp(aligned_week_start)
                if not bool(pd.isna(aligned_week_ts)):
                    aligned_week_ts_clean = cast(pd.Timestamp, aligned_week_ts)
                    comp_raw = comparison_weekly_totals.get(aligned_week_ts)
                    if comp_raw is not None and not pd.isna(comp_raw):
                        comp_value = float(comp_raw)
                        comp_label = format_week_range(aligned_week_ts_clean)
            weekly_html_parts.append(
                render_sales_entry_card(
                    format_week_range(week_start),
                    week_value,
                    comp_label,
                    comp_value,
                )
            )
        st.markdown("".join(weekly_html_parts), unsafe_allow_html=True)


with tab_trips:
    trips_month_cols = st.columns([1, 1])

    with trips_month_cols[0]:
        st.markdown("<div class='fw-section-title'>Monthly Trips To Date</div>", unsafe_allow_html=True)
        st.markdown(
            render_trips_card(
                month_visits_to_date,
                f"Trips MTD: {month_start_ts:%b %d} – {actual_month_end:%b %d}",
                month_visits_to_date_comp,
                f"Prior Year: {month_td_comp_start:%b %d} – {month_td_comp_end:%b %d}",
            ),
            unsafe_allow_html=True,
        )

    with trips_month_cols[1]:
        st.markdown("<div class='fw-section-title'>Monthly Trips Estimate</div>", unsafe_allow_html=True)
        trips_est_delta = (
            f"{month_visits_estimate_delta_pct:+.1f}%"
            if month_visits_estimate_delta_pct is not None
            else "—"
        )
        st.markdown(
            render_trips_card(
                full_month_visits_estimate_total,
                f"Trips Est: {month_start_ts:%b %d} – {full_month_end_ts:%b %d, %Y} <span style='color:#19c37d;font-weight:600;margin-left:0.35rem;'>{trips_est_delta}</span>",
                month_visits_estimate_comp,
                f"Prior Year: {month_label_est_comp}",
            ),
            unsafe_allow_html=True,
        )

    trips_mix_cols = st.columns([1, 1])
    with trips_mix_cols[0]:
        st.markdown("<div class='fw-section-title'>Standard Visits</div>", unsafe_allow_html=True)
        st.markdown(
            render_trips_card(
                month_standard_to_date,
                f"Standard: {month_start_ts:%b %d} – {actual_month_end:%b %d}",
                month_standard_comp,
                f"Prior Year: {month_td_comp_start:%b %d} – {month_td_comp_end:%b %d}",
            ),
            unsafe_allow_html=True,
        )

    with trips_mix_cols[1]:
        st.markdown("<div class='fw-section-title'>Classpass Visits</div>", unsafe_allow_html=True)
        st.markdown(
            render_trips_card(
                month_classpass_to_date,
                f"Classpass: {month_start_ts:%b %d} – {actual_month_end:%b %d}",
                month_classpass_comp,
                f"Prior Year: {month_td_comp_start:%b %d} – {month_td_comp_end:%b %d}",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='fw-section-title'>Trips Breakdown</div>", unsafe_allow_html=True)
    trips_chart_cols = st.columns([1.25, 0.35])
    with trips_chart_cols[0]:
        trips_chart_current = build_chart_data(filtered_df, "Current", "Current", column="total_visits").sort_values("date")
        trips_chart_comparison = build_chart_data(comparison_df, "Comparison", "Comparison", column="total_visits").sort_values("date")
        if trips_chart_current.empty:
            st.info("Not enough data to display the snapshot chart.")
        else:
            trips_comparison_color = "#3f4a78"
            trips_current_color = "#cda643"
            trips_chart_current["x_axis"] = trips_chart_current["date"].dt.strftime("%b %d")
            trips_chart_current["display_label"] = trips_chart_current["date"].dt.strftime("%b %d, %Y")
            trips_chart_current["comparison_label"] = trips_chart_current["display_label"]

            trips_comparison_trimmed = trips_chart_comparison.head(len(trips_chart_current)).copy()
            trips_comparison_trimmed["x_axis"] = trips_chart_current["x_axis"].values[:len(trips_comparison_trimmed)]
            trips_comparison_trimmed["display_label"] = trips_chart_current["display_label"].values[:len(trips_comparison_trimmed)]
            trips_comparison_trimmed["comparison_label"] = trips_comparison_trimmed["date"].dt.strftime("%b %d, %Y")

            trips_chart_df = pd.concat([trips_chart_current, trips_comparison_trimmed], ignore_index=True)

            trips_current_range = f"{start_date:%b %d} – {end_date:%b %d, %Y}"
            trips_comparison_range = f"{comp_start_date:%b %d} – {comp_end_date:%b %d, %Y}"
            trips_current_delta = sales_card_delta(range_trips, comparison_trips)
            trips_comparison_delta = sales_card_delta(comparison_trips, range_trips)
            trips_header_html = (
                "<div class='sales-bar-container legend-dual'>"
                "<div class='legend-row'>"
                f"<span class='legend-entry'><span class='legend-swatch' style='background:{trips_current_color};'></span><span class='legend-label'>Current</span><span class='legend-value' style='color:{trips_current_color};'>{format_number(range_trips_display, 0)}</span><span class='legend-period'>{trips_current_range}</span><span class='legend-delta'>{trips_current_delta}</span></span>"
                f"<span class='legend-entry'><span class='legend-swatch' style='background:{trips_comparison_color};'></span><span class='legend-label'>Comparison</span><span class='legend-value' style='color:{trips_comparison_color};'>{format_number(comparison_trips, 0)}</span><span class='legend-period'>{trips_comparison_range}</span><span class='legend-delta'>{trips_comparison_delta}</span></span>"
                "</div>"
                "</div>"
            )
            st.markdown(trips_header_html, unsafe_allow_html=True)

            trips_chart = (
                alt.Chart(trips_chart_df)
                .mark_bar(width=14, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("x_axis:N", title="", axis=alt.Axis(labelColor="#aeb3d1", labelPadding=8, labelAngle=0)),
                    xOffset="series:N",
                    y=alt.Y("netsales:Q", title="Trips", axis=alt.Axis(labelColor="#aeb3d1")),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(range=[trips_comparison_color, trips_current_color], domain=["Comparison", "Current"]),
                        title="",
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("series:N", title="Series"),
                        alt.Tooltip("display_label:N", title="Current Date"),
                        alt.Tooltip("netsales:Q", title="Trips", format=","),
                        alt.Tooltip("comparison_label:N", title="Comparison Date"),
                    ],
                )
                .properties(width=1187, height=240)
            )
            st.altair_chart(trips_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    trips_cols = st.columns(2)
    trips_summary_df = cast(pd.DataFrame, studio_df.copy())
    trips_summary_df["date"] = pd.to_datetime(trips_summary_df["date"], errors="coerce")
    trips_summary_df["week_start"] = trips_summary_df["date"].dt.to_period("W-SUN").dt.start_time

    trips_comparison_summary_df = cast(pd.DataFrame, comparison_df.copy())
    trips_comparison_summary_df["date"] = pd.to_datetime(trips_comparison_summary_df["date"], errors="coerce")
    trips_comparison_summary_df["week_start"] = trips_comparison_summary_df["date"].dt.to_period("W-SUN").dt.start_time
    trips_comparison_daily_totals = trips_comparison_summary_df.groupby("date")["total_visits"].sum().sort_index(ascending=False)
    trips_comparison_weekly_totals = trips_comparison_summary_df.groupby("week_start")["total_visits"].sum().sort_index(ascending=False)

    with trips_cols[0]:
        st.markdown("<div class='fw-section-title'>Daily Trips</div>", unsafe_allow_html=True)
        trips_daily_totals = trips_summary_df.groupby("date")["total_visits"].sum().sort_index(ascending=False)
        trips_daily_rows = trips_daily_totals.head(6).reset_index()
        trips_daily_html_parts = []
        trips_comparison_daily_rows = trips_comparison_daily_totals.head(len(trips_daily_rows)).reset_index()
        for idx, row in enumerate(trips_daily_rows.to_dict("records")):
            day = cast(pd.Timestamp, pd.Timestamp(row["date"]))
            day_value = float(row["total_visits"])
            comp_label = "—"
            comp_value: Optional[float] = None
            if idx < len(trips_comparison_daily_rows):
                comp_row = trips_comparison_daily_rows.iloc[idx]
                comp_day = cast(pd.Timestamp, pd.Timestamp(comp_row["date"]))
                comp_value = float(comp_row["total_visits"])
                comp_label = comp_day.strftime("%b %d, %Y")
            trips_daily_html_parts.append(
                render_trips_entry_card(
                    day.strftime("%b %d, %Y"),
                    day_value,
                    comp_label,
                    comp_value,
                )
            )
        st.markdown("".join(trips_daily_html_parts), unsafe_allow_html=True)

    with trips_cols[1]:
        st.markdown("<div class='fw-section-title'>Weekly Trips</div>", unsafe_allow_html=True)
        trips_weekly_totals = trips_summary_df.groupby("week_start")["total_visits"].sum().sort_index(ascending=False)
        trips_weekly_rows = trips_weekly_totals.head(6).reset_index()
        trips_weekly_html_parts = []
        trips_comparison_weekly_rows = trips_comparison_weekly_totals.head(len(trips_weekly_rows)).reset_index()
        for idx, row in enumerate(trips_weekly_rows.to_dict("records")):
            week_start = cast(pd.Timestamp, pd.Timestamp(row["week_start"]))
            week_end = week_start + pd.Timedelta(days=6)
            week_value = float(row["total_visits"])
            comp_label = "—"
            comp_value: Optional[float] = None
            if idx < len(trips_comparison_weekly_rows):
                comp_row = trips_comparison_weekly_rows.iloc[idx]
                comp_week = cast(pd.Timestamp, pd.Timestamp(comp_row["week_start"]))
                comp_week_end = comp_week + pd.Timedelta(days=6)
                comp_value = float(comp_row["total_visits"])
                comp_label = f"{comp_week:%b %d} - {comp_week_end:%b %d, %Y}"
            trips_weekly_html_parts.append(
                render_trips_entry_card(
                    f"{week_start:%b %d} - {week_end:%b %d, %Y}",
                    week_value,
                    comp_label,
                    comp_value,
                )
            )
        st.markdown("".join(trips_weekly_html_parts), unsafe_allow_html=True)
