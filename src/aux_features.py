from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


def _clean_name(value: object) -> str:
    text = str(value).lower().strip()
    for old, new in [
        (" ", "_"),
        ("-", "_"),
        ("/", "_"),
        ("&", "and"),
        ("#", ""),
    ]:
        text = text.replace(old, new)
    return "".join(ch for ch in text if ch.isalnum() or ch == "_")


def _mean_or_zero(series: pd.Series) -> pd.Series:
    return series.fillna(0.0)


@lru_cache(maxsize=1)
def build_aux_daily(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Build one daily dataframe from all non-sales source tables."""
    data_dir = Path(data_dir)

    sales = pd.read_csv(data_dir / "sales.csv", parse_dates=["Date"])
    orders = pd.read_csv(data_dir / "orders.csv", parse_dates=["order_date"])
    order_items = pd.read_csv(data_dir / "order_items.csv", low_memory=False)
    products = pd.read_csv(data_dir / "products.csv")
    customers = pd.read_csv(data_dir / "customers.csv", parse_dates=["signup_date"])
    geography = pd.read_csv(data_dir / "geography.csv")
    payments = pd.read_csv(data_dir / "payments.csv")
    shipments = pd.read_csv(
        data_dir / "shipments.csv", parse_dates=["ship_date", "delivery_date"]
    )
    returns = pd.read_csv(data_dir / "returns.csv", parse_dates=["return_date"])
    reviews = pd.read_csv(data_dir / "reviews.csv", parse_dates=["review_date"])
    traffic = pd.read_csv(data_dir / "web_traffic.csv", parse_dates=["date"])
    promos = pd.read_csv(
        data_dir / "promotions.csv", parse_dates=["start_date", "end_date"]
    )
    inventory = pd.read_csv(data_dir / "inventory.csv", parse_dates=["snapshot_date"])

    idx = pd.DatetimeIndex(sales.sort_values("Date").Date, name="Date")
    aux = pd.DataFrame(index=idx)

    orders2 = orders.merge(
        customers[
            [
                "customer_id",
                "signup_date",
                "gender",
                "age_group",
                "acquisition_channel",
            ]
        ],
        on="customer_id",
        how="left",
    ).merge(geography[["zip", "region"]], on="zip", how="left")
    orders2["new_customer_order"] = (
        orders2.signup_date.dt.normalize() == orders2.order_date.dt.normalize()
    ).astype(int)
    orders2["returned_status"] = (orders2.order_status == "returned").astype(int)
    orders2["delivered_status"] = (orders2.order_status == "delivered").astype(int)
    orders2["mobile_device"] = (orders2.device_type == "mobile").astype(int)
    orders2["desktop_device"] = (orders2.device_type == "desktop").astype(int)
    orders2["credit_card_payment"] = (orders2.payment_method == "credit_card").astype(
        int
    )
    orders2["cod_payment"] = (orders2.payment_method == "cod").astype(int)
    for source in ["paid_search", "organic_search", "email", "social_media"]:
        orders2[f"source_{source}"] = (orders2.order_source == source).astype(int)

    order_agg = orders2.groupby("order_date").agg(
        order_count=("order_id", "size"),
        unique_customers=("customer_id", "nunique"),
        new_customer_order_share=("new_customer_order", "mean"),
        returned_status_share=("returned_status", "mean"),
        delivered_status_share=("delivered_status", "mean"),
        mobile_share=("mobile_device", "mean"),
        desktop_share=("desktop_device", "mean"),
        credit_card_share=("credit_card_payment", "mean"),
        cod_share=("cod_payment", "mean"),
        paid_search_share=("source_paid_search", "mean"),
        organic_search_share=("source_organic_search", "mean"),
        email_share=("source_email", "mean"),
        social_media_share=("source_social_media", "mean"),
    )
    aux = aux.join(order_agg, how="left")

    items = order_items.merge(
        orders[["order_id", "order_date", "zip"]], on="order_id", how="left"
    ).merge(
        products[["product_id", "category", "segment", "price", "cogs"]],
        on="product_id",
        how="left",
    )
    items["gross_sales"] = items.quantity * items.unit_price
    items["net_item_sales"] = items.gross_sales - items.discount_amount
    items["discount_rate"] = np.where(
        items.gross_sales > 0, items.discount_amount / items.gross_sales, 0.0
    )
    items["promo_item"] = (items.promo_id.notna() | items.promo_id_2.notna()).astype(int)
    items["product_margin_rate"] = np.where(
        items.price > 0, (items.price - items.cogs) / items.price, np.nan
    )
    item_agg = items.groupby("order_date").agg(
        units=("quantity", "sum"),
        gross_sales=("gross_sales", "sum"),
        net_item_sales=("net_item_sales", "sum"),
        discount_amount=("discount_amount", "sum"),
        avg_unit_price=("unit_price", "mean"),
        avg_discount_rate=("discount_rate", "mean"),
        promo_item_share=("promo_item", "mean"),
        distinct_products=("product_id", "nunique"),
        avg_product_margin_rate=("product_margin_rate", "mean"),
    )
    aux = aux.join(item_agg, how="left")

    top_categories = (
        items.groupby("category").net_item_sales.sum().sort_values(ascending=False)
        .head(6)
        .index
    )
    cat_daily = items[items.category.isin(top_categories)].pivot_table(
        index="order_date",
        columns="category",
        values="net_item_sales",
        aggfunc="sum",
    )
    cat_daily = cat_daily.div(items.groupby("order_date").net_item_sales.sum(), axis=0)
    cat_daily.columns = [f"cat_share_{_clean_name(c)}" for c in cat_daily.columns]
    aux = aux.join(cat_daily, how="left")

    items_geo = items.merge(geography[["zip", "region"]], on="zip", how="left")
    region_daily = items_geo.pivot_table(
        index="order_date",
        columns="region",
        values="net_item_sales",
        aggfunc="sum",
    )
    region_daily = region_daily.div(
        items_geo.groupby("order_date").net_item_sales.sum(), axis=0
    )
    region_daily.columns = [
        f"region_share_{_clean_name(c)}" for c in region_daily.columns
    ]
    aux = aux.join(region_daily, how="left")

    pay = payments.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    pay_agg = pay.groupby("order_date").agg(
        payment_value=("payment_value", "sum"),
        avg_installments=("installments", "mean"),
    )
    aux = aux.join(pay_agg, how="left")

    ship = shipments.copy()
    ship["shipping_days"] = (ship.delivery_date - ship.ship_date).dt.days
    ship_agg = ship.groupby("ship_date").agg(
        shipments=("order_id", "size"),
        shipping_fee=("shipping_fee", "sum"),
        ship_days_mean=("shipping_days", "mean"),
    )
    aux = aux.join(ship_agg, how="left")

    ret_agg = returns.groupby("return_date").agg(
        return_rows=("order_id", "size"),
        return_qty=("return_quantity", "sum"),
        refund_amount=("refund_amount", "sum"),
    )
    aux = aux.join(ret_agg, how="left")

    rev_agg = reviews.groupby("review_date").agg(
        review_rows=("order_id", "size"),
        avg_rating=("rating", "mean"),
    )
    aux = aux.join(rev_agg, how="left")

    traffic_agg = traffic.groupby("date").agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_session_duration_sec=("avg_session_duration_sec", "mean"),
    )
    aux = aux.join(traffic_agg, how="left")

    promo_records = []
    for _, row in promos.iterrows():
        for date in pd.date_range(row.start_date, row.end_date, freq="D"):
            promo_records.append(
                (
                    date,
                    row.discount_value,
                    row.stackable_flag,
                    row.min_order_value,
                )
            )
    promo_daily = pd.DataFrame(
        promo_records,
        columns=[
            "Date",
            "discount_value",
            "stackable_flag",
            "min_order_value",
        ],
    )
    if len(promo_daily):
        promo_agg = promo_daily.groupby("Date").agg(
            active_promos=("discount_value", "size"),
            avg_promo_discount_value=("discount_value", "mean"),
            any_stackable_promo=("stackable_flag", "max"),
            avg_promo_min_order_value=("min_order_value", "mean"),
        )
        aux = aux.join(promo_agg, how="left")

    inv_agg = inventory.groupby("snapshot_date").agg(
        stock_on_hand=("stock_on_hand", "sum"),
        units_received_inv=("units_received", "sum"),
        units_sold_inv=("units_sold", "sum"),
        stockout_days=("stockout_days", "sum"),
        mean_days_of_supply=("days_of_supply", "mean"),
        mean_fill_rate=("fill_rate", "mean"),
        stockout_ratio=("stockout_flag", "mean"),
        overstock_ratio=("overstock_flag", "mean"),
        reorder_ratio=("reorder_flag", "mean"),
        mean_sell_through_rate=("sell_through_rate", "mean"),
    )
    aux = aux.join(inv_agg, how="left")
    aux[list(inv_agg.columns)] = aux[list(inv_agg.columns)].ffill()

    zero_cols = [
        "order_count",
        "unique_customers",
        "units",
        "gross_sales",
        "net_item_sales",
        "discount_amount",
        "distinct_products",
        "payment_value",
        "shipments",
        "shipping_fee",
        "return_rows",
        "return_qty",
        "refund_amount",
        "review_rows",
        "sessions",
        "unique_visitors",
        "page_views",
        "active_promos",
    ]
    for col in [c for c in zero_cols if c in aux.columns]:
        aux[col] = aux[col].fillna(0.0)

    share_cols = [
        c
        for c in aux.columns
        if c.endswith("_share") or c.startswith("cat_share_") or c.startswith("region_share_")
    ]
    for col in share_cols:
        aux[col] = aux[col].fillna(0.0)

    for col in aux.columns:
        if aux[col].isna().any() and col not in inv_agg.columns:
            aux[col] = aux[col].fillna(aux[col].median())

    aux["avg_order_value"] = aux.net_item_sales / np.maximum(aux.order_count, 1.0)
    aux["units_per_order"] = aux.units / np.maximum(aux.order_count, 1.0)
    aux["sessions_per_order"] = aux.sessions / np.maximum(aux.order_count, 1.0)
    aux["conversion_order_per_session"] = aux.order_count / np.maximum(aux.sessions, 1.0)
    aux["refund_per_order"] = aux.refund_amount / np.maximum(aux.order_count, 1.0)
    aux["return_qty_per_order"] = aux.return_qty / np.maximum(aux.order_count, 1.0)

    return aux.sort_index()


def aux_feature_groups(aux_daily: pd.DataFrame | None = None) -> dict[str, list[str]]:
    aux = build_aux_daily() if aux_daily is None else aux_daily
    volume = [
        "order_count",
        "unique_customers",
        "units",
        "gross_sales",
        "net_item_sales",
        "payment_value",
        "sessions",
        "page_views",
        "shipments",
        "return_qty",
        "refund_amount",
    ]
    marketing_web = [
        "avg_discount_rate",
        "promo_item_share",
        "active_promos",
        "avg_promo_discount_value",
        "paid_search_share",
        "organic_search_share",
        "email_share",
        "social_media_share",
        "bounce_rate",
        "avg_session_duration_sec",
        "conversion_order_per_session",
    ]
    mix_geo_product = [
        "avg_order_value",
        "avg_unit_price",
        "units_per_order",
        "avg_product_margin_rate",
        "mobile_share",
        "desktop_share",
        "credit_card_share",
        "cod_share",
        "avg_installments",
        *[c for c in aux.columns if c.startswith("cat_share_")],
        *[c for c in aux.columns if c.startswith("region_share_")],
    ]
    ops_quality_inventory = [
        "returned_status_share",
        "delivered_status_share",
        "ship_days_mean",
        "shipping_fee",
        "return_qty_per_order",
        "refund_per_order",
        "review_rows",
        "avg_rating",
        "stockout_ratio",
        "overstock_ratio",
        "reorder_ratio",
        "mean_fill_rate",
        "mean_sell_through_rate",
        "mean_days_of_supply",
    ]
    groups = {
        "volume": volume,
        "marketing_web": marketing_web,
        "mix_geo_product": mix_geo_product,
        "ops_quality_inventory": ops_quality_inventory,
    }
    groups["all_aux"] = [
        c
        for group_cols in groups.values()
        for c in group_cols
    ]
    return {
        name: [c for c in cols if c in aux.columns and aux[c].notna().any()]
        for name, cols in groups.items()
    }


def build_aux_feature_matrix(
    target_dates: pd.Series,
    as_of: pd.Timestamp,
    columns: list[str],
    *,
    aux_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build leakage-safe priors from auxiliary daily aggregates."""
    if not columns:
        return pd.DataFrame(index=range(len(target_dates)))

    aux = build_aux_daily() if aux_daily is None else aux_daily
    dates = pd.to_datetime(target_dates)
    hist = aux.loc[:as_of].copy()
    hist_cal = hist.reset_index()
    hist_cal["year"] = hist_cal.Date.dt.year
    hist_cal["month"] = hist_cal.Date.dt.month
    hist_cal["dow"] = hist_cal.Date.dt.dayofweek
    hist_cal["doy"] = hist_cal.Date.dt.dayofyear
    hist_cal["week"] = hist_cal.Date.dt.isocalendar().week.astype(int)

    base = hist_cal[hist_cal.year >= 2019]
    if len(base) < 500:
        base = hist_cal

    cal = pd.DataFrame({"Date": dates})
    cal["month"] = cal.Date.dt.month
    cal["dow"] = cal.Date.dt.dayofweek
    cal["doy"] = cal.Date.dt.dayofyear
    cal["week"] = cal.Date.dt.isocalendar().week.astype(int)

    parts: list[pd.Series] = []
    for col in columns:
        s = hist[col]
        month_mean = base.groupby("month")[col].mean()
        month_dow_mean = base.groupby(["month", "dow"])[col].mean()
        doy_mean = base.groupby("doy")[col].mean()
        week_mean = base.groupby("week")[col].mean()
        roll30 = s.rolling(30, min_periods=10).mean()

        month = cal.month.map(month_mean)
        month_dow = pd.Series(
            [month_dow_mean.get((m, d), np.nan) for m, d in zip(cal.month, cal.dow)],
            index=cal.index,
        ).fillna(month)
        doy = cal.doy.map(doy_mean)
        week = cal.week.map(week_mean)
        lag548 = pd.Series(s.reindex(dates - pd.Timedelta(days=548)).values)
        lag730 = pd.Series(s.reindex(dates - pd.Timedelta(days=730)).values)
        year1 = pd.Series(s.reindex(dates - pd.DateOffset(years=1)).values)
        year2 = pd.Series(s.reindex(dates - pd.DateOffset(years=2)).values)
        roll30_548 = pd.Series(
            roll30.reindex(dates - pd.Timedelta(days=548)).values
        )
        roll30_730 = pd.Series(
            roll30.reindex(dates - pd.Timedelta(days=730)).values
        )

        for suffix, values in [
            ("month_mean", month),
            ("month_dow_mean", month_dow),
            ("doy_mean", doy),
            ("week_mean", week),
            ("lag548", lag548),
            ("lag730", lag730),
            ("year1", year1),
            ("year2", year2),
            ("roll30_548", roll30_548),
            ("roll30_730", roll30_730),
        ]:
            values = pd.Series(values).reset_index(drop=True)
            values.name = f"aux_{col}_{suffix}"
            parts.append(values)

    return pd.concat(parts, axis=1).reset_index(drop=True)
