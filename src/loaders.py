"""Data loaders for VIN Datathon 2026."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_all_tables() -> dict[str, pd.DataFrame]:
    """Load all 15 CSV files, parse dates, set dtypes. Returns dict keyed by table name."""
    tables: dict[str, pd.DataFrame] = {}

    tables["sales"] = pd.read_csv(
        DATA_DIR / "sales.csv",
        parse_dates=["Date"],
    )

    tables["customers"] = pd.read_csv(
        DATA_DIR / "customers.csv",
        parse_dates=["signup_date"],
        dtype={"customer_id": "int32", "zip": "int32"},
    )

    tables["orders"] = pd.read_csv(
        DATA_DIR / "orders.csv",
        parse_dates=["order_date"],
        dtype={"order_id": "int32", "customer_id": "int32", "zip": "int32"},
    )

    tables["order_items"] = pd.read_csv(
        DATA_DIR / "order_items.csv",
        dtype={"order_id": "int32", "product_id": "int32", "promo_id_2": "str"},
        low_memory=False,
    )

    tables["products"] = pd.read_csv(
        DATA_DIR / "products.csv",
        dtype={"product_id": "int32"},
    )

    tables["promotions"] = pd.read_csv(
        DATA_DIR / "promotions.csv",
        parse_dates=["start_date", "end_date"],
    )

    tables["shipments"] = pd.read_csv(
        DATA_DIR / "shipments.csv",
        parse_dates=["ship_date", "delivery_date"],
        dtype={"order_id": "int32"},
    )

    tables["returns"] = pd.read_csv(
        DATA_DIR / "returns.csv",
        parse_dates=["return_date"],
        dtype={"order_id": "int32", "product_id": "int32"},
    )

    tables["reviews"] = pd.read_csv(
        DATA_DIR / "reviews.csv",
        parse_dates=["review_date"],
        dtype={"order_id": "int32", "product_id": "int32", "customer_id": "int32"},
    )

    tables["payments"] = pd.read_csv(
        DATA_DIR / "payments.csv",
        dtype={"order_id": "int32"},
    )

    tables["geography"] = pd.read_csv(
        DATA_DIR / "geography.csv",
        dtype={"zip": "int32"},
    )

    tables["inventory"] = pd.read_csv(
        DATA_DIR / "inventory.csv",
        parse_dates=["snapshot_date"],
        dtype={"product_id": "int32"},
    )

    tables["web_traffic"] = pd.read_csv(
        DATA_DIR / "web_traffic.csv",
        parse_dates=["date"],
    )

    return tables
