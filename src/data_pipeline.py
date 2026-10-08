from pathlib import Path
import numpy as np
import pandas as pd


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Parse mixed date formats robustly without relying on one fixed format."""
    return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)


def money_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
        errors="coerce",
    )


def clean_and_integrate(clients_path: str | Path, properties_path: str | Path) -> pd.DataFrame:
    clients = pd.read_csv(clients_path)
    properties = pd.read_csv(properties_path)

    clients = clients.drop_duplicates().copy()
    properties = properties.drop_duplicates().copy()

    # Normalize categorical text.
    categorical_client = [
        "client_type", "gender", "country", "region",
        "acquisition_purpose", "loan_applied", "referral_channel"
    ]
    for col in categorical_client:
        if col in clients.columns:
            clients[col] = clients[col].astype(str).str.strip()

    for col in ["unit_category", "listing_status"]:
        if col in properties.columns:
            properties[col] = properties[col].astype(str).str.strip()

    clients["date_of_birth"] = parse_mixed_dates(clients["date_of_birth"])
    properties["transaction_date"] = parse_mixed_dates(properties["transaction_date"])
    properties["sale_price_numeric"] = money_to_numeric(properties["sale_price"])

    reference_date = properties["transaction_date"].max()
    if pd.isna(reference_date):
        reference_date = pd.Timestamp.today().normalize()
    clients["age"] = ((reference_date - clients["date_of_birth"]).dt.days / 365.25).round(1)

    sold = properties[properties["listing_status"].str.lower().eq("sold")].copy()
    sold = sold[sold["client_ref"].notna()].copy()

    agg = sold.groupby("client_ref", dropna=False).agg(
        properties_purchased=("listing_id", "count"),
        total_investment=("sale_price_numeric", "sum"),
        average_property_price=("sale_price_numeric", "mean"),
        total_floor_area_sqft=("floor_area_sqft", "sum"),
        average_floor_area_sqft=("floor_area_sqft", "mean"),
        first_purchase_date=("transaction_date", "min"),
        last_purchase_date=("transaction_date", "max"),
    ).reset_index().rename(columns={"client_ref": "client_id"})

    agg["purchase_span_days"] = (
        agg["last_purchase_date"] - agg["first_purchase_date"]
    ).dt.days.fillna(0)

    result = clients.merge(agg, on="client_id", how="left")
    numeric_purchase = [
        "properties_purchased", "total_investment", "average_property_price",
        "total_floor_area_sqft", "average_floor_area_sqft", "purchase_span_days"
    ]
    for col in numeric_purchase:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    result["has_property_purchase"] = (result["properties_purchased"] > 0).astype(int)

    return result
