"""
NBA Score Dashboard – Fashion E-Commerce
Next Best Offer Algorithm mit Streamlit

Datenstruktur:
  users.csv          id, first_name, last_name, email, age, gender, state,
                     street_address, postal_code, city, country,
                     latitude, longitude, traffic_source, created_at
  orders.csv         order_id, user_id, status, gender, created_at,
                     returned_at, shipped_at, delivered_at, num_of_item
  order_items.csv    id, order_id, user_id, product_id, inventory_item_id,
                     status, created_at, shipped_at, delivered_at,
                     returned_at, sale_price
  products.csv       id, cost, category, name, brand, retail_price,
                     department, sku, distribution_center_id
  inventory_items.csv id, product_id, created_at, sold_at, cost,
                      product_category, product_name, product_brand,
                      product_retail_price, product_department,
                      product_sku, product_distribution_center_id
  events.csv         id, user_id, sequence_number, session_id, created_at,
                     ip_address, city, state, postal_code, browser,
                     traffic_source, uri, event_type
"""

import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import gdown

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Score Dashboard | Fashion Store",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
div[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 12px 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────────────
DRIVE_IDS: dict[str, str] = {
    "users": "12NzZrANo4mCIoze9qKpUxgsC3z3DMd6-",
    "orders": "14WB16ZOh8u9zA3K7OPudLzMmWkSTDHYk",
    "order_items": "1p1DVgvu-1BMnYY3ZZ92P7S62dOAam4Sb",
    "products": "1QsgPtrsO9FJgVmHnaiN12qziUAK3_GjB",
    "inventory_items": "1S19ZWsDXtlSV5SH7BAWqNgPYzR_aFxTO",
    "events": "1cV2o0uYLfFdD8H1NVkHoJ-GAh0WKUXdh",
}

EVENTS_SAMPLE_ROWS = 400_000

NBA_WEIGHTS = {
    "relevanz": 0.35,
    "conv_signal": 0.30,
    "marge": 0.20,
    "popularitaet": 0.15,
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ── Data Loading ─────────────────────────────────────────────────────────────
def _download_if_missing(name: str) -> Path:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        url = f"https://drive.google.com/uc?id={DRIVE_IDS[name]}"
        gdown.download(url, str(path), quiet=True)
    return path


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    files = list(DRIVE_IDS.keys())
    progress = st.progress(0, text="Lade Daten von Google Drive …")
    dfs: dict[str, pd.DataFrame] = {}

    for i, name in enumerate(files):
        progress.progress(i / len(files), text=f"Lade {name}.csv …")
        path = _download_if_missing(name)
        if name == "events":
            dfs[name] = pd.read_csv(path, nrows=EVENTS_SAMPLE_ROWS)
        else:
            dfs[name] = pd.read_csv(path)

    progress.progress(1.0, text="✓ Alle Dateien geladen")
    progress.empty()
    return dfs


# ── NBA Score Engine ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_product_metrics(
    _products: pd.DataFrame,
    _order_items: pd.DataFrame,
    _events: pd.DataFrame,
) -> pd.DataFrame:
    prods = _products.copy()

    # Marge
    prods["raw_margin"] = (
        (prods["retail_price"] - prods["cost"]) / prods["retail_price"].clip(lower=0.01)
    ).clip(0, 1)
    lo, hi = prods["raw_margin"].min(), prods["raw_margin"].max()
    prods["marge"] = (prods["raw_margin"] - lo) / (hi - lo + 1e-9)

    # Popularität
    oi = _order_items.copy()
    completed = (
        oi[oi["status"] == "Complete"]
        .groupby("product_id")
        .size()
        .rename("complete_cnt")
    )
    returned = (
        oi[oi["status"] == "Returned"].groupby("product_id").size().rename("return_cnt")
    )
    total_cnt = oi.groupby("product_id").size().rename("total_cnt")

    stats = pd.concat([total_cnt, completed, returned], axis=1).fillna(0)
    stats["return_rate"] = stats["return_cnt"] / stats["total_cnt"].clip(lower=1)
    stats["raw_pop"] = stats["complete_cnt"] * (1 - stats["return_rate"])
    stats["popularitaet"] = stats["raw_pop"] / (stats["raw_pop"].max() + 1e-9)

    prods = prods.merge(
        stats[["popularitaet", "return_rate"]],
        left_on="id",
        right_index=True,
        how="left",
    )
    prods["popularitaet"] = prods["popularitaet"].fillna(0)
    prods["return_rate"] = prods["return_rate"].fillna(0)

    # Conv Signal (aus Events)
    ev = _events.copy()
    if "uri" in ev.columns and "event_type" in ev.columns:
        ev["product_id_ev"] = (
            ev["uri"].str.extract(r"/products/(\d+)", expand=False).astype(float)
        )
        views = (
            ev[ev["event_type"] == "product"]
            .groupby("product_id_ev")
            .size()
            .rename("view_cnt")
        )
        purchases = (
            ev[ev["event_type"] == "purchase"]
            .groupby("product_id_ev")
            .size()
            .rename("ev_purchase_cnt")
        )
        conv_df = pd.concat([views, purchases], axis=1).fillna(0)
        conv_df["raw_conv"] = conv_df["ev_purchase_cnt"] / (conv_df["view_cnt"] + 1)
        max_conv = conv_df["raw_conv"].max()
        conv_df["conv_signal"] = conv_df["raw_conv"] / (max_conv + 1e-9)
        prods = prods.merge(
            conv_df[["conv_signal"]], left_on="id", right_index=True, how="left"
        )
    else:
        prods["conv_signal"] = 0.0

    median_conv = prods["conv_signal"].median()
    prods["conv_signal"] = prods["conv_signal"].fillna(median_conv)

    return prods.reset_index(drop=True)


def _user_relevanz(
    user_id: int,
    candidates: pd.DataFrame,
    order_items: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.Series:
    user_row = users[users["id"] == user_id]
    if user_row.empty:
        return pd.Series(0.5, index=candidates.index)

    gender = user_row["gender"].iloc[0]
    user_orders = order_items[order_items["user_id"] == user_id]

    dept_score = candidates["department"].apply(
        lambda d: (
            1.0
            if (gender == "F" and d == "Women") or (gender == "M" and d == "Men")
            else 0.35
        )
    )

    if user_orders.empty:
        return (0.4 * dept_score + 0.6 * 0.5).clip(0, 1)

    bought = candidates[candidates["id"].isin(user_orders["product_id"])]
    cat_dist = bought["category"].value_counts(normalize=True)
    cat_score = candidates["category"].map(cat_dist).fillna(0.0)

    known_brands = set(bought["brand"].dropna())
    brand_score = candidates["brand"].apply(
        lambda b: 1.0 if b in known_brands else 0.15
    )

    avg_price = (
        user_orders["sale_price"].mean()
        if "sale_price" in user_orders.columns
        else candidates["retail_price"].mean()
    )
    if avg_price > 0:
        price_score = (
            1 - ((candidates["retail_price"] - avg_price).abs() / (avg_price + 1e-9))
        ).clip(0, 1)
    else:
        price_score = pd.Series(0.5, index=candidates.index)

    return (
        0.40 * cat_score + 0.20 * brand_score + 0.25 * price_score + 0.15 * dept_score
    ).clip(0, 1)


def get_top_n(
    user_id: int,
    dfs: dict[str, pd.DataFrame],
    product_metrics: pd.DataFrame,
    top_n: int = 3,
    w: dict[str, float] | None = None,
) -> pd.DataFrame:
    if w is None:
        w = NBA_WEIGHTS

    oi = dfs["order_items"]
    users = dfs["users"]

    bought_ids = set(oi[oi["user_id"] == user_id]["product_id"].unique())
    candidates = (
        product_metrics[~product_metrics["id"].isin(bought_ids)]
        .copy()
        .reset_index(drop=True)
    )
    if candidates.empty:
        candidates = product_metrics.copy().reset_index(drop=True)

    candidates["relevanz"] = _user_relevanz(user_id, candidates, oi, users).values

    total_w = sum(w.values()) or 1.0
    candidates["nba_score"] = (
        (w["relevanz"] / total_w) * candidates["relevanz"]
        + (w["conv_signal"] / total_w) * candidates["conv_signal"]
        + (w["marge"] / total_w) * candidates["marge"]
        + (w["popularitaet"] / total_w) * candidates["popularitaet"]
    )

    cols = [
        "id",
        "name",
        "category",
        "brand",
        "department",
        "retail_price",
        "relevanz",
        "conv_signal",
        "marge",
        "popularitaet",
        "nba_score",
    ]
    return candidates.nlargest(top_n, "nba_score")[cols].reset_index(drop=True)


# ── Seite 1: Dataset-Übersicht ───────────────────────────────────────────────
def page_overview(dfs: dict[str, pd.DataFrame]) -> None:
    st.title("📊 Dataset-Übersicht")

    users = dfs["users"]
    orders = dfs["orders"]
    oi = dfs["order_items"]
    products = dfs["products"]
    events = dfs["events"]
    inv_items = dfs["inventory_items"]

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("👥 Users", f"{len(users):,}")
    k2.metric("📦 Orders", f"{len(orders):,}")
    k3.metric("🛍️ Order Items", f"{len(oi):,}")
    k4.metric("👕 Produkte", f"{products['id'].nunique():,}")
    k5.metric("📦 Lagerartikel", f"{len(inv_items):,}")
    k6.metric("📈 Events (Sample)", f"{len(events):,}")

    complete_pct = (orders["status"] == "Complete").mean() * 100
    return_pct = (oi["status"] == "Returned").mean() * 100
    avg_price = oi["sale_price"].mean() if "sale_price" in oi.columns else 0
    avg_items = orders["num_of_item"].mean() if "num_of_item" in orders.columns else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("✅ Completion Rate", f"{complete_pct:.1f}%")
    m2.metric("↩️ Return Rate", f"{return_pct:.1f}%")
    m3.metric("💵 Ø Sale Price", f"${avg_price:.2f}")
    m4.metric("📦 Ø Items/Order", f"{avg_items:.1f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Order Status")
        status_df = orders["status"].value_counts().reset_index()
        status_df.columns = ["Status", "Anzahl"]
        fig = px.pie(
            status_df,
            names="Status",
            values="Anzahl",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Altersverteilung der User")
        fig = px.histogram(
            users,
            x="age",
            nbins=30,
            color="gender",
            color_discrete_map={"M": "#636EFA", "F": "#EF553B"},
            barmode="overlay",
            opacity=0.72,
        )
        fig.update_layout(
            xaxis_title="Alter", yaxis_title="Anzahl User", legend_title="Geschlecht"
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 12 Produktkategorien")
        cat_df = products["category"].value_counts().head(12).reset_index()
        cat_df.columns = ["Kategorie", "Produkte"]
        fig = px.bar(
            cat_df,
            x="Produkte",
            y="Kategorie",
            orientation="h",
            color="Produkte",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Retail-Preisverteilung (99. Perzentil)")
        p99 = products["retail_price"].quantile(0.99)
        fig = px.histogram(
            products[products["retail_price"] <= p99],
            x="retail_price",
            nbins=60,
            color="department",
            color_discrete_map={"Men": "#636EFA", "Women": "#EF553B"},
            barmode="overlay",
            opacity=0.72,
        )
        fig.update_layout(
            xaxis_title="Preis ($)", yaxis_title="Anzahl", legend_title="Abteilung"
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Traffic Sources")
        src_df = users["traffic_source"].value_counts().reset_index()
        src_df.columns = ["Quelle", "User"]
        fig = px.bar(
            src_df,
            x="Quelle",
            y="User",
            color="Quelle",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Länder nach Userzahl")
        country_df = users["country"].value_counts().head(10).reset_index()
        country_df.columns = ["Land", "User"]
        fig = px.bar(
            country_df,
            x="User",
            y="Land",
            orientation="h",
            color="User",
            color_continuous_scale="Greens",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    if "event_type" in events.columns:
        st.subheader("Event-Typen im Sample")
        ev_df = events["event_type"].value_counts().reset_index()
        ev_df.columns = ["Event-Typ", "Anzahl"]
        fig = px.bar(
            ev_df,
            x="Event-Typ",
            y="Anzahl",
            color="Event-Typ",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ── Seite 2: NBA Score Berechnung ────────────────────────────────────────────
def page_nba_scores(
    dfs: dict[str, pd.DataFrame],
    product_metrics: pd.DataFrame,
) -> None:
    st.title("🎯 NBA Score Berechnung")

    st.markdown("""
> **NBA Score** `= 0.35 × Relevanz + 0.30 × ConvSignal + 0.20 × Marge + 0.15 × Popularität`
>
> Produkt-globale Komponenten (Relevanz = 0.5 als Baseline).
> Personalisierten Score → **Interaktiver Simulator**.
""")

    with st.expander("⚙️ Score-Gewichte anpassen", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        w_rel = c1.slider("Relevanz", 0.0, 1.0, 0.35, 0.05, key="ov_rel")
        w_conv = c2.slider("ConvSignal", 0.0, 1.0, 0.30, 0.05, key="ov_conv")
        w_mar = c3.slider("Marge", 0.0, 1.0, 0.20, 0.05, key="ov_mar")
        w_pop = c4.slider("Popularität", 0.0, 1.0, 0.15, 0.05, key="ov_pop")
        total_w = w_rel + w_conv + w_mar + w_pop
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"⚠️ Summe = {total_w:.2f}. Wird normalisiert.")

    total_w = (w_rel + w_conv + w_mar + w_pop) or 1.0

    pm = product_metrics.copy()
    pm["nba_base"] = (
        (w_rel / total_w) * 0.5
        + (w_conv / total_w) * pm["conv_signal"]
        + (w_mar / total_w) * pm["marge"]
        + (w_pop / total_w) * pm["popularitaet"]
    )

    st.divider()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Top 20 Produkte (NBA-Basis-Score)")
        top20 = pm.nlargest(20, "nba_base")[
            [
                "name",
                "category",
                "brand",
                "retail_price",
                "conv_signal",
                "marge",
                "popularitaet",
                "return_rate",
                "nba_base",
            ]
        ].rename(
            columns={
                "name": "Produkt",
                "category": "Kategorie",
                "brand": "Marke",
                "retail_price": "Preis ($)",
                "conv_signal": "ConvSignal",
                "marge": "Marge",
                "popularitaet": "Popularität",
                "return_rate": "Retourenquote",
                "nba_base": "NBA-Score",
            }
        )
        st.dataframe(
            top20.style.background_gradient(subset=["NBA-Score"], cmap="YlOrRd").format(
                {
                    "Preis ($)": "${:.2f}",
                    "ConvSignal": "{:.3f}",
                    "Marge": "{:.3f}",
                    "Popularität": "{:.3f}",
                    "Retourenquote": "{:.1%}",
                    "NBA-Score": "{:.3f}",
                }
            ),
            use_container_width=True,
            height=500,
        )

    with col2:
        st.subheader("Score-Verteilung")
        fig = px.histogram(
            pm,
            x="nba_base",
            nbins=50,
            color_discrete_sequence=["#636EFA"],
            labels={"nba_base": "NBA-Score"},
        )
        fig.update_layout(yaxis_title="Anzahl Produkte", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Komponenten-Korrelation")
        corr = pm[["conv_signal", "marge", "popularitaet", "nba_base"]].corr()
        fig2 = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        fig2.update_layout(margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📦 Ø Score-Komponenten nach Kategorie (Top 15)")
    cat_scores = (
        pm.groupby("category")[["conv_signal", "marge", "popularitaet", "nba_base"]]
        .mean()
        .reset_index()
        .sort_values("nba_base", ascending=False)
        .head(15)
    )
    fig = go.Figure()
    for comp, color, label in [
        ("conv_signal", "#EF553B", "ConvSignal"),
        ("marge", "#00CC96", "Marge"),
        ("popularitaet", "#AB63FA", "Popularität"),
    ]:
        fig.add_trace(
            go.Bar(
                name=label,
                x=cat_scores["category"],
                y=cat_scores[comp],
                marker_color=color,
            )
        )
    fig.update_layout(
        barmode="stack",
        xaxis_title="Kategorie",
        yaxis_title="Ø Score-Komponente",
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Marge vs. Popularität (Bubble: ConvSignal)")
    sample = pm.sample(min(500, len(pm)), random_state=42)
    fig = px.scatter(
        sample,
        x="marge",
        y="popularitaet",
        size="conv_signal",
        color="category",
        hover_data=["name", "brand", "retail_price"],
        labels={"marge": "Marge", "popularitaet": "Popularität"},
        opacity=0.7,
    )
    fig.update_layout(margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)


# ── Seite 3: Interaktiver Simulator ──────────────────────────────────────────
def page_simulator(
    dfs: dict[str, pd.DataFrame],
    product_metrics: pd.DataFrame,
) -> None:
    st.title("🔮 Interaktiver NBA Simulator")
    st.markdown("""
Gib eine **User-ID** ein – der Algorithmus berechnet die **Top-3 personalisierten Empfehlungen**
auf Basis von Kaufhistorie, Demographie und Produkteigenschaften.
""")

    users = dfs["users"]
    oi = dfs["order_items"]

    col_input, col_profile = st.columns([1, 2])

    with col_input:
        all_ids = sorted(users["id"].unique().tolist())
        min_id, max_id = int(all_ids[0]), int(all_ids[-1])

        user_id = st.number_input(
            "User-ID", min_value=min_id, max_value=max_id, value=int(all_ids[0]), step=1
        )

        if st.button("🎲 Zufälliger User", use_container_width=True):
            st.session_state["sim_user_id"] = int(np.random.choice(all_ids))

        if "sim_user_id" in st.session_state:
            user_id = st.session_state["sim_user_id"]
            st.info(f"Zufälliger User: **{user_id}**")

        calc_btn = st.button(
            "🚀 Empfehlungen berechnen", type="primary", use_container_width=True
        )

        st.divider()
        st.markdown("**Gewichte anpassen:**")
        w_rel = st.slider("Relevanz", 0.0, 1.0, 0.35, 0.05, key="sim_rel")
        w_conv = st.slider("ConvSignal", 0.0, 1.0, 0.30, 0.05, key="sim_conv")
        w_mar = st.slider("Marge", 0.0, 1.0, 0.20, 0.05, key="sim_mar")
        w_pop = st.slider("Popularität", 0.0, 1.0, 0.15, 0.05, key="sim_pop")
        custom_w = {
            "relevanz": w_rel,
            "conv_signal": w_conv,
            "marge": w_mar,
            "popularitaet": w_pop,
        }

    with col_profile:
        u_row = users[users["id"] == user_id]
        if not u_row.empty:
            u = u_row.iloc[0]
            st.subheader(f"👤 {u['first_name']} {u['last_name']} (ID {user_id})")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Geschlecht", u["gender"])
            m2.metric("Alter", int(u["age"]))
            m3.metric("Land", u["country"])
            m4.metric("Traffic Source", u["traffic_source"])

            user_orders = oi[oi["user_id"] == user_id]
            completed = user_orders[user_orders["status"] == "Complete"]
            returned = user_orders[user_orders["status"] == "Returned"]

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Bestellungen", len(user_orders))
            m6.metric("Abgeschlossen", len(completed))
            m7.metric("Retouren", len(returned))
            spend = (
                completed["sale_price"].sum()
                if "sale_price" in completed.columns
                else 0
            )
            m8.metric("Umsatz (Compl.)", f"${spend:.2f}")

            if not user_orders.empty:
                bought_cats = (
                    user_orders.merge(
                        product_metrics[["id", "category"]],
                        left_on="product_id",
                        right_on="id",
                        how="left",
                    )["category"]
                    .value_counts()
                    .head(5)
                )
                if not bought_cats.empty:
                    fig = px.bar(
                        bought_cats.reset_index().rename(
                            columns={"index": "Kategorie", "category": "Käufe"}
                        ),
                        x="Käufe",
                        y="Kategorie",
                        orientation="h",
                        color_discrete_sequence=["#636EFA"],
                    )
                    fig.update_layout(
                        title="Top Kategorien (Kaufhistorie)",
                        yaxis={"categoryorder": "total ascending"},
                        margin=dict(t=30, b=0),
                        height=200,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    if calc_btn:
        with st.spinner("Berechne NBA-Scores …"):
            recs = get_top_n(user_id, dfs, product_metrics, top_n=3, w=custom_w)
        st.session_state["sim_recs"] = recs
        st.session_state["sim_recs_uid"] = user_id

    recs = st.session_state.get("sim_recs", pd.DataFrame())
    rec_uid = st.session_state.get("sim_recs_uid", -1)

    if not recs.empty and rec_uid == user_id:
        st.subheader(f"🏆 Top-3 Empfehlungen für User {user_id}")

        medals = ["🥇", "🥈", "🥉"]
        borders = ["#FFD700", "#C0C0C0", "#CD7F32"]
        bg_cols = ["#FFFBEA", "#F8F8F8", "#FFF5EE"]

        cols = st.columns(3)
        for rank, (col, (_, row)) in enumerate(zip(cols, recs.iterrows())):
            with col:
                st.markdown(
                    f"""<div style="border-left:5px solid {borders[rank]};
                        background:{bg_cols[rank]}; padding:12px; border-radius:8px;">
                        <h3 style="margin:0 0 4px">{medals[rank]} Platz {rank + 1}</h3>
                        </div>""",
                    unsafe_allow_html=True,
                )
                name_str = str(row["name"])
                st.markdown(
                    f"**{name_str[:55] + '…' if len(name_str) > 55 else name_str}**"
                )
                st.write(f"🏷️ `{row['category']}` | 🏢 `{row['brand']}`")
                st.write(f"👗 `{row['department']}` | 💵 `${row['retail_price']:.2f}`")
                st.divider()

                total_w = sum(custom_w.values()) or 1.0
                score_parts = {
                    "Relevanz": row["relevanz"] * custom_w["relevanz"] / total_w,
                    "ConvSignal": row["conv_signal"]
                    * custom_w["conv_signal"]
                    / total_w,
                    "Marge": row["marge"] * custom_w["marge"] / total_w,
                    "Popularität": row["popularitaet"]
                    * custom_w["popularitaet"]
                    / total_w,
                }
                fig = go.Figure(
                    go.Bar(
                        x=list(score_parts.values()),
                        y=list(score_parts.keys()),
                        orientation="h",
                        marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"],
                        text=[f"{v:.3f}" for v in score_parts.values()],
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    title=f"NBA-Score: {row['nba_score']:.3f}",
                    height=220,
                    margin=dict(l=0, r=5, t=30, b=0),
                    xaxis=dict(range=[0, 0.4]),
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Score-Tabelle im Detail")
        detail = recs.rename(
            columns={
                "name": "Produkt",
                "category": "Kategorie",
                "brand": "Marke",
                "department": "Abteilung",
                "retail_price": "Preis ($)",
                "relevanz": "Relevanz",
                "conv_signal": "ConvSignal",
                "marge": "Marge",
                "popularitaet": "Popularität",
                "nba_score": "NBA-Score",
            }
        )
        st.dataframe(
            detail.drop(columns=["id"])
            .style.background_gradient(subset=["NBA-Score"], cmap="YlOrRd")
            .format(
                {
                    "Preis ($)": "${:.2f}",
                    "Relevanz": "{:.3f}",
                    "ConvSignal": "{:.3f}",
                    "Marge": "{:.3f}",
                    "Popularität": "{:.3f}",
                    "NBA-Score": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    elif not recs.empty and rec_uid != user_id:
        st.info("Klicke auf **Empfehlungen berechnen**, um den Score zu aktualisieren.")


# ── Seite 4: Segment-Analyse ─────────────────────────────────────────────────
def page_segments(
    dfs: dict[str, pd.DataFrame],
    product_metrics: pd.DataFrame,
) -> None:
    st.title("📈 Segment-Analyse")

    users = dfs["users"]
    orders = dfs["orders"]
    oi = dfs["order_items"]

    users = users.copy()
    users["age_group"] = pd.cut(
        users["age"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["< 25", "25–34", "35–44", "45–54", "55+"],
    )

    seg_option = st.selectbox(
        "Segment-Dimension:",
        ["Geschlecht", "Altersgruppe", "Land (Top 10)", "Traffic Source"],
    )

    seg_map = {
        "Geschlecht": "gender",
        "Altersgruppe": "age_group",
        "Land (Top 10)": "country",
        "Traffic Source": "traffic_source",
    }
    seg_col = seg_map[seg_option]

    seg_users = users.copy()
    if seg_option == "Land (Top 10)":
        top10 = users["country"].value_counts().head(10).index
        seg_users = seg_users[seg_users["country"].isin(top10)]

    orders_ext = orders.merge(
        seg_users[["id", seg_col]], left_on="user_id", right_on="id", how="inner"
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Bestellungen nach {seg_option}")
        agg = orders_ext.groupby(seg_col).size().reset_index(name="Bestellungen")
        fig = px.bar(
            agg,
            x=seg_col,
            y="Bestellungen",
            color=seg_col,
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={seg_col: seg_option},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Conversion Rate nach {seg_option}")
        total_seg = orders_ext.groupby(seg_col).size()
        complete_seg = (
            orders_ext[orders_ext["status"] == "Complete"].groupby(seg_col).size()
        )
        conv_rate = (complete_seg / total_seg * 100).reset_index()
        conv_rate.columns = [seg_option, "Conv Rate (%)"]
        fig = px.bar(
            conv_rate,
            x=seg_option,
            y="Conv Rate (%)",
            color=seg_option,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(yaxis_ticksuffix="%", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    oi_ext = oi.merge(
        seg_users[["id", seg_col]], left_on="user_id", right_on="id", how="inner"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Ø Sale Price nach {seg_option}")
        if "sale_price" in oi_ext.columns:
            avg_price = oi_ext.groupby(seg_col)["sale_price"].mean().reset_index()
            avg_price.columns = [seg_option, "Ø Preis ($)"]
            fig = px.bar(
                avg_price,
                x=seg_option,
                y="Ø Preis ($)",
                color=seg_option,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Retourenquote nach {seg_option}")
        total_items = oi_ext.groupby(seg_col).size()
        returned_items = oi_ext[oi_ext["status"] == "Returned"].groupby(seg_col).size()
        ret_rate = (returned_items / total_items * 100).reset_index()
        ret_rate.columns = [seg_option, "Retourenquote (%)"]
        fig = px.bar(
            ret_rate,
            x=seg_option,
            y="Retourenquote (%)",
            color=seg_option,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(yaxis_ticksuffix="%", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"🏆 Top 10 Produkte (Complete) pro Segment: {seg_option}")
    oi_prod = oi_ext.merge(
        product_metrics[["id", "name", "category"]].rename(
            columns={"id": "product_id"}
        ),
        on="product_id",
        how="left",
    )

    segments = sorted(oi_prod[seg_col].dropna().unique())
    tabs = st.tabs([str(s) for s in segments])

    for tab, seg_val in zip(tabs, segments):
        with tab:
            seg_data = oi_prod[
                (oi_prod[seg_col] == seg_val) & (oi_prod["status"] == "Complete")
            ]
            top_prod = (
                seg_data.groupby(["product_id", "name", "category"])
                .size()
                .reset_index(name="Käufe")
                .sort_values("Käufe", ascending=False)
                .head(10)
            )
            if top_prod.empty:
                st.info("Keine abgeschlossenen Bestellungen in diesem Segment.")
            else:
                fig = px.bar(
                    top_prod,
                    x="Käufe",
                    y="name",
                    orientation="h",
                    color="category",
                    labels={"name": "Produkt"},
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    margin=dict(t=10),
                    height=350,
                    legend_title="Kategorie",
                )
                st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"🎯 NBA-Score-Komponenten nach {seg_option}")
    oi_nba = oi_ext.merge(
        product_metrics[["id", "marge", "popularitaet", "conv_signal"]].rename(
            columns={"id": "product_id"}
        ),
        on="product_id",
        how="left",
    )
    nba_seg = (
        oi_nba.groupby(seg_col)[["marge", "popularitaet", "conv_signal"]]
        .mean()
        .reset_index()
    )

    fig = go.Figure()
    for comp, color in [
        ("conv_signal", "#EF553B"),
        ("marge", "#00CC96"),
        ("popularitaet", "#AB63FA"),
    ]:
        fig.add_trace(
            go.Bar(
                name=comp.replace("_", " ").title(),
                x=nba_seg[seg_col].astype(str),
                y=nba_seg[comp],
                marker_color=color,
            )
        )
    fig.update_layout(
        barmode="group",
        xaxis_title=seg_option,
        yaxis_title="Ø Score-Komponente",
        legend_title="Komponente",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.sidebar.title("🛍️ NBA Score Dashboard")
    st.sidebar.caption("Fashion E-Commerce · Next Best Offer")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dataset-Übersicht",
            "🎯 NBA Score Berechnung",
            "🔮 Interaktiver Simulator",
            "📈 Segment-Analyse",
        ],
    )

    st.sidebar.divider()
    st.sidebar.markdown("""
**NBA-Score Formel:**
                        """)
    st.sidebar.divider()
    st.sidebar.markdown("""
**Datenquellen:**
- 6 CSV-Dateien (Google Drive)
- TheLook Fashion E-Commerce
- ~125k Users · ~181k Orders
""")

    with st.spinner("Lade Daten …"):
        dfs = load_data()

    with st.spinner("Berechne Produktmetriken …"):
        product_metrics = compute_product_metrics(
            dfs["products"], dfs["order_items"], dfs["events"]
        )

    if page == "📊 Dataset-Übersicht":
        page_overview(dfs)
    elif page == "🎯 NBA Score Berechnung":
        page_nba_scores(dfs, product_metrics)
    elif page == "🔮 Interaktiver Simulator":
        page_simulator(dfs, product_metrics)
    elif page == "📈 Segment-Analyse":
        page_segments(dfs, product_metrics)


if __name__ == "__main__":
    main()
