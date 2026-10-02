"""
Generates tableau/Flight_Delays.twbx: a dark-navy dashboard with a row of eight
KPI tiles and six charts (3 x 2 grid), built from the CSVs that
export_dashboard_data.py writes to data/processed/.

The look (theme, tiles, clean axes, borderless zones) follows the Retail
Analytics workbook so the two navy dashboards match. prepare_dashboard_data()
turns the exported aggregates into the small tables the dashboard plots
(dash_*.csv, written next to them); it only reads files already on disk.

Tableau Public only opens workbooks whose data sources are extracts, so every
input table is written to a .hyper file (Tableau Hyper API) and packaged with
the workbook XML into a .twbx.

Runs on the Windows host or inside the Airflow container. The original CSV path
recorded in the workbook (used only for "Refresh Extract") should be the host
path: set TABLEAU_DATA_DIR when running in Docker.
"""
import csv
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import pandas as pd
from tableauhyperapi import (Connection, CreateMode, HyperProcess, SqlType, TableDefinition,
                             TableName, Telemetry, escape_string_literal)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = Path(os.environ.get("PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
TABLEAU_DIR = Path(os.environ.get("TABLEAU_DIR", PROJECT_ROOT / "tableau"))
WORKBOOK_NAME = "Flight_Delays"
TWBX_PATH = TABLEAU_DIR / f"{WORKBOOK_NAME}.twbx"
DATA_DIR_IN_WORKBOOK = os.environ.get("TABLEAU_DATA_DIR", str(PROCESSED_DIR))

# Same navy theme as Retail_Analytics.twbx.
THEME = {
    "background": "#0f1b2d",
    "text": "#9fb9d8",
    "title": "#ffffff",
    "kpi_label": "#7fb2e5",
    "kpi_value": "#ffffff",
    "mark": "#2f6db5",
    "palette": "blue_10_0",
    # Heatmap/treemap cells: light-to-mid blue so navy labels stay readable on every cell.
    "cell_ramp": ["#d6e6f7", "#4a86c8"],
    "label_on_mark": "#0f1b2d",
}

HYPER_TYPES = {
    "integer": SqlType.big_int(),
    "real": SqlType.double(),
    "date": SqlType.date(),
    "datetime": SqlType.timestamp(),
    "string": SqlType.text(),
}
EXTRACT_TABLE = TableName("Extract", "Extract")

# prefix -> (derivation, column-instance type)
AGGREGATIONS = {
    "none": ("None", "nominal"),
    "sum": ("Sum", "quantitative"),
    "avg": ("Avg", "quantitative"),
    "twk": ("Week-Trunc", "quantitative"),
}

CAPTIONS = {
    "month_name": "Month",
    "day_name": "Day of Week",
    "delay_rate": "Delay Rate (15+ min late)",
    "carrier": "Carrier",
    "carrier_name": "Carrier (full name)",
    "on_time_rate": "On-Time Arrival %",
    "flight_week": "Flight Date",
    "airport_code": "Airport",
    "scheduled_flights": "Scheduled Flights",
    "actual_rate": "Actual Delay Rate ({test})",
    "predicted_rate": "Mean Predicted Probability",
    "model": "Model",
    "metric": "Metric",
    "value": "Score (0-1)",
}

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr"]  # replaced from the data in configure_labels()
METRICS = ["AUC-PR", "Precision", "Recall"]
MODELS = ["Random", "Route rule", "Inbound rule", "LogReg", "LightGBM"]
MODEL_SHORT = {"Historical rate rule (carrier+route)": "Route rule", "Inbound-delay rule": "Inbound rule",
               "Logistic regression": "LogReg", "LightGBM": "LightGBM"}

KPIS = [  # (column in dash_kpis.csv, tile label)
    ("total_flights", "TOTAL FLIGHTS"),
    ("on_time", "ON-TIME ARRIVAL"),
    ("avg_delay", "AVG DELAY"),
    ("cancel_rate", "CANCEL RATE"),
    ("worst_carrier", "WORST CARRIER"),
    ("auc_pr", "LIGHTGBM AUC-PR"),
    ("precision", "PRECISION (20%)"),
    ("recall", "RECALL (20%)"),
]

DATASOURCES = {
    "kpi": {"caption": "KPIs", "csv": "dash_kpis.csv"},
    "heat": {"caption": "Delay Rate by Day and Month", "csv": "dash_delay_heatmap.csv"},
    "carrier": {"caption": "On-Time by Carrier", "csv": "dash_ontime_by_carrier.csv"},
    "weekly": {"caption": "Weekly Delay Rate", "csv": "dash_weekly_delay.csv"},
    "airport": {"caption": "Busiest Origin Airports", "csv": "dash_top_airports.csv"},
    "mcarrier": {"caption": "Predicted vs Actual by Carrier", "csv": "dash_model_by_carrier.csv"},
    "models": {"caption": "Model Comparison", "csv": "dash_model_comparison.csv"},
}

CHARTS = [
    # Heatmap (highlight table): January storms and the weekday pattern at a glance.
    {"name": "Delay Rate by Day of Week and Month", "ds": "heat", "mark": "Square",
     "rows": [("none", "month_name")], "cols": [("none", "day_name")],
     "color": ("sum", "delay_rate"), "text": [("sum", "delay_rate")], "labels": True,
     "ramp": "cell", "manual_order": [("month_name", MONTHS), ("day_name", DAYS)]},
    {"name": "On-Time Arrival % by Carrier", "ds": "carrier", "mark": "Bar",
     "rows": [("none", "carrier")], "cols": [("sum", "on_time_rate")],
     "color": ("sum", "on_time_rate"), "labels": True,
     "label_size": 7, "header_size": 7,  # 15 carriers in a short tile
     "sort_desc_by": ("carrier", ("sum", "on_time_rate"))},
    {"name": "Weekly Delay Rate (15+ min late), {years}", "ds": "weekly", "mark": "Area",
     "rows": [("sum", "delay_rate")], "cols": [("twk", "flight_week")]},
    # Treemap: the 20 busiest origin airports, sized by flights, shaded by delay rate.
    {"name": "20 Busiest Airports, Shaded by Delay Rate", "ds": "airport", "mark": "Square",
     "rows": [], "cols": [], "size": ("sum", "scheduled_flights"), "color": ("sum", "delay_rate"),
     "text": [("none", "airport_code"), ("sum", "delay_rate")], "labels": True, "ramp": "cell"},
    # Scatter: one dot per carrier; distance from the diagonal is model error.
    {"name": "{test}: Predicted vs Actual by Carrier", "ds": "mcarrier", "mark": "Circle",
     "rows": [("sum", "predicted_rate")], "cols": [("sum", "actual_rate")],
     "detail": ("none", "carrier_name"), "text": [("none", "carrier")], "labels": True,
     "label_size": 8},
    {"name": "{test}, 2 h Before: Top-20% Flags", "ds": "models", "mark": "Bar",
     "rows": [("none", "model")], "cols": [("none", "metric"), ("sum", "value")],
     "color": ("sum", "value"), "labels": True,
     "manual_order": [("metric", METRICS), ("model", MODELS)]},
]

DASHBOARD_NAME = "US Flight Delays and Delay Model ({period})"
KPI_HEIGHT, GAP = 12000, 600
CHART_W = (100000 - 4 * GAP) // 3
CHART_H = (100000 - KPI_HEIGHT - 4 * GAP) // 2
KPI_W = (100000 - (len(KPIS) + 1) * GAP) // len(KPIS)

CLEAN_RULES = """
          <style-rule element='gridline'>
            <format attr='line-visibility' scope='rows' value='off' />
            <format attr='line-visibility' scope='cols' value='off' />
          </style-rule>
          <style-rule element='zeroline'>
            <format attr='line-visibility' value='off' />
          </style-rule>
          <style-rule element='axis'>
            <format attr='line-visibility' value='off' />
          </style-rule>
          <style-rule element='table-div'>
            <format attr='line-visibility' scope='rows' value='off' />
            <format attr='line-visibility' scope='cols' value='off' />
          </style-rule>"""

FORMATS = {c: "p0.0%" for c in ("delay_rate", "on_time_rate", "actual_rate", "predicted_rate")}
FORMATS["value"] = "n#,##0.000"

ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------- dashboard data

def short_carrier(name):
    return re.sub(r"\s+(Air Lines|Airlines|Airways|Air)$", "", name)


def prepare_dashboard_data():
    """Turns the exported aggregates into the small tables the dashboard plots."""
    carriers = pd.read_csv(PROCESSED_DIR / "ontime_by_carrier.csv")
    dow = pd.read_csv(PROCESSED_DIR / "ontime_by_dow_month.csv")
    daily = pd.read_csv(PROCESSED_DIR / "ontime_daily.csv", parse_dates=["flight_date"])
    origin = pd.read_csv(PROCESSED_DIR / "ontime_by_origin.csv")
    by_carrier = pd.read_csv(PROCESSED_DIR / "model_by_carrier.csv")
    metrics = pd.read_csv(PROCESSED_DIR / "model_metrics.csv")
    cm = pd.read_csv(PROCESSED_DIR / "confusion_matrix.csv")

    def rate(delayed, completed):
        return (delayed / completed).round(4)

    heat = dow[["month_name", "day_name"]].copy()
    heat["delay_rate"] = rate(dow["delayed_flights"], dow["completed_flights"])
    heat.to_csv(PROCESSED_DIR / "dash_delay_heatmap.csv", index=False)

    car = pd.DataFrame({"carrier": carriers["carrier_name"].map(short_carrier),
                        "on_time_rate": rate(carriers["completed_flights"] - carriers["delayed_flights"],
                                             carriers["completed_flights"])})
    car.to_csv(PROCESSED_DIR / "dash_ontime_by_carrier.csv", index=False)

    # Sunday-start weeks, matching Tableau's Week-Trunc (en_US), which the chart uses.
    # The first (Jan 1-6) and last (Apr 28-30) weeks are partial.
    daily["flight_week"] = daily["flight_date"] - pd.to_timedelta((daily["flight_date"].dt.dayofweek + 1) % 7, unit="D")
    weekly = daily.groupby("flight_week")[["delayed_flights", "completed_flights"]].sum().reset_index()
    weekly["delay_rate"] = rate(weekly["delayed_flights"], weekly["completed_flights"])
    weekly["flight_week"] = weekly["flight_week"].dt.strftime("%Y-%m-%d")
    weekly[["flight_week", "delay_rate"]].to_csv(PROCESSED_DIR / "dash_weekly_delay.csv", index=False)

    top = origin.nlargest(20, "scheduled_flights")
    top = pd.DataFrame({"airport_code": top["airport_code"], "scheduled_flights": top["scheduled_flights"],
                        "delay_rate": rate(top["delayed_flights"], top["completed_flights"])})
    top.to_csv(PROCESSED_DIR / "dash_top_airports.csv", index=False)

    mc = pd.DataFrame({"carrier_name": by_carrier["carrier_name"],
                       "carrier": by_carrier["carrier_name"].map(short_carrier),
                       "actual_rate": (by_carrier["actual_delay_rate_pct"] / 100).round(4),
                       "predicted_rate": (by_carrier["mean_predicted_probability_pct"] / 100).round(4)})
    mc.to_csv(PROCESSED_DIR / "dash_model_by_carrier.csv", index=False)

    # A random ranking flagging 20% has expected AUC-PR = precision = base rate, recall = 20%.
    counts = cm.set_index(["actual", "predicted"])["flights"]
    base_rate = counts.loc["Delayed"].sum() / counts.sum()
    flag_rate = counts.xs("Delayed", level="predicted").sum() / counts.sum()
    rows = [("Random", "AUC-PR", base_rate), ("Random", "Precision", base_rate),
            ("Random", "Recall", flag_rate)]
    for _, m in metrics.iterrows():
        short = MODEL_SHORT[m["model"]]
        rows += [(short, "AUC-PR", m["auc_pr"]), (short, "Precision", m["precision"]),
                 (short, "Recall", m["recall"])]
    pd.DataFrame(rows, columns=["model", "metric", "value"]).round(4).to_csv(
        PROCESSED_DIR / "dash_model_comparison.csv", index=False)

    completed = carriers["completed_flights"].sum()
    worst = carriers.loc[(carriers["delayed_flights"] / carriers["completed_flights"]).idxmax()]
    lgbm = metrics.set_index("model").loc["LightGBM"]
    kpis = {
        "total_flights": f"{carriers['scheduled_flights'].sum() / 1e6:.2f}M",
        "on_time": f"{1 - carriers['delayed_flights'].sum() / completed:.1%}",
        # Mean of the carrier averages weighted by completed flights = overall mean arrival delay.
        "avg_delay": f"{(carriers['avg_arr_delay_minutes'] * carriers['completed_flights']).sum() / completed:.1f} min",
        "cancel_rate": f"{carriers['cancelled_flights'].sum() / carriers['scheduled_flights'].sum():.2%}",
        "worst_carrier": short_carrier(worst["carrier_name"]),
        "auc_pr": f"{lgbm['auc_pr']:.3f}",
        # From the LightGBM confusion matrix, two decimals: 0.3205 in model_metrics.csv is
        # 32.046%, which one-decimal rounding of the 4-dp value would misstate as 32.1%.
        "precision": f"{counts.loc[('Delayed', 'Delayed')] / counts.xs('Delayed', level='predicted').sum():.2%}",
        "recall": f"{counts.loc[('Delayed', 'Delayed')] / counts.loc['Delayed'].sum():.2%}",
    }
    pd.DataFrame([kpis]).to_csv(PROCESSED_DIR / "dash_kpis.csv", index=False)
    print("KPIs:", kpis)


# ---------------------------------------------------------------- workbook XML

def infer_type(values):
    """Returns (datatype, role, type) for a CSV column from sample values."""
    vals = [v for v in values if v not in ("", None)]
    if vals and all(re.fullmatch(r"-?\d+", v) for v in vals):
        return "integer", "measure", "quantitative"
    if vals and all(re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", v) for v in vals):
        return "real", "measure", "quantitative"
    if vals and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) for v in vals):
        return "date", "dimension", "ordinal"
    if vals and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", v) for v in vals):
        return "datetime", "dimension", "ordinal"
    return "string", "dimension", "nominal"


def read_schema(csv_path, sample_rows=1000):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        samples = [row for _, row in zip(range(sample_rows), reader)]
    schema = []
    for i, name in enumerate(header):
        if not ID_RE.match(name):
            raise ValueError(f"{csv_path.name}: column {name!r} needs quoting support")
        schema.append((name, i, *infer_type([r[i] for r in samples if i < len(r)])))
    return schema


def caption(col):
    return CAPTIONS.get(col, col.replace("_", " ").title())


def ds_name(key):
    return f"federated.{key}"


def instance_name(prefix, col):
    kind = "nk" if AGGREGATIONS[prefix][1] == "nominal" else "qk"
    return f"[{prefix}:{col}:{kind}]"


def field_ref(ds_key, prefix, col):
    return f"[{ds_name(ds_key)}].{instance_name(prefix, col)}"


def write_hyper(csv_path, schema, hyper_path, hyper):
    """Loads csv_path into Extract.Extract in a new .hyper file; returns row count."""
    table = TableDefinition(EXTRACT_TABLE, [
        TableDefinition.Column(name, HYPER_TYPES[dt]) for name, _, dt, _, _ in schema
    ])
    with Connection(hyper.endpoint, hyper_path, CreateMode.CREATE_AND_REPLACE) as conn:
        conn.catalog.create_schema(EXTRACT_TABLE.schema_name)
        conn.catalog.create_table(table)
        return conn.execute_command(
            f"COPY {EXTRACT_TABLE} FROM {escape_string_literal(str(csv_path))} "
            "WITH (format csv, header, NULL '')"
        )


def extract_xml(key, row_count):
    now = datetime.now()
    return f"""      <extract count='-1' enabled='true' units='records'>
        <connection access_mode='readonly' authentication='auth-none' author-locale='en_US' class='hyper' dbname='Data/Extracts/{key}.hyper' default-settings='hyper' schema='Extract' sslmode='' tablename='Extract' update-time={quoteattr(now.strftime('%m/%d/%Y %I:%M:%S %p'))} username='tableau'>
          <relation name='Extract' table='[Extract].[Extract]' type='table' />
          <refresh>
            <refresh-event add-from-file-path={quoteattr(key)} increment-value='%null%' refresh-type='create' rows-inserted='{row_count}' timestamp-start={quoteattr(now.strftime('%Y-%m-%d %H:%M:%S.000'))} />
          </refresh>
        </connection>
      </extract>"""


def datasource_xml(key, spec, schema, row_count):
    csv_file = spec["csv"]
    stem = Path(csv_file).stem
    relation_cols = "\n".join(
        f"            <column datatype='{dt}' name={quoteattr(n)} ordinal='{i}' />"
        for n, i, dt, _, _ in schema
    )
    ds_cols = "\n".join(
        f"      <column caption={quoteattr(caption(n))} datatype='{dt}'"
        + (f" default-format='{FORMATS[n]}'" if n in FORMATS else "")
        + f" name='[{n}]' role='{role}' type='{typ}' />"
        for n, _, dt, role, typ in schema
    )
    return f"""    <datasource caption={quoteattr(spec['caption'])} inline='true' name='{ds_name(key)}' version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption={quoteattr(stem)} name='textscan.{key}'>
            <connection class='textscan' directory={quoteattr(DATA_DIR_IN_WORKBOOK.replace(chr(92), '/'))} filename={quoteattr(csv_file)} password='' server='' />
          </named-connection>
        </named-connections>
        <relation connection='textscan.{key}' name={quoteattr(csv_file)} table='[{stem}#csv]' type='table'>
          <columns character-set='UTF-8' header='yes' locale='en_US' separator=','>
{relation_cols}
          </columns>
        </relation>
      </connection>
      <aliases enabled='yes' />
{ds_cols}
{extract_xml(key, row_count)}
    </datasource>"""


def title_xml(text, color, size, align=None):
    align_attr = " fontalignment='1'" if align == "center" else ""
    return f"""      <layout-options>
        <title>
          <formatted-text>
            <run bold='true' fontcolor='{color}' fontsize='{size}'{align_attr}>{escape(text)}</run>
          </formatted-text>
        </title>
      </layout-options>"""


def sheet_style_xml(extra_rules=""):
    return f"""        <style>
          <style-rule element='table'>
            <format attr='background-color' value='{THEME["background"]}' />
          </style-rule>
          <style-rule element='worksheet'>
            <format attr='color' value='{THEME["text"]}' />
          </style-rule>{CLEAN_RULES}{extra_rules}
        </style>"""


def gradient_xml(field, colors):
    """Custom blue ramp for a continuous colour measure (a custom-interpolated palette is
    honoured by Tableau; used where labels sit inside the marks)."""
    stops = "\n".join(f"                <color>{c}</color>" for c in colors)
    return f"""
          <style-rule element='mark'>
            <encoding attr='color' field='{field}' type='custom-interpolated'>
              <color-palette custom='true' name='' type='ordered-sequential'>
{stops}
              </color-palette>
            </encoding>
          </style-rule>"""


def pane_style_xml(labels, fixed_color):
    formats = []
    if labels:
        formats.append("<format attr='mark-labels-show' value='true' />")
    if fixed_color:
        formats.append(f"<format attr='mark-color' value='{THEME['mark']}' />")
    if not formats:
        return ""
    body = "\n".join(f"                {f}" for f in formats)
    return f"""
            <style>
              <style-rule element='mark'>
{body}
              </style-rule>
            </style>"""


def shelf(key, fields):
    """Multiple fields on one shelf are nested (Tableau's '/' operator); empty -> ''."""
    refs = [field_ref(key, p, c) for p, c in fields]
    if not refs:
        return ""
    return refs[0] if len(refs) == 1 else "(" + " / ".join(refs) + ")"


def shelf_xml(tag, key, fields):
    value = shelf(key, fields)
    return f"<{tag}>{value}</{tag}>" if value else f"<{tag} />"


def dependencies_xml(key, types, used):
    deps = []
    for col in dict.fromkeys(c for _, c in used):
        dt, role, typ = types[col]
        fmt = f" default-format='{FORMATS[col]}'" if col in FORMATS else ""
        deps.append(f"            <column caption={quoteattr(caption(col))} datatype='{dt}'{fmt} "
                    f"name='[{col}]' role='{role}' type='{typ}' />")
        for prefix, c in used:
            if c == col:
                derivation, itype = AGGREGATIONS[prefix]
                deps.append(
                    f"            <column-instance column='[{col}]' derivation='{derivation}' "
                    f"name='{instance_name(prefix, col)}' pivot='key' type='{itype}' />"
                )
    return "\n".join(deps)


def chart_xml(ws, schemas):
    key = ws["ds"]
    types = {n: (dt, role, typ) for n, _, dt, role, typ in schemas[key]}
    extra = [ws[k] for k in ("color", "detail", "size") if k in ws] + ws.get("text", [])
    used = list(dict.fromkeys(ws["rows"] + ws["cols"] + extra))
    sort_xml = ""
    if "sort_desc_by" in ws:
        dim, measure = ws["sort_desc_by"]
        sort_xml += f"""
          <sort class='computed' column='{field_ref(key, "none", dim)}' direction='DESC' using='{field_ref(key, *measure)}' />"""
    for dim, values in ws.get("manual_order", []):
        buckets = "\n".join(f"              <bucket>{escape(chr(34) + v + chr(34))}</bucket>" for v in values)
        sort_xml += f"""
          <sort class='manual' column='{field_ref(key, "none", dim)}' direction='ASC'>
            <dictionary>
{buckets}
            </dictionary>
          </sort>"""
    enc = []
    if "color" in ws:
        enc.append(f"<color column='{field_ref(key, *ws['color'])}' />")
    if "size" in ws:
        enc.append(f"<size column='{field_ref(key, *ws['size'])}' />")
    enc += [f"<text column='{field_ref(key, *t)}' />" for t in ws.get("text", [])]
    if "detail" in ws:
        enc.append(f"<lod column='{field_ref(key, *ws['detail'])}' />")
    encodings = ""
    if enc:
        body = "\n".join(f"              {e}" for e in enc)
        encodings = f"""
            <encodings>
{body}
            </encodings>"""
    in_cell = ws.get("ramp") == "cell"
    if ws.get("text") and (in_cell or "label_size" in ws):
        # Labels inside the light-to-mid cells get navy text; a smaller size where marks are dense.
        label = "&#10;".join(f"&lt;{field_ref(key, *t)}&gt;" for t in ws["text"])
        color = THEME["label_on_mark"] if in_cell else THEME["text"]
        size = f" fontsize='{ws['label_size']}'" if "label_size" in ws else ""
        encodings += f"""
            <customized-label>
              <formatted-text>
                <run fontcolor='{color}'{size}>{label}</run>
              </formatted-text>
            </customized-label>"""
    elif "label_size" in ws and "color" in ws:
        # Bar labels: the value shown on the colour/length measure, in a smaller font.
        label = f"&lt;{field_ref(key, *ws['color'])}&gt;"
        encodings += f"""
            <customized-label>
              <formatted-text>
                <run fontcolor='{THEME["text"]}' fontsize='{ws["label_size"]}'>{label}</run>
              </formatted-text>
            </customized-label>"""
    encodings += pane_style_xml(ws.get("labels", False), fixed_color="color" not in ws)
    extra_rules = gradient_xml(field_ref(key, *ws["color"]), THEME["cell_ramp"]) if in_cell else ""
    if "header_size" in ws:
        extra_rules += f"""
          <style-rule element='header'>
            <format attr='font-size' value='{ws["header_size"]}' />
          </style-rule>"""
    return f"""    <worksheet name={quoteattr(ws['name'])}>
{title_xml(ws['name'], THEME['title'], 12)}
      <table>
        <view>
          <datasources>
            <datasource caption={quoteattr(DATASOURCES[key]['caption'])} name='{ds_name(key)}' />
          </datasources>
          <datasource-dependencies datasource='{ds_name(key)}'>
{dependencies_xml(key, types, used)}
          </datasource-dependencies>{sort_xml}
          <aggregation value='true' />
        </view>
{sheet_style_xml(extra_rules)}
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class='{ws['mark']}' />{encodings}
          </pane>
        </panes>
        {shelf_xml('rows', key, ws['rows'])}
        {shelf_xml('cols', key, ws['cols'])}
      </table>
    </worksheet>"""


def kpi_style_xml():
    return f"""        <style>
          <style-rule element='table'>
            <format attr='background-color' value='{THEME["background"]}' />
          </style-rule>
          <style-rule element='cell'>
            <format attr='font-size' value='20' />
            <format attr='font-weight' value='bold' />
            <format attr='color' value='{THEME["kpi_value"]}' />
            <format attr='text-align' value='center' />
            <format attr='vertical-align' value='center' />
          </style-rule>{CLEAN_RULES}
        </style>"""


def kpi_xml(column, label, schemas):
    """A big-number tile: one text mark showing a preformatted value, titled with the label."""
    types = {n: (dt, role, typ) for n, _, dt, role, typ in schemas["kpi"]}
    ref = field_ref("kpi", "none", column)
    return f"""    <worksheet name={quoteattr(label)}>
{title_xml(label, THEME['kpi_label'], 9, align='center')}
      <table>
        <view>
          <datasources>
            <datasource caption='KPIs' name='{ds_name("kpi")}' />
          </datasources>
          <datasource-dependencies datasource='{ds_name("kpi")}'>
{dependencies_xml("kpi", types, [("none", column)])}
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
{kpi_style_xml()}
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class='Text' />
            <encodings>
              <text column='{ref}' />
            </encodings>
            <customized-label>
              <formatted-text>
                <run bold='true' fontalignment='1' fontcolor='{THEME["kpi_value"]}' fontsize='24'>&lt;{ref}&gt;</run>
              </formatted-text>
            </customized-label>
          </pane>
        </panes>
        <rows />
        <cols />
      </table>
    </worksheet>"""


def zones():
    """KPI tiles across the top, charts in a 3 x 2 grid below."""
    placed, zone_id = [], 2
    for i, (_, label) in enumerate(KPIS):
        x = GAP + i * (KPI_W + GAP)
        placed.append((zone_id, label, x, GAP, KPI_W, KPI_HEIGHT)); zone_id += 1
    for i, chart in enumerate(CHARTS):
        col, row = i % 3, i // 3
        x = GAP + col * (CHART_W + GAP)
        y = KPI_HEIGHT + 2 * GAP + row * (CHART_H + GAP)
        placed.append((zone_id, chart["name"], x, y, CHART_W, CHART_H)); zone_id += 1
    zone_style = ("            <zone-style>\n"
                  "              <format attr='border-style' value='none' />\n"
                  "              <format attr='border-width' value='0' />\n"
                  "              <format attr='margin' value='4' />\n"
                  "            </zone-style>\n")
    return "\n".join(
        f"          <zone h='{h}' id='{zid}' name={quoteattr(name)} w='{w}' x='{x}' y='{y}'>\n"
        f"{zone_style}          </zone>"
        for zid, name, x, y, w, h in placed
    )


def dashboard_xml():
    return f"""    <dashboard name={quoteattr(DASHBOARD_NAME)}>
      <style>
        <style-rule element='table'>
          <format attr='background-color' value='{THEME["background"]}' />
        </style-rule>
      </style>
      <size maxheight='900' maxwidth='1600' minheight='900' minwidth='1600' />
      <zones>
        <zone h='100000' id='1' type-v2='layout-basic' w='100000' x='0' y='0'>
{zones()}
        </zone>
      </zones>
    </dashboard>"""


def windows_xml():
    names = [label for _, label in KPIS] + [c["name"] for c in CHARTS]
    viewpoints = "\n".join(
        f"        <viewpoint name={quoteattr(n)}>\n          <zoom type='entire-view' />\n        </viewpoint>"
        for n in names
    )
    return f"""  <windows source-height='30'>
    <window class='dashboard' maximized='true' name={quoteattr(DASHBOARD_NAME)}>
      <viewpoints>
{viewpoints}
      </viewpoints>
      <active id='-1' />
    </window>
  </windows>"""


def configure_labels():
    """Titles, captions and the month order come from the data, so the workbook stays right as
    new BTS months arrive (the look is unchanged)."""
    global DASHBOARD_NAME
    dow = pd.read_csv(PROCESSED_DIR / "ontime_by_dow_month.csv")
    daily = pd.read_csv(PROCESSED_DIR / "ontime_daily.csv", parse_dates=["flight_date"])
    test = pd.read_csv(PROCESSED_DIR / "model_metrics.csv")["test_month"].iloc[0]   # e.g. "Apr 2024"
    first, last = daily["flight_date"].min(), daily["flight_date"].max()
    period = (f"{first:%b}-{last:%b %Y}" if first.year == last.year else f"{first:%b %Y}-{last:%b %Y}")
    years = str(first.year) if first.year == last.year else f"{first.year}-{last.year}"
    MONTHS[:] = list(dow.sort_values("flight_month")["month_name"].drop_duplicates())
    fmt = {"test": test, "period": period, "years": years}
    for c in CHARTS:
        c["name"] = c["name"].format(**fmt)
    CAPTIONS["actual_rate"] = CAPTIONS["actual_rate"].format(**fmt)
    DASHBOARD_NAME = DASHBOARD_NAME.format(**fmt)
    print(f"Labels: {DASHBOARD_NAME!r}, test month {test}, months {MONTHS}")


def build():
    prepare_dashboard_data()
    configure_labels()
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="twbx_"))
    schemas, row_counts, hyper_files = {}, {}, {}
    # log_dir keeps hyperd.log out of the working directory; work_dir is deleted below.
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
                      parameters={"log_dir": str(work_dir)}) as hyper:
        for key, spec in DATASOURCES.items():
            path = PROCESSED_DIR / spec["csv"]
            if not path.exists():
                raise FileNotFoundError(f"{path} not found; run export_dashboard_data.py first")
            schemas[key] = read_schema(path)
            hyper_files[key] = work_dir / f"{key}.hyper"
            row_counts[key] = write_hyper(path, schemas[key], hyper_files[key], hyper)
            print(f"  {spec['csv']}: {row_counts[key]} rows -> {key}.hyper")
    datasources = "\n".join(
        datasource_xml(k, s, schemas[k], row_counts[k]) for k, s in DATASOURCES.items()
    )
    worksheets = "\n".join(
        [kpi_xml(col, label, schemas) for col, label in KPIS] + [chart_xml(c, schemas) for c in CHARTS]
    )
    xml = f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook original-version='18.1' source-build='2026.2.0 (20262.26.0819.2015)' source-platform='win' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <preferences>
    <preference name='ui.encoding.shelf.height' value='24' />
    <preference name='ui.shelf.height' value='26' />
  </preferences>
  <datasources>
{datasources}
  </datasources>
  <worksheets>
{worksheets}
  </worksheets>
  <dashboards>
{dashboard_xml()}
  </dashboards>
{windows_xml()}
</workbook>
"""
    with zipfile.ZipFile(TWBX_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{WORKBOOK_NAME}.twb", xml)
        for key, hyper_path in hyper_files.items():
            z.write(hyper_path, f"Data/Extracts/{key}.hyper")
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"Wrote {TWBX_PATH}")


if __name__ == "__main__":
    build()
