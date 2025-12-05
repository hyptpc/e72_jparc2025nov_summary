import numpy as np
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch


CSV_FILE = "E72_downlog.csv"
RUN_CSV = "E72_runlog.csv"  

DUMMY_COLOR = "#D9D9D9"
PEDESTAL_COLOR = "#D2691E"
TPC_DEAD_COLOR = "magenta"
OTHERS_COLOR= "#00FFFF" #Commissioning

DEBUG = 1

def make_autopct(sizes):
    def _autopct(pct):
        total = sum(sizes)
        hours = pct / 100.0 * total
        return f"{pct:.1f}%\n({hours:.1f} h)"
    return _autopct

def normalize_time_str(t) -> str:
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return ""  

    t = str(t).strip()
    if t.lower() == "nan" or t == "":
        return ""
    
    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", t):
        return t

    m = re.match(r"^(\d{1,2}):(\d{2})$", t)
    if m:
        h, mnt = m.groups()
        return f"{int(h):02d}:{mnt}:00"

    return t

def classify_run(title: str) -> str:
    t = title.lower()

    if "physics run" in t:
        return "Physics"
    #elif "pedestal" in t:
    #return "Pedestal"
    dummy_keys = [
        "dummy", "junk", "beam check",
        "tune", "tuning", "adjustment",
        "beam test", "test", "clock", "pedestal"
    ]

    if any(k in t for k in dummy_keys):
        return "Dummy"
    return "Other"


    
df_raw = pd.read_csv(CSV_FILE, header=None)
if DEBUG:
    print("Raw shape:", df_raw.shape)
    print(df_raw.head())


header_row_idx = None

for i, row in df_raw.iterrows():
    vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
    if "time" in vals and "status" in vals:
        header_row_idx = i
        break

if header_row_idx is None:
    raise RuntimeError("Could not find header row")

print("Detected header row index:", header_row_idx)


header = df_raw.iloc[header_row_idx].astype(str).tolist()
df = df_raw.iloc[header_row_idx + 1:].copy()
df.columns = header


df.columns = df.columns.map(lambda x: str(x).replace("\ufeff", "").strip())
print("Columns after cleaning:", df.columns.tolist())

df = df.loc[:, ~(df.columns.str.contains("^Unnamed") & df.isna().all())]


cols = df.columns.tolist()

time_col = None
for c in cols:
    if "time" == c.lower().strip():
        time_col = c
        break
if time_col is None:
    time_col = cols[0]

status_col = None
for c in cols:
    if "status" == c.lower().strip():
        status_col = c
        break
if status_col is None and len(cols) > 1:
    status_col = cols[1]

note_col = None
for c in cols:
    if "note" == c.lower().strip():
        note_col = c
        break
if note_col is None and len(cols) > 2:
    note_col = cols[2]

print("Using columns -> time:", time_col,
      ", status:", status_col,
      ", note:", note_col)


df = df[df[time_col].notna()].copy()
df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col)

intervals = []
current = None

for _, row in df.iterrows():
    t = row[time_col]
    status = str(row[status_col]).strip().lower()
    note = row[note_col] if note_col is not None else ""

    if status == "start":
        if current is not None:
            intervals.append({
                "start": current["start"],
                "end": t,
                "start_note": current["start_note"],
                "stop_note": "",    
            })
        current = {"start": t, "start_note": note}

    elif status in ("stop", "end"):
        if current is not None:
            intervals.append({
                "start": current["start"],
                "end": t,
                "start_note": current["start_note"],
                "stop_note": note, 
            })
            current = None

gdf = pd.DataFrame(intervals)
if gdf.empty:
    raise RuntimeError("No interval")

gdf["duration"] = gdf["end"] - gdf["start"]

total_beam = gdf["duration"].sum()
tot_sec = total_beam.total_seconds()
tot_hours = int(tot_sec // 3600)
tot_minutes = int((tot_sec % 3600) // 60)
total_beam_str = f"{tot_hours:02d}:{tot_minutes:02d} h"

E72_PLANNED = 312.0  # h
E45_PLANNED = 12.0   # h
PLANNED_TOTAL = E72_PLANNED + E45_PLANNED  # 324 h

actual_hours = total_beam.total_seconds() / 3600.0
lost_hours = max(PLANNED_TOTAL - actual_hours, 0.0)

print(f"\nPlanned total : {PLANNED_TOTAL:.1f} h "
      f"(E72 {E72_PLANNED:.1f} h + E45 {E45_PLANNED:.1f} h)")
print(f"Delivered     : {actual_hours:.2f} h")
print(f"Lost vs plan  : {lost_hours:.2f} h")

if DEBUG:
    print("\n===== All intervals (start–end pairs) =====")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    gdf_print = gdf.fillna("")
    print(gdf_print[["start", "end", "start_note", "stop_note", "duration"]])
    print("\nTotal intervals:", len(gdf))


run_raw = pd.read_csv(RUN_CSV, header=None)

run_header_idx = None
for i, row in run_raw.iterrows():
    vals = [str(v).strip().lower() for v in row.values if pd.notna(v)]
    if "date" in vals and "run number" in vals:
        run_header_idx = i
        break

if run_header_idx is None:
    raise RuntimeError("Run CSV에서 헤더(Date/Run Number)를 찾지 못했어.")

run_header = run_raw.iloc[run_header_idx].astype(str).tolist()
run_df = run_raw.iloc[run_header_idx + 1:].copy()
run_df.columns = run_header

run_df = run_df.loc[:, ~run_df.columns.str.contains("^Unnamed")]
run_df = run_df[run_df["Date"].notna()].copy()

run_df["Date"]       = run_df["Date"].astype(str).str.strip()
run_df["Start Time"] = run_df["Start Time"].astype(str).str.strip().apply(normalize_time_str)
run_df["End Time"]   = run_df["End Time"].astype(str).str.strip().apply(normalize_time_str)
run_df["Title"]      = run_df["Title"].astype(str).str.strip()

start_str = (run_df["Date"].fillna("") + " " + run_df["Start Time"].fillna("")).str.strip()
end_str   = (run_df["Date"].fillna("") + " " + run_df["End Time"].fillna("")).str.strip()

run_df["run_start"] = pd.to_datetime(start_str, errors="coerce")
run_df["run_end"]   = pd.to_datetime(end_str,   errors="coerce")

bad = run_df[run_df["run_start"].isna() | run_df["run_end"].isna()][
    ["Date", "Start Time", "End Time", "Title"]
]

if not bad.empty:
    print("\n[WARN] These runs have invalid or empty times and will be ignored:")
    print(bad.to_string(index=False))

run_df = run_df[run_df["run_start"].notna() & run_df["run_end"].notna()].copy()




MOM_LIST = [0.645, 0.665, 0.685, 0.715, 0.735, 0.755,
            0.790, 0.814, 0.842, 0.870, 0.902, 0.933]
MOM_STR_LIST = [f"{m:.3f}" for m in MOM_LIST]

def extract_momentum(title: str):
    for m_str in MOM_STR_LIST:
        if m_str in title:
            return m_str
    m = re.search(r"0\.\d{3}", title)
    if m:
        return m.group(0)
    return None


run_df["TPC_DAQ_Dead"] = pd.to_numeric(
    run_df["TPC DAQ Dead"], errors="coerce"
).fillna(0).astype(int)
run_df["TPC_DAQ_Dead"] = pd.to_numeric(run_df["TPC DAQ Dead"], errors="coerce").fillna(0).astype(int)
run_df["base_category"] = run_df["Title"].apply(classify_run)
run_df["category"] = run_df["base_category"]
run_df.loc[run_df["TPC_DAQ_Dead"] == 1, "category"] = "TPC_Dead"

run_df["momentum"] = run_df["Title"].apply(extract_momentum)

beam_segments = []

for _, b in gdf.iterrows():
    b_start = b["start"]
    b_end   = b["end"]
    start_note = b["start_note"]
    stop_note  = b["stop_note"]

    runs = run_df[(run_df["run_end"] > b_start) & (run_df["run_start"] < b_end)].copy()
    runs = runs.sort_values("run_start")

    ptr = b_start

    if runs.empty:
        beam_segments.append({
            "start": b_start,
            "end":   b_end,
            "start_note": start_note,
            "stop_note":  stop_note,
            "category": "NoRun",
            "momentum": None,
            "title": "",
        })
        continue

    for _, r in runs.iterrows():
        r_start = r["run_start"]
        r_end   = r["run_end"]

        if r_start > ptr:
            beam_segments.append({
                "start": ptr,
                "end":   min(r_start, b_end),
                "start_note": start_note,
                "stop_note":  stop_note,
                "category": "NoRun",
                "momentum": None,
                "title": "",
            })
            ptr = r_start
            if ptr >= b_end:
                break

        seg_start = max(ptr, r_start)
        seg_end   = min(b_end, r_end)
        if seg_end > seg_start:
            beam_segments.append({
                "start": seg_start,
                "end":   seg_end,
                "start_note": start_note,
                "stop_note":  stop_note,
                "category": r["category"],
                "momentum": r["momentum"],
                "title": r["Title"],
            })
            ptr = seg_end
            if ptr >= b_end:
                break

    if ptr < b_end:
        beam_segments.append({
            "start": ptr,
            "end":   b_end,
            "start_note": start_note,
            "stop_note":  stop_note,
            "category": "NoRun",
            "momentum": None,
            "title": "",
        })

beam_run_df = pd.DataFrame(beam_segments)
beam_run_df["duration"] = beam_run_df["end"] - beam_run_df["start"]


segments = []
for _, r in beam_run_df.iterrows():
    start = r["start"]
    end   = r["end"]
    start_note = r["start_note"]
    stop_note  = r["stop_note"]
    category   = r["category"]
    momentum   = r["momentum"]
    title      = r["title"]

    day = start.normalize()
    last_day = end.normalize()

    while day <= last_day:
        day_start = day
        day_end   = day + pd.Timedelta(days=1)

        seg_start = max(start, day_start)
        seg_end   = min(end,   day_end)

        if seg_end > seg_start:
            segments.append({
                "date": day.date(),
                "seg_start": seg_start,
                "seg_end":   seg_end,
                "start_note": start_note,
                "stop_note":  stop_note,
                "category":   category,
                "momentum":   momentum,
                "title":      title,
            })

        day += pd.Timedelta(days=1)

segdf = pd.DataFrame(segments)

segdf["duration"] = segdf["seg_end"] - segdf["seg_start"]
segdf["dur_h"]   = segdf["duration"] / pd.Timedelta(hours=1)

phys_mask = segdf["momentum"].notna()
phys      = segdf[phys_mask]

mom_hours = phys.groupby("momentum")["dur_h"].sum()
cat_hours = segdf.groupby("category")["dur_h"].sum()



if DEBUG:
    print("\n===== All day-segments after splitting by date =====")
    print(segdf.fillna(""))
    print("\nTotal segments:", len(segdf))



DUMMY_DATE = pd.to_datetime("2000-01-01")

segdf["momentum"] = pd.to_numeric(segdf["momentum"], errors="coerce")

segdf.loc[~segdf["momentum"].isin(MOM_LIST), "momentum"] = np.nan

segdf["category"] = segdf["category"].fillna("Other")

segdf.loc[segdf["category"] != "Physics", "momentum"] = np.nan

segdf["start_hour"] = (
    segdf["seg_start"].dt.hour
    + segdf["seg_start"].dt.minute / 60.0
    + segdf["seg_start"].dt.second / 3600.0
)
segdf["end_hour"] = (
    segdf["seg_end"].dt.hour
    + segdf["seg_end"].dt.minute / 60.0
    + segdf["seg_end"].dt.second / 3600.0
)

mask_midnight = (segdf["end_hour"] == 0) & (segdf["seg_end"] > segdf["seg_start"])
segdf.loc[mask_midnight, "end_hour"] = 24.0

segdf["start_t"] = DUMMY_DATE + pd.to_timedelta(segdf["start_hour"], unit="h")
segdf["end_t"]   = DUMMY_DATE + pd.to_timedelta(segdf["end_hour"],   unit="h")
segdf["dur_t"]   = segdf["end_t"] - segdf["start_t"]

uniq_mom = sorted([m for m in segdf["momentum"].dropna().unique()])

valid_mom = sorted(segdf["momentum"].dropna().unique())

cmap = plt.colormaps["Paired"]

colors = [cmap(i) for i in range(len(valid_mom))]
mom_color = {m: c for m, c in zip(valid_mom, colors)}



def get_color(m):
    if pd.isna(m):
        return DUMMY_COLOR
    return mom_color.get(m, DUMMY_COLOR)

unique_dates = sorted(segdf["date"].unique())
n_days = len(unique_dates)


fig = plt.figure(figsize=(16, 9))

outer_gs = gridspec.GridSpec(
    nrows=1, ncols=2,
    width_ratios=[4, 1],
    wspace=0.25,
    figure=fig,
)

left_gs = outer_gs[0].subgridspec(nrows=n_days, ncols=1, hspace=0.1)

right_gs = outer_gs[1].subgridspec(
    nrows=2, ncols=1,
    height_ratios=[0.3, 0.7],   # 필요하면 조정
    hspace=0.3,
)

axes = [fig.add_subplot(left_gs[i, 0]) for i in range(n_days)]

for ax, d in zip(axes, unique_dates):
    sub = segdf[segdf["date"] == d]
    for _, r in sub.iterrows():
        #if r["category"] not in ["Physics", "Dummy", "Pedestal", "Other", "TPC_Dead"]:
        #    continue
        if r["category"] == "Dummy":
            bar_color = DUMMY_COLOR
        elif r["category"] == "Pedestal":
            bar_color = PEDESTAL_COLOR
        elif r["category"] == "Other":
            bar_color = OTHERS_COLOR
        elif r["category"] == "TPC_Dead":
            bar_color = TPC_DEAD_COLOR
        elif r["category"] == "Physics":
            bar_color = mom_color.get(r["momentum"], DUMMY_COLOR)
        else:
            bar_color = DUMMY_COLOR
        ax.barh(
            y=0,
            left=r["start_t"],
            width=r["dur_t"],
            height=0.6,
            color=bar_color,
            edgecolor="none"
        )

    ax.set_yticks([0])
    ax.set_yticklabels([pd.to_datetime(str(d)).strftime("%Y-%m-%d")])
    ax.set_xlim(DUMMY_DATE, DUMMY_DATE + pd.Timedelta(hours=24))
    ax.margins(x=0)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

locator = mdates.HourLocator(interval=2)   
formatter = mdates.DateFormatter("%H:%M")
axes[-1].xaxis.set_major_locator(locator)
axes[-1].xaxis.set_major_formatter(formatter)

plt.xlabel("Time")
fig.text(0.08, 0.9, "Date", ha="center", va="top", fontsize=10)
plt.suptitle("E72 Run Summary")

ax_leg = fig.add_subplot(right_gs[0, 0])
ax_leg.axis("off")


legend_elements = []

used_mom = (
    segdf.loc[segdf["category"]=="Physics", "momentum"]
    .dropna()
    .astype(float)
    .map(lambda x: f"{x:.3f}")
    .sort_values()
    .unique()
)

for m in used_mom:
    c = mom_color[float(m)]
    h = mom_hours.get(m, 0.0)   
    mom_mev = int(round(float(m) * 1000)) 
    legend_label = f"{mom_mev} MeV/c Physics Run ({h:.1f} h)"
    legend_elements.append(
        Patch(facecolor=c, edgecolor="none", label=legend_label)
    )

for cat_name, color, plain_label in [
    ("TPC_Dead",  TPC_DEAD_COLOR, "TPC DAQ Dead"),
    ("Other",  OTHERS_COLOR, "Commissioing"),
    #("Pedestal",  PEDESTAL_COLOR, "Pedestal"),
    ("Dummy",     DUMMY_COLOR,    "Dummy"),
]:
    h = cat_hours.get(cat_name, 0.0)
    legend_elements.append(
        Patch(
            facecolor=color,
            edgecolor="none",
            label=f"{plain_label} ({h:.1f} h)"
        )
    )
    

ax_leg.legend(
    handles=legend_elements,
    loc="upper left",
    fontsize=9,
    frameon=False,
)




ax_donut = fig.add_subplot(right_gs[1, 0])
ax_donut.set_xticks([])
ax_donut.set_yticks([])
ax_donut.set_frame_on(False)
for spine in ax_donut.spines.values():
    spine.set_visible(False)
ax_donut = fig.add_axes([0.65, 0.05, 0.36, 0.50])
ax_donut.axis("off")


outer_values = [actual_hours, lost_hours]
outer_labels = ["Beamtime", "Donwtime"]

delivered_h = segdf["dur_h"].sum()

plan_e72 = 312.0
plan_e45 = 12.0
planned_total = plan_e72 + plan_e45
lost_h = max(planned_total - delivered_h, 0.0)  

detail_labels = []
detail_sizes  = []
detail_colors = []


for m_str, h in mom_hours.items():
    if h <= 0:
        continue

    key = float(m_str)  # convert string -> float

    mom_mev = int(round(key * 1000))
    label = f"{mom_mev} MeV/c Physics"
    if key < 0.61:
        continue
    detail_labels.append(label)
    detail_sizes.append(h)
    detail_colors.append(mom_color[key])    

special_cats = ["Pedestal", "TPC_Dead", "Other", "Dummy"]
cat_color = {
    "TPC_Dead": TPC_DEAD_COLOR,
    "Pedestal": PEDESTAL_COLOR,
    "Dummy": DUMMY_COLOR,
    "Other": OTHERS_COLOR,
}

for cat in special_cats:
    h = float(cat_hours.get(cat, 0.0))
    if h <= 0:
        continue
    label = cat.replace("_", " ")
    detail_labels.append(label)
    detail_sizes.append(h)
    detail_colors.append(cat_color[cat])      

inner_total = sum(detail_sizes)
if abs(inner_total - delivered_h) > 1e-3:
    print(f"[WARN] inner breakdown sum {inner_total:.2f} h vs delivered {delivered_h:.2f} h")


outer_sizes = [delivered_h, lost_hours]
outer_labels = ["Beamtime", "Downtime"]

outer_radius = 1.0
ring_width  = 0.30

outer_wedges, _ = ax_donut.pie(
    outer_sizes,
    radius=outer_radius,
    labels=None,          
    colors=["lavender", "darkgray"],
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=ring_width, edgecolor="white")
)


inner_radius = outer_radius - ring_width
inner_sizes  = detail_sizes + [lost_hours]
inner_colors = detail_colors + [(0, 0, 0, 0)] 

detail_wedges, _ = ax_donut.pie(
    inner_sizes,
    radius=inner_radius,
    labels=None,
    colors=inner_colors,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=ring_width, edgecolor="white")
)


outer_mid_r = outer_radius - ring_width / 2.0

for w, label, value in zip(outer_wedges, outer_labels, outer_sizes):
    ang = 0.5 * (w.theta1 + w.theta2)
    ang_rad = np.deg2rad(ang)
    x = outer_mid_r * np.cos(ang_rad) * 1.1
    y = outer_mid_r * np.sin(ang_rad) * 1.1

    ax_donut.text(
        x, y,
        f"{label}\n{value/PLANNED_TOTAL*100:.1f}%\n({value:.1f} h)",
        ha="center", va="center",
        fontsize=10,
        color="black"
    )
    

inner_mid_r = inner_radius - ring_width / 2.0

total_delivered = delivered_h 
for w, label, h in zip(detail_wedges, detail_labels, detail_sizes):
    ang = 0.5 * (w.theta1 + w.theta2)
    ang_rad = np.deg2rad(ang)

    x = inner_mid_r * np.cos(ang_rad)
    y = inner_mid_r * np.sin(ang_rad)

    if "Physics" in label:
        mom = label.split()[0]




    
ax_donut.set(aspect="equal")

n_axes = len(axes)

for i, ax in enumerate(axes):
    show_xlabels = (i == n_axes - 1)
    ax.tick_params(axis="x", labelbottom=show_xlabels)



plt.tight_layout()
plt.savefig("e72_run_chart.pdf", dpi=300)
plt.show()
