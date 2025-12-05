import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def make_autopct(sizes):
    def _autopct(pct):
        total = sum(sizes)
        hours = pct / 100.0 * total
        return f"{pct:.1f}%\n({hours:.1f} h)"
    return _autopct

CSV_FILE = "E72_downlog.csv"

df_raw = pd.read_csv(CSV_FILE, header=None)
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


print("\n===== All intervals (start–end pairs) =====")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

gdf_print = gdf.fillna("")
print(gdf_print[["start", "end", "start_note", "stop_note", "duration"]])
print("\nTotal intervals:", len(gdf))


segments = []
for _, r in gdf.iterrows():
    start = r["start"]
    end = r["end"]
    start_note = r["start_note"]
    stop_note = r["stop_note"]

    day = start.normalize()
    last_day = end.normalize()

    while day <= last_day:
        day_start = day
        day_end = day + pd.Timedelta(days=1)

        seg_start = max(start, day_start)
        seg_end = min(end, day_end)

        if seg_end > seg_start:
            segments.append({
                "date": day.date(),
                "seg_start": seg_start,
                "seg_end": seg_end,
                "start_note": start_note,
                "stop_note": stop_note,
            })

        day += pd.Timedelta(days=1)

segdf = pd.DataFrame(segments)
segdf["duration"] = segdf["seg_end"] - segdf["seg_start"]

print("\n===== All day-segments after splitting by date =====")
print(segdf.fillna(""))
print("\nTotal segments:", len(segdf))


DUMMY_DATE = pd.to_datetime("2000-01-01")

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

#Separation from the midnight
mask_midnight = (segdf["end_hour"] == 0) & (segdf["seg_end"] > segdf["seg_start"])
segdf.loc[mask_midnight, "end_hour"] = 24.0

segdf["start_t"] = DUMMY_DATE + pd.to_timedelta(segdf["start_hour"], unit="h")
segdf["end_t"]   = DUMMY_DATE + pd.to_timedelta(segdf["end_hour"], unit="h")
segdf["dur_t"]   = segdf["end_t"] - segdf["start_t"]

daily_dur = segdf.groupby("date")["duration"].sum()

unique_dates = sorted(segdf["date"].unique())
n = len(unique_dates)

fig_height = min(9, 0.4 * n + 2)
fig, axes = plt.subplots(n, 1, figsize=(13, fig_height), sharex=True)

if n == 1:
    axes = [axes]

for ax, d in zip(axes, unique_dates):
    sub = segdf[segdf["date"] == d]

    for _, r in sub.iterrows():
        ax.barh(
            y=0,
            width=r["dur_t"],
            left=r["start_t"],
            height=0.3,
            color="skyblue",  
        )

    ax.set_yticks([])

    date_str = pd.to_datetime(str(d)).strftime("%Y-%m-%d")
    total_sec = daily_dur[d].total_seconds()
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    dur_str = f"{hours:02d}:{minutes:02d} h"

    ax.set_ylabel(
        f"{date_str}\n{dur_str}",
        rotation=0,
        labelpad=35,
        fontsize=8,
        ha="right",
        va="center"
    )

    ax.set_xlim(DUMMY_DATE, DUMMY_DATE + pd.Timedelta(hours=24))
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

locator = mdates.HourLocator(interval=2)
formatter = mdates.DateFormatter("%H:%M")
axes[-1].xaxis.set_major_locator(locator)
axes[-1].xaxis.set_major_formatter(formatter)

plt.xlabel("Time")

fig.text(0.04, 0.98, "Date", ha="center", va="top", fontsize=10)

plt.suptitle("Run#93 K1.8BR Summary")
plt.tight_layout()
plt.savefig("K18BR_beam_timeline.pdf", format="pdf", dpi=300)



fig2, ax2 = plt.subplots(figsize=(7, 7))

outer_values = [E72_PLANNED, E45_PLANNED]
outer_labels = [f"E72 planned\n({E72_PLANNED} h)", f"E45 planned\n({E45_PLANNED} h)"]
outer_colors = ["#CFA7E8", "#A8E78B"]

inner_values = [actual_hours, lost_hours]
inner_labels = ["Beamtime", "Downtime"]
inner_colors = ["lavender", "darkgray"] 




# Outer ring
wedges_outer, _ = ax2.pie(
    outer_values,
    radius=1.3, colors=outer_colors, startangle=90,
    wedgeprops=dict(width=0.28, edgecolor="white")
)

# Inner ring
wedges_inner, _ = ax2.pie(
    inner_values, radius=1.0, colors=inner_colors, startangle=90,
    wedgeprops=dict(width=0.28, edgecolor="white")
)

ax2.axis("equal")

for wedge, label in zip(wedges_outer, outer_labels):

    ang = (wedge.theta2 + wedge.theta1) / 2.0
    ang_rad = np.deg2rad(ang)

    r_outer = wedge.r
    r_inner = wedge.r - wedge.width
    r_mid   = r_inner + wedge.width / 2.0

    x_start = r_mid * np.cos(ang_rad)
    y_start = r_mid * np.sin(ang_rad)

    x_text = (r_outer + 0.2) * np.cos(ang_rad)
    y_text = (r_outer + 0.2) * np.sin(ang_rad)

    ax2.annotate(
        label,
        xy=(x_start, y_start),
        xytext=(x_text, y_text),
        arrowprops=dict(arrowstyle="->", lw=1.4),
        ha="center",
        va="center",
        fontsize=20,
    )

for wedge, label, value in zip(wedges_inner, inner_labels, inner_values):
    ang = (wedge.theta2 + wedge.theta1) / 2.0
    ang_rad = np.deg2rad(ang)

    r_outer = wedge.r
    r_inner = wedge.r - wedge.width
    r_mid   = r_inner + wedge.width / 2.0

    x_start = r_mid * np.cos(ang_rad)
    y_start = r_mid * np.sin(ang_rad)

    x_text = (r_inner - 0.2) * np.cos(ang_rad)
    y_text = (r_inner - 0.2) * np.sin(ang_rad)

    ax2.annotate(
        f"{label}\n{value/PLANNED_TOTAL*100:.1f}%\n({value:.1f} h)",
        xy=(x_start, y_start),
        xytext=(x_text, y_text),
        #arrowprops=dict(arrowstyle="->", lw=1.4),
        ha="center",
        va="center",
        fontsize=20
    )


    ax2.set_title(
    "Run#93 K1.8BR Beamtime",
    fontsize=18,
    pad=50
)
plt.tight_layout()
plt.savefig("beamtime_donut_chart.pdf", dpi=300)
plt.show()

