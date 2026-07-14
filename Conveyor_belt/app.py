"""
Streamlit dashboard for the Smart Bag Counting & Classification PoC.

Simulates a multi-line camera monitoring wall: 4 fixed sample videos loop
continuously in the background (real detection running on every frame),
styled to match the target mockup. Line 3 periodically simulates a
conveyor jam. PoC only — fixed local video files stand in for live camera
feeds, no persistence, no real multi-camera hardware.
"""

import base64
import time

import plotly.graph_objects as go
import streamlit as st

from stream_worker import LineWorker

st.set_page_config(page_title="Multi-Line Bag Counting System", layout="wide", page_icon="📦", initial_sidebar_state="collapsed")

LINE_CONFIGS = [
    {"key": "line1", "label": "LINE 1", "video": "conveyorbelt_sample_1.mp4", "direction": "vertical", "simulate_jam": False},
    {"key": "line2", "label": "LINE 2", "video": "conveyorbelt_sample_2.mp4", "direction": "horizontal", "simulate_jam": False},
    {"key": "line3", "label": "LINE 3", "video": "conveyorbelt_sample_3.mp4", "direction": "vertical", "simulate_jam": True},
    {"key": "line4", "label": "LINE 4", "video": "conveyorbelt_sample_4.mp4", "direction": "vertical", "simulate_jam": False},
]

REFRESH_INTERVAL_SEC = 0.5

DARK_CSS = """
<style>
html, body, .stApp { background-color: #0a0f1a; color: #e6edf3; overflow: hidden; }
section[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { display: none; }
div.block-container {
    padding: 0.4rem 0.8rem 0.2rem 0.8rem;
    max-width: 100%; max-height: 100vh;
}
div[data-testid="stVerticalBlock"] { gap: 0.35rem; }
div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }

/* Header bar: one self-contained HTML block on the left, native buttons right */
.hdr {
    display: flex; align-items: center; gap: 18px; flex-wrap: nowrap;
    background: #10192b; border: 1px solid #24314a; border-radius: 6px;
    padding: 7px 14px; height: 5vh; min-height: 38px;
}
.hdr-title {
    font-size: 0.92rem; letter-spacing: 0.07em; color: #f0f4f8;
    font-weight: 700; white-space: nowrap; text-transform: uppercase;
}
.hdr-stat {
    font-size: 0.78rem; color: #b7c3d6; white-space: nowrap;
    padding-left: 16px; border-left: 1px solid #24314a;
}
.hdr-stat b { color: #f0f4f8; }
.status-ok { color: #3ddc84; font-weight: 700; }
.status-off { color: #8fa1bd; font-weight: 700; }

div[data-testid="stButton"] button {
    padding: 0.1rem 0.8rem; font-size: 0.78rem; min-height: 0; height: 2rem;
    border-radius: 5px;
}

/* Unified card treatment */
.line-panel, .side-panel {
    background: #10192b; border: 1px solid #24314a; border-radius: 6px;
    padding: 6px 8px;
}
.line-panel { position: relative; }
.line-panel.jam {
    border: 2px solid #e0433a;
    animation: pulse-red 1.4s ease-in-out infinite;
}
@keyframes pulse-red {
    0%   { box-shadow: 0 0 4px 0 rgba(224,67,58,0.25); }
    50%  { box-shadow: 0 0 16px 4px rgba(224,67,58,0.55); }
    100% { box-shadow: 0 0 4px 0 rgba(224,67,58,0.25); }
}
.line-title-row {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;
}
.line-title { font-size: 0.7rem; letter-spacing: 0.08em; color: #8fa1bd; font-weight: 700; text-transform: uppercase; }
.jam-badge {
    background: #3a1414; color: #ff6b60; border: 1px solid #e0433a; border-radius: 4px;
    padding: 0px 6px; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.03em;
}
.line-count-flour { color: #f2b84b; font-size: 0.95rem; font-weight: 700; }
.line-count-semolina { color: #4bb4f2; font-size: 0.95rem; font-weight: 700; }
.line-count-mixed { font-size: 0.9rem; font-weight: 700; color: #f0f4f8; }

/* Right rail: alerts + logs, internal scroll only */
.side-panel.alerts { height: 13vh; overflow-y: auto; }
.side-panel.logs { height: 30vh; overflow-y: auto; }
.side-title { font-size: 0.68rem; letter-spacing: 0.08em; color: #8fa1bd; text-transform: uppercase; margin-bottom: 4px; }
.alert-line { font-size: 0.68rem; padding: 1px 0; font-family: monospace; }
.alert-warn { color: #f2b84b; }
.alert-ok { color: #6f83a3; }
.log-line { font-size: 0.66rem; color: #b7c3d6; font-family: monospace; }
.log-time { color: #6f83a3; }

/* Chart cards via bordered containers, matching the card system */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #10192b; border: 1px solid #24314a !important; border-radius: 6px;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div { padding: 4px 8px !important; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


def get_workers():
    if "workers" not in st.session_state:
        st.session_state.workers = {
            cfg["key"]: LineWorker(
                name=cfg["label"],
                video_path=cfg["video"],
                direction=cfg["direction"],
                simulate_jam=cfg["simulate_jam"],
            )
            for cfg in LINE_CONFIGS
        }
    return st.session_state.workers


def get_alerts():
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    return st.session_state.alerts


workers = get_workers()
alerts = get_alerts()

if "running" not in st.session_state:
    st.session_state.running = False
if "prev_jam" not in st.session_state:
    st.session_state.prev_jam = {cfg["key"]: False for cfg in LINE_CONFIGS}

snapshots = {key: w.snapshot() for key, w in workers.items()}

grand_total = sum(s["total_count"] for s in snapshots.values())
active_lines = sum(1 for s in snapshots.values() if s["status"] == "running")
any_jam = any(s["jam_active"] for s in snapshots.values())

for cfg in LINE_CONFIGS:
    snap = snapshots[cfg["key"]]
    was_jam = st.session_state.prev_jam[cfg["key"]]
    if snap["jam_active"] and not was_jam:
        alerts.insert(0, {"text": f"{cfg['label']} Jam Detected", "level": "warn", "ts": time.strftime("%H:%M:%S")})
    elif was_jam and not snap["jam_active"]:
        alerts.insert(0, {"text": f"{cfg['label']} Jam Detected (Resolved)", "level": "ok", "ts": time.strftime("%H:%M:%S")})
    st.session_state.prev_jam[cfg["key"]] = snap["jam_active"]
alerts[:] = alerts[:8]

status_text = "OPERATIONAL" if st.session_state.running else "STOPPED"
status_class = "status-ok" if st.session_state.running else "status-off"

hdr_col, btn_col = st.columns([5, 1.1], vertical_alignment="center")
with hdr_col:
    st.markdown(
        f"""<div class="hdr">
<span class="hdr-title">Multi-Line Bag Counting System</span>
<span class="hdr-stat">GRAND TOTAL: <b>{grand_total:,} BAGS</b></span>
<span class="hdr-stat">ACTIVE LINES: <b>{active_lines}/4</b></span>
<span class="hdr-stat">STATUS: <span class="{status_class}">{status_text}</span></span>
</div>""",
        unsafe_allow_html=True,
    )
with btn_col:
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("Start", type="primary", width="stretch", disabled=st.session_state.running):
            for w in workers.values():
                w.start()
            st.session_state.running = True
            st.session_state.alerts = []
            st.rerun()
    with bc2:
        if st.button("Stop", width="stretch", disabled=not st.session_state.running):
            for w in workers.values():
                w.stop()
            st.session_state.running = False
            st.rerun()

grid_col, side_col = st.columns([3, 1])

with grid_col:
    row1 = st.columns(2)
    row2 = st.columns(2)
    line_slots = {
        "line1": row1[0], "line2": row1[1],
        "line3": row2[0], "line4": row2[1],
    }

    for cfg in LINE_CONFIGS:
        key = cfg["key"]
        snap = snapshots[key]
        with line_slots[key]:
            panel_class = "line-panel jam" if snap["jam_active"] else "line-panel"
            jam_badge = '<span class="jam-badge">⚠ JAM DETECTED</span>' if snap["jam_active"] else ""

            flour = snap["product_counts"].get("Flour", 0)
            semolina = snap["product_counts"].get("Semolina", 0)
            if flour and semolina:
                count_html = (
                    f'<span class="line-count-flour">FLOUR: {flour}</span>'
                    f'&nbsp;|&nbsp;<span class="line-count-semolina">SEMOLINA: {semolina}</span>'
                )
            elif flour:
                count_html = f'<span class="line-count-flour">FLOUR: {flour}</span>'
            elif semolina:
                count_html = f'<span class="line-count-semolina">SEMOLINA: {semolina}</span>'
            else:
                count_html = '<span class="line-count-mixed">COUNT: 0</span>'

            if snap["frame_jpeg"] is not None:
                b64 = base64.b64encode(snap["frame_jpeg"]).decode("ascii")
                image_html = (
                    f'<img src="data:image/jpeg;base64,{b64}" '
                    'style="width:100%;height:36vh;object-fit:cover;border-radius:4px;display:block;">'
                )
            else:
                image_html = (
                    '<div style="height:36vh;display:flex;align-items:center;justify-content:center;'
                    'color:#4a5a75;background:#0a1120;border-radius:4px;">Not running</div>'
                )

            panel_html = f"""<div class="{panel_class}">
<div class="line-title-row"><span class="line-title">{cfg['label']}</span>{jam_badge}</div>
<div style="margin-bottom:6px;">{count_html}</div>
{image_html}
</div>"""
            st.markdown(panel_html, unsafe_allow_html=True)

with side_col:
    if alerts:
        alert_lines = "".join(
            f'<div class="alert-line {"alert-warn" if a["level"] == "warn" else "alert-ok"}">{a["ts"]}  {a["text"]}</div>'
            for a in alerts
        )
    else:
        alert_lines = '<span style="color:#6f83a3;font-size:0.8rem;">No alerts</span>'
    st.markdown(
        f'<div class="side-panel alerts"><div class="side-title">Alerts</div>{alert_lines}</div>',
        unsafe_allow_html=True,
    )

    all_events = []
    for cfg in LINE_CONFIGS:
        snap = snapshots[cfg["key"]]
        for ev in snap["events"]:
            all_events.append((cfg["label"], ev))
    all_events.sort(key=lambda pair: pair[1]["elapsed_sec"], reverse=True)
    if all_events:
        log_lines = "".join(
            f'<div class="log-line"><span class="log-time">[{ev["elapsed_sec"]:>5.1f}s]</span> '
            f'{label} — {ev["product"]} {ev["size_label"]}</div>'
            for label, ev in all_events[:30]
        )
    else:
        log_lines = '<span style="color:#6f83a3;font-size:0.8rem;">No events yet</span>'
    st.markdown(
        f'<div class="side-panel logs"><div class="side-title">Logs</div>{log_lines}</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="side-title">Hourly count by line</div>', unsafe_allow_html=True)
        colors = {"line1": "#4bb4f2", "line2": "#f2b84b", "line3": "#3ddc84", "line4": "#e0433a"}
        fig = go.Figure()
        for cfg in LINE_CONFIGS:
            snap = snapshots[cfg["key"]]
            fig.add_trace(go.Bar(name=cfg["label"], x=["now"], y=[snap["total_count"]], marker_color=colors[cfg["key"]]))
        fig.update_layout(
            barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3", size=9), margin=dict(l=6, r=6, t=2, b=2),
            height=105, showlegend=False,
            xaxis=dict(gridcolor="#24314a"), yaxis=dict(gridcolor="#24314a"),
        )
        st.plotly_chart(fig, width="stretch", key=f"hourly_{grand_total}")

    with st.container(border=True):
        st.markdown('<div class="side-title">Product mix (total)</div>', unsafe_allow_html=True)
        total_flour = sum(s["product_counts"].get("Flour", 0) for s in snapshots.values())
        total_semolina = sum(s["product_counts"].get("Semolina", 0) for s in snapshots.values())
        if total_flour + total_semolina > 0:
            fig = go.Figure(data=[go.Pie(
                labels=["Flour", "Semolina"], values=[total_flour, total_semolina],
                marker=dict(colors=["#4bb4f2", "#f2b84b"]), hole=0.5, textinfo="percent",
            )])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3", size=9), margin=dict(l=6, r=6, t=2, b=2),
                height=105, legend=dict(orientation="v", x=1.0, font=dict(size=9)),
            )
            st.plotly_chart(fig, width="stretch", key=f"mix_{grand_total}")
        else:
            st.markdown('<span style="color:#6f83a3;font-size:0.8rem;">No bags counted yet</span>', unsafe_allow_html=True)

if st.session_state.running:
    time.sleep(REFRESH_INTERVAL_SEC)
    st.rerun()
