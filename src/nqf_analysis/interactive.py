"""Create focused, presentation-ready standalone Plotly explorers."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .config import REMOTENESS_ORDER, STATE_ORDER, output_paths
from .spatial import prepare_projected_services


PAGE_CSS = r"""
:root {
  --ink: #102a43;
  --muted: #627d98;
  --line: #d9e2e7;
  --paper: #ffffff;
  --canvas: #f4f7f8;
  --navy: #16324f;
  --teal: #2a9d8f;
  --coral: #d1495b;
  --amber: #f4a261;
  --blue: #457b9d;
}
* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; background: var(--canvas); color: var(--ink); }
body { font-family: Inter, "Segoe UI", Arial, sans-serif; letter-spacing: 0; }
button { font: inherit; letter-spacing: 0; }
.page { width: min(1560px, 100%); margin: 0 auto; padding: 28px 34px 44px; }
.page-header { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin-bottom: 22px; }
.eyebrow { color: var(--teal); font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
h1 { margin: 5px 0 6px; font-size: clamp(26px, 3vw, 39px); line-height: 1.08; letter-spacing: 0; }
.subtitle { margin: 0; color: var(--muted); font-size: 15px; line-height: 1.5; max-width: 920px; }
.status { flex: 0 0 auto; color: var(--navy); background: #eaf1f5; border: 1px solid #cbd9e2; border-radius: 999px; padding: 8px 12px; font-size: 12px; font-weight: 750; }
.kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }
.kpi { min-height: 86px; background: var(--paper); border: 1px solid var(--line); border-radius: 7px; padding: 14px 16px; }
.kpi-label { color: var(--muted); font-size: 11px; font-weight: 750; text-transform: uppercase; }
.kpi-value { margin-top: 8px; color: var(--ink); font-size: 22px; line-height: 1.05; font-weight: 800; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px 24px; background: var(--paper); border: 1px solid var(--line); border-radius: 7px; padding: 12px 16px; margin-bottom: 12px; }
.control-row { display: flex; align-items: flex-start; gap: 14px; min-width: 0; }
.control-label { flex: 0 0 104px; padding-top: 7px; color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; }
.segments { display: flex; flex-wrap: wrap; gap: 6px; }
.segment { min-height: 31px; border: 1px solid #cbd9e2; border-radius: 5px; background: #fff; color: #334e68; padding: 6px 11px; font-size: 12px; font-weight: 700; cursor: pointer; transition: background .12s, border-color .12s, color .12s; }
.segment:hover { border-color: #829ab1; background: #f6f9fb; }
.segment.active { border-color: var(--navy); background: var(--navy); color: #fff; }
.segment:focus-visible { outline: 3px solid rgba(42,157,143,.28); outline-offset: 1px; }
.analysis-line { display: flex; align-items: center; min-height: 42px; gap: 10px; background: #edf6f5; border-left: 4px solid var(--teal); padding: 9px 13px; margin-bottom: 12px; color: #243b53; font-size: 13px; line-height: 1.45; }
.analysis-label { flex: 0 0 auto; color: #167d73; font-size: 11px; font-weight: 850; text-transform: uppercase; }
.chart-shell { position: relative; background: var(--paper); border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
.plotly-graph-div { width: 100% !important; }
.chart-note { display: flex; justify-content: space-between; gap: 16px; border-top: 1px solid var(--line); padding: 9px 14px; color: var(--muted); font-size: 11px; line-height: 1.4; }
.detail-band { display: grid; grid-template-columns: minmax(190px, .72fr) minmax(0, 2.28fr); gap: 18px; margin-top: 12px; background: var(--paper); border: 1px solid var(--line); border-radius: 7px; padding: 16px 18px; min-height: 102px; }
.detail-kicker { color: var(--teal); font-size: 11px; font-weight: 850; text-transform: uppercase; }
.detail-title { margin-top: 5px; font-size: 18px; line-height: 1.25; font-weight: 800; }
.detail-copy { color: #334e68; font-size: 13px; line-height: 1.5; }
.detail-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.detail-metric { border-left: 2px solid #bcccdc; padding: 2px 8px; min-width: 0; }
.detail-metric span { display: block; color: var(--muted); font-size: 10px; font-weight: 750; text-transform: uppercase; }
.detail-metric strong { display: block; margin-top: 4px; color: var(--ink); font-size: 14px; line-height: 1.25; overflow-wrap: anywhere; }
.badge { display: inline-flex; align-items: center; margin: 5px 5px 0 0; padding: 4px 7px; border-radius: 4px; background: #eef2f4; color: #486581; font-size: 10px; font-weight: 800; }
.badge.alert { background: #fae8eb; color: #a1283b; }
.map-key { display: flex; flex-wrap: wrap; gap: 10px; }
.map-key span::before { content: ""; display: inline-block; width: 9px; height: 9px; margin-right: 5px; border-radius: 2px; background: var(--swatch); }
@media (max-width: 980px) {
  .page { padding: 20px 16px 34px; }
  .page-header { align-items: flex-start; }
  .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .kpi:last-child { grid-column: 1 / -1; }
  .detail-band { grid-template-columns: 1fr; }
  .detail-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .controls { display: block; }
  .control-row { margin-bottom: 9px; }
  .control-row:last-child { margin-bottom: 0; }
}
@media (max-width: 620px) {
  .page-header { display: block; }
  .status { display: inline-block; margin-top: 12px; }
  .control-row { display: block; }
  .control-label { padding: 0 0 6px; }
  .kpis { grid-template-columns: 1fr 1fr; }
  .kpi { min-height: 76px; padding: 12px; }
  .kpi-value { font-size: 19px; }
  .detail-grid { grid-template-columns: 1fr 1fr; }
  .chart-note { display: block; }
}
"""


REMOTENESS_SHORT = {
    "Major Cities of Australia": "Major Cities",
    "Inner Regional Australia": "Inner Regional",
    "Outer Regional Australia": "Outer Regional",
    "Remote Australia": "Remote",
    "Very Remote Australia": "Very Remote",
}

REMOTENESS_FILL = {
    "Major Cities of Australia": "rgba(69,123,157,0.15)",
    "Inner Regional Australia": "rgba(42,157,143,0.15)",
    "Outer Regional Australia": "rgba(242,207,91,0.16)",
    "Remote Australia": "rgba(244,162,97,0.16)",
    "Very Remote Australia": "rgba(209,73,91,0.16)",
}


def _json_records(frame, columns=None, rename=None):
    """Serialize a compact, browser-safe record list."""
    selected = frame.copy() if columns is None else frame.loc[:, columns].copy()
    if rename:
        selected = selected.rename(columns=rename)
    return selected.to_json(orient="records", double_precision=6).replace("</", "<\\/")


def _button_group(group, options, active):
    buttons = []
    for value, label in options:
        active_class = " active" if value == active else ""
        pressed = "true" if value == active else "false"
        buttons.append(
            '<button class="segment{}" type="button" data-group="{}" '
            'data-value="{}" aria-pressed="{}">{}</button>'.format(
                active_class, group, value, pressed, label
            )
        )
    return '<div class="segments">{}</div>'.format("".join(buttons))


def _control_row(label, buttons):
    return (
        '<div class="control-row"><div class="control-label">{}</div>{}</div>'.format(
            label, buttons
        )
    )


def _kpis(items):
    return '<section class="kpis">{}</section>'.format(
        "".join(
            '<div class="kpi"><div class="kpi-label">{}</div>'
            '<div class="kpi-value" id="{}">{}</div></div>'.format(label, item_id, value)
            for item_id, label, value in items
        )
    )


def _plot_fragment(figure, div_id):
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=True,
        div_id=div_id,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        },
    )


def _write_page(
    path,
    title,
    subtitle,
    stage,
    kpis,
    controls,
    analysis_id,
    figure,
    div_id,
    detail_html,
    script,
    chart_note,
):
    html = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>{}</title><style>{}</style></head><body>".format(title, PAGE_CSS)
        + '<main class="page"><header class="page-header"><div>'
        + '<div class="eyebrow">ACECQA decision explorer</div><h1>{}</h1>'.format(title)
        + '<p class="subtitle">{}</p></div><div class="status">{}</div></header>'.format(
            subtitle, stage
        )
        + kpis
        + '<section class="controls">{}</section>'.format(controls)
        + '<div class="analysis-line"><span class="analysis-label">Reading</span>'
        + '<span id="{}"></span></div>'.format(analysis_id)
        + '<section class="chart-shell">{}<div class="chart-note">{}</div></section>'.format(
            _plot_fragment(figure, div_id), chart_note
        )
        + detail_html
        + "</main><script>{}</script></body></html>".format(script)
    )
    Path(path).write_text(html, encoding="utf-8")


def _iter_polygons(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return []


def _add_boundary_layers(figure, boundaries):
    """Add restrained remoteness fills and stronger state outlines."""
    areas = boundaries.to_crs("EPSG:4326").copy()
    areas["geometry"] = areas.geometry.simplify(0.025, preserve_topology=True)
    for remoteness in REMOTENESS_ORDER:
        x_values = []
        y_values = []
        selected = areas[areas["RA_NAME21"] == remoteness]
        for geometry in selected.geometry:
            for polygon in _iter_polygons(geometry):
                x, y = polygon.exterior.xy
                x_values.extend(x)
                y_values.extend(y)
                x_values.append(None)
                y_values.append(None)
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                fill="toself",
                fillcolor=REMOTENESS_FILL[remoteness],
                line={"color": "rgba(98,125,152,0.25)", "width": 0.55},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    states = areas.dissolve(by="boundary_state_abbr")
    state_x = []
    state_y = []
    for geometry in states.geometry:
        for polygon in _iter_polygons(geometry):
            x, y = polygon.exterior.xy
            state_x.extend(x)
            state_y.extend(y)
            state_x.append(None)
            state_y.append(None)
    figure.add_trace(
        go.Scatter(
            x=state_x,
            y=state_y,
            mode="lines",
            line={"color": "rgba(16,42,67,0.72)", "width": 1.05},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _service_landscape_page(data, boundaries, output_path):
    projected = prepare_projected_services(data).copy()
    projected["grid_x"] = np.floor(projected.geometry.x / 50000).astype(int)
    projected["grid_y"] = np.floor(projected.geometry.y / 50000).astype(int)

    columns = [
        "ServiceApprovalNumber",
        "ServiceName",
        "ServiceType",
        "State",
        "Suburb",
        "remoteness_area",
        "longitude",
        "latitude",
        "OverallRating",
        "is_rated",
        "is_below_nqs",
        "nearest_transport_distance_km",
        "NumberOfApprovedPlaces",
        "annual_weekly_operating_hours",
        "detailed_offering_combination",
        "grid_x",
        "grid_y",
    ]
    rename = {
        "ServiceApprovalNumber": "id",
        "ServiceName": "name",
        "ServiceType": "type",
        "State": "state",
        "Suburb": "suburb",
        "remoteness_area": "remote",
        "longitude": "lon",
        "latitude": "lat",
        "OverallRating": "rating",
        "is_rated": "rated",
        "is_below_nqs": "below",
        "nearest_transport_distance_km": "transport",
        "NumberOfApprovedPlaces": "capacity",
        "annual_weekly_operating_hours": "hours",
        "detailed_offering_combination": "offering",
        "grid_x": "gx",
        "grid_y": "gy",
    }
    service_json = _json_records(projected, columns, rename)

    figure = go.Figure()
    _add_boundary_layers(figure, boundaries)
    figure.add_trace(
        go.Scattergl(
            x=[],
            y=[],
            mode="markers",
            marker={"size": 10, "color": "#457B9D", "opacity": 0.82},
            hoverinfo="text",
            showlegend=False,
        )
    )
    marker_trace_index = len(figure.data) - 1
    figure.update_layout(
        height=650,
        margin={"l": 16, "r": 16, "t": 12, "b": 10},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFB",
        hoverlabel={"bgcolor": "#102A43", "font": {"color": "white", "size": 12}},
        dragmode="pan",
        xaxis={
            "range": [111.0, 155.5],
            "visible": False,
            "fixedrange": False,
            "constrain": "domain",
        },
        yaxis={
            "range": [-45.0, -9.0],
            "visible": False,
            "fixedrange": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        uirevision="service-landscape",
    )

    controls = "".join(
        [
            _control_row(
                "Map layer",
                _button_group(
                    "layer",
                    [("cluster", "50 km overview"), ("service", "Service locations")],
                    "cluster",
                ),
            ),
            _control_row(
                "Measure",
                _button_group(
                    "metric",
                    [
                        ("count", "Service volume"),
                        ("quality", "Quality"),
                        ("transport", "Transport"),
                        ("capacity", "Capacity"),
                    ],
                    "count",
                ),
            ),
            _control_row(
                "Service type",
                _button_group(
                    "type",
                    [
                        ("All", "All services"),
                        ("Centre-Based Care", "Centre-Based Care"),
                        ("Family Day Care", "Family Day Care"),
                    ],
                    "All",
                ),
            ),
            _control_row(
                "State",
                _button_group(
                    "state", [("All", "Australia")] + [(s, s) for s in STATE_ORDER], "All"
                ),
            ),
            _control_row(
                "Remoteness",
                _button_group(
                    "remote",
                    [("All", "All areas")]
                    + [(value, REMOTENESS_SHORT[value]) for value in REMOTENESS_ORDER],
                    "All",
                ),
            ),
        ]
    )
    kpis = _kpis(
        [
            ("map-services", "Mapped services", "-"),
            ("map-rated", "Rating coverage", "-"),
            ("map-below", "Below NQS", "-"),
            ("map-transport", "Median to station", "-"),
            ("map-capacity", "Median capacity", "-"),
        ]
    )
    detail = r"""
<section class="detail-band" id="map-detail">
  <div><div class="detail-kicker">Selected geography</div><div class="detail-title">Australia</div></div>
  <div class="detail-copy">Select a 50 km cell or service marker to inspect its quality, accessibility and operating profile.</div>
</section>
"""
    note = (
        '<div id="map-legend">Marker size represents service volume.</div>'
        '<div class="map-key"><span style="--swatch:#457b9d">Major Cities</span>'
        '<span style="--swatch:#2a9d8f">Regional</span>'
        '<span style="--swatch:#d1495b">Remote</span></div>'
    )

    script = r"""
const SERVICES = __SERVICE_DATA__;
const MAP_TRACE = __MAP_TRACE__;
const gd = document.getElementById('service-map');
const selection = {layer: 'cluster', metric: 'count', type: 'All', state: 'All', remote: 'All'};
let currentMarks = [];

const finite = v => v !== null && v !== undefined && Number.isFinite(Number(v));
const fmt = (v, digits=1) => finite(v) ? Number(v).toLocaleString('en-AU', {minimumFractionDigits: digits, maximumFractionDigits: digits}) : 'Not available';
const countFmt = v => Number(v || 0).toLocaleString('en-AU');
const escapeHtml = value => String(value ?? 'Not available').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
function median(values) {
  const clean = values.filter(finite).map(Number).sort((a,b) => a-b);
  if (!clean.length) return null;
  const middle = Math.floor(clean.length / 2);
  return clean.length % 2 ? clean[middle] : (clean[middle - 1] + clean[middle]) / 2;
}
function filteredServices() {
  return SERVICES.filter(d =>
    (selection.type === 'All' || d.type === selection.type) &&
    (selection.state === 'All' || d.state === selection.state) &&
    (selection.remote === 'All' || d.remote === selection.remote)
  );
}
function summary(rows) {
  const rated = rows.filter(d => d.rated === true).length;
  const below = rows.filter(d => d.rated === true && d.below === true).length;
  return {
    n: rows.length,
    rated,
    coverage: rows.length ? rated / rows.length * 100 : null,
    belowPct: rated ? below / rated * 100 : null,
    transport: median(rows.map(d => d.transport)),
    capacity: median(rows.map(d => d.capacity)),
    hours: median(rows.map(d => d.hours))
  };
}
function cluster(rows) {
  const groups = new Map();
  rows.forEach(d => {
    const key = `${d.gx}|${d.gy}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(d);
  });
  return Array.from(groups, ([key, members]) => {
    const s = summary(members);
    const centres = members.filter(d => d.type === 'Centre-Based Care').length;
    return {
      kind: 'cluster', key, members, ...s,
      lon: members.reduce((a,d) => a + Number(d.lon), 0) / members.length,
      lat: members.reduce((a,d) => a + Number(d.lat), 0) / members.length,
      centrePct: members.length ? centres / members.length * 100 : null
    };
  });
}
function metricStyle(marks) {
  if (selection.metric === 'count') {
    if (selection.layer === 'service') {
      return {color: marks.map(d => d.type === 'Family Day Care' ? '#e76f51' : '#457b9d'), showscale: false};
    }
    return {color: marks.map(d => d.n), colorscale: [[0,'#8ec6df'],[.45,'#367f9e'],[1,'#102a43']], cmin: 1, cmax: Math.max(10, ...marks.map(d => d.n)), showscale: true, title: 'Services'};
  }
  if (selection.metric === 'quality') {
    if (selection.layer === 'service') {
      return {color: marks.map(d => !d.rated ? '#a7b1b8' : d.below ? '#d1495b' : '#2a9d8f'), showscale: false};
    }
    return {color: marks.map(d => d.belowPct), colorscale: [[0,'#2a9d8f'],[.45,'#f2cf5b'],[1,'#d1495b']], cmin: 0, cmax: 35, showscale: true, title: 'Below NQS %'};
  }
  if (selection.metric === 'transport') {
    return {color: marks.map(d => finite(d.transport) ? Math.log10(Math.max(.03, Number(d.transport))) : null), colorscale: [[0,'#55a88f'],[.42,'#f2cf5b'],[.72,'#f08a5d'],[1,'#a1283b']], cmin: -1, cmax: 3, showscale: true, title: 'Distance (log km)', tickvals: [-1,0,1,2,3], ticktext: ['0.1','1','10','100','1,000']};
  }
  return {color: marks.map(d => d.capacity), colorscale: [[0,'#a9c5d6'],[.35,'#58a89f'],[.7,'#367694'],[1,'#102a43']], cmin: 0, cmax: 150, showscale: true, title: 'Places'};
}
function hoverText(mark) {
  if (mark.kind === 'cluster') {
    return `<b>50 km service area</b><br>${countFmt(mark.n)} services | ${fmt(mark.coverage)}% rated<br>` +
      `Below NQS: ${fmt(mark.belowPct)}% of ${countFmt(mark.rated)} rated<br>` +
      `Median station distance: ${fmt(mark.transport)} km<br>Median capacity: ${fmt(mark.capacity,0)} places`;
  }
  return `<b>${escapeHtml(mark.name)}</b><br>${escapeHtml(mark.suburb)}, ${escapeHtml(mark.state)}<br>` +
    `${escapeHtml(mark.type)} | ${escapeHtml(mark.rating)}<br>` +
    `Station distance: ${fmt(mark.transport)} km<br>Capacity: ${fmt(mark.capacity,0)} places`;
}
function updateKpis(rows) {
  const s = summary(rows);
  document.getElementById('map-services').textContent = countFmt(s.n);
  document.getElementById('map-rated').textContent = finite(s.coverage) ? `${fmt(s.coverage)}%` : 'n/a';
  document.getElementById('map-below').textContent = finite(s.belowPct) ? `${fmt(s.belowPct)}%` : 'n/a';
  document.getElementById('map-transport').textContent = finite(s.transport) ? `${fmt(s.transport)} km` : 'n/a';
  document.getElementById('map-capacity').textContent = finite(s.capacity) ? fmt(s.capacity,0) : 'n/a';
}
function updateReading(rows) {
  const xr = gd._fullLayout?.xaxis?.range || [111,155.5];
  const yr = gd._fullLayout?.yaxis?.range || [-45,-9];
  const visible = rows.filter(d => d.lon >= Math.min(...xr) && d.lon <= Math.max(...xr) && d.lat >= Math.min(...yr) && d.lat <= Math.max(...yr));
  const all = summary(rows), local = summary(visible);
  const geography = visible.length === rows.length ? 'Selected population' : 'Visible map extent';
  const comparison = finite(local.belowPct) && finite(all.belowPct) && visible.length !== rows.length ? ` versus ${fmt(all.belowPct)}% across the current filters` : '';
  document.getElementById('map-analysis').textContent = visible.length
    ? `${geography}: ${countFmt(local.n)} services; ${fmt(local.coverage)}% have a current rating, ${fmt(local.belowPct)}% of rated services are below NQS${comparison}, and median station distance is ${fmt(local.transport)} km.`
    : 'No mapped services fall inside the current map extent and filters.';
}
function fitRows(rows) {
  if (!rows.length) return;
  if (selection.state === 'All' && selection.remote === 'All') {
    Plotly.relayout(gd, {'xaxis.range': [111,155.5], 'yaxis.range': [-45,-9]});
    return;
  }
  const xs = rows.map(d => Number(d.lon)), ys = rows.map(d => Number(d.lat));
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const xpad = Math.max(1.2, (xmax-xmin)*.12), ypad = Math.max(.8, (ymax-ymin)*.12);
  Plotly.relayout(gd, {'xaxis.range': [xmin-xpad,xmax+xpad], 'yaxis.range': [ymin-ypad,ymax+ypad]});
}
function showDetail(mark) {
  const root = document.getElementById('map-detail');
  if (mark.kind === 'cluster') {
    root.innerHTML = `<div><div class="detail-kicker">Selected 50 km cell</div><div class="detail-title">${countFmt(mark.n)} services</div></div>` +
      `<div class="detail-grid"><div class="detail-metric"><span>Rating coverage</span><strong>${fmt(mark.coverage)}%</strong></div>` +
      `<div class="detail-metric"><span>Below NQS</span><strong>${fmt(mark.belowPct)}% of rated</strong></div>` +
      `<div class="detail-metric"><span>Station distance</span><strong>${fmt(mark.transport)} km median</strong></div>` +
      `<div class="detail-metric"><span>Capacity</span><strong>${fmt(mark.capacity,0)} places median</strong></div>` +
      `<div class="detail-metric"><span>Centre-based mix</span><strong>${fmt(mark.centrePct)}%</strong></div>` +
      `<div class="detail-metric"><span>Weekly hours</span><strong>${fmt(mark.hours)} h median</strong></div></div>`;
  } else {
    root.innerHTML = `<div><div class="detail-kicker">Selected service</div><div class="detail-title">${escapeHtml(mark.name)}</div></div>` +
      `<div class="detail-grid"><div class="detail-metric"><span>Location</span><strong>${escapeHtml(mark.suburb)}, ${escapeHtml(mark.state)}</strong></div>` +
      `<div class="detail-metric"><span>Service type</span><strong>${escapeHtml(mark.type)}</strong></div>` +
      `<div class="detail-metric"><span>Overall rating</span><strong>${escapeHtml(mark.rating)}</strong></div>` +
      `<div class="detail-metric"><span>Remoteness</span><strong>${escapeHtml(mark.remote)}</strong></div>` +
      `<div class="detail-metric"><span>Station distance</span><strong>${fmt(mark.transport)} km</strong></div>` +
      `<div class="detail-metric"><span>Capacity</span><strong>${fmt(mark.capacity,0)} places</strong></div>` +
      `<div class="detail-metric"><span>Offering</span><strong>${escapeHtml(mark.offering)}</strong></div>` +
      `<div class="detail-metric"><span>Weekly hours</span><strong>${fmt(mark.hours)} h</strong></div></div>`;
  }
}
function updateMap(resetView=false) {
  const rows = filteredServices();
  currentMarks = selection.layer === 'cluster' ? cluster(rows) : rows.map(d => ({...d, kind:'service'}));
  const style = metricStyle(currentMarks);
  const sizes = selection.layer === 'cluster' ? currentMarks.map(d => Math.max(8, Math.min(42, 5 + Math.sqrt(d.n)*3.3))) : currentMarks.map(() => 6.5);
  const opacity = selection.layer === 'cluster' ? .92 : .8;
  Plotly.restyle(gd, {
    x: [currentMarks.map(d => d.lon)], y: [currentMarks.map(d => d.lat)],
    text: [currentMarks.map(hoverText)], hovertemplate: ['%{text}<extra></extra>'],
    'marker.size': [sizes], 'marker.color': [style.color], 'marker.opacity': [opacity],
    'marker.colorscale': [style.colorscale || null], 'marker.cmin': [style.cmin ?? null], 'marker.cmax': [style.cmax ?? null],
    'marker.showscale': [style.showscale],
    'marker.colorbar': [{title: {text: style.title || '', side: 'right'}, thickness: 12, len: .52, x: 1.01, tickvals: style.tickvals, ticktext: style.ticktext, outlinewidth: 0}],
    'marker.line': [{color: selection.layer === 'cluster' ? '#ffffff' : 'rgba(255,255,255,.75)', width: selection.layer === 'cluster' ? 1.1 : .45}]
  }, [MAP_TRACE]);
  document.getElementById('map-legend').textContent = selection.layer === 'cluster'
    ? 'Equal-area 50 km cells; marker size represents service volume. Percentages use rated services as the denominator.'
    : 'Each marker is one service. Grey quality markers have no current overall rating.';
  updateKpis(rows);
  if (resetView) fitRows(rows); else updateReading(rows);
}
document.querySelectorAll('.segment').forEach(button => button.addEventListener('click', () => {
  const group = button.dataset.group;
  selection[group] = button.dataset.value;
  document.querySelectorAll(`[data-group="${group}"]`).forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', b === button ? 'true' : 'false'); });
  updateMap(group === 'state' || group === 'remote');
}));
gd.on('plotly_click', event => { const index = event.points?.[0]?.pointNumber; if (index !== undefined && currentMarks[index]) showDetail(currentMarks[index]); });
gd.on('plotly_relayout', () => updateReading(filteredServices()));
updateMap(false);
""".replace("__SERVICE_DATA__", service_json).replace(
        "__MAP_TRACE__", str(marker_trace_index)
    )

    _write_page(
        output_path,
        "Service landscape",
        "Where services are located, who they serve, and how quality, transport access and capacity change across geography.",
        "1 | National landscape",
        kpis,
        controls,
        "map-analysis",
        figure,
        "service-map",
        detail,
        script,
        note,
    )


def _quality_page(tables, output_path):
    quality = tables["filtered_quality_benchmark"].copy()
    columns = [
        "service_filter",
        "services_in_filter",
        "state",
        "measure",
        "rated_services",
        "meeting_or_above_pct",
        "filter_national_meeting_or_above_pct",
        "gap_from_filter_national_pp",
    ]
    quality_json = _json_records(quality, columns)
    initial = quality[quality["service_filter"] == "All services"]
    measures = ["Overall"] + ["QA{}".format(i) for i in range(1, 8)]
    pivot = initial.pivot(index="state", columns="measure", values="gap_from_filter_national_pp")
    pivot = pivot.reindex(index=STATE_ORDER, columns=measures)

    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=measures,
            y=STATE_ORDER,
            colorscale=[[0, "#B83245"], [0.5, "#F7F9FA"], [1, "#2A9D8F"]],
            zmid=0,
            zmin=-15,
            zmax=15,
            text=np.where(np.isfinite(pivot.to_numpy()), np.char.add(np.round(pivot.to_numpy(), 1).astype(str), " pp"), ""),
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar={"title": {"text": "Gap (pp)"}, "thickness": 13, "len": 0.72, "outlinewidth": 0},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        height=600,
        margin={"l": 60, "r": 28, "t": 22, "b": 54},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis={"side": "top", "tickfont": {"size": 12, "color": "#334E68"}, "fixedrange": True},
        yaxis={"autorange": "reversed", "tickfont": {"size": 12, "color": "#334E68"}, "fixedrange": True},
        hoverlabel={"bgcolor": "#102A43", "font": {"color": "white"}},
    )
    controls = "".join(
        [
            _control_row(
                "Service type",
                _button_group(
                    "scope",
                    [
                        ("All services", "All services"),
                        ("Centre-Based Care", "Centre-Based Care"),
                        ("Family Day Care", "Family Day Care"),
                    ],
                    "All services",
                ),
            ),
            _control_row(
                "Display",
                _button_group(
                    "mode",
                    [("gap", "Gap from benchmark"), ("rate", "Meeting NQS or above")],
                    "gap",
                ),
            ),
        ]
    )
    kpis = _kpis(
        [
            ("quality-services", "Services in scope", "-"),
            ("quality-national", "National overall", "-"),
            ("quality-lowest", "Lowest state overall", "-"),
            ("quality-gap", "Largest QA gap", "-"),
            ("quality-rated", "Rated in lowest state", "-"),
        ]
    )
    detail = r"""
<section class="detail-band" id="quality-detail">
  <div><div class="detail-kicker">Selected comparison</div><div class="detail-title">Click a matrix cell</div></div>
  <div class="detail-copy">Each cell compares one state and quality measure with the national benchmark for the selected service type.</div>
</section>
"""
    note = (
        "Green indicates performance above the within-service-type national benchmark; red indicates below. "
        "Missing cells are not imputed. QA = National Quality Standard quality area."
    )
    script = r"""
const QUALITY = __QUALITY_DATA__;
const gd = document.getElementById('quality-heatmap');
const states = ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA'];
const measures = ['Overall','QA1','QA2','QA3','QA4','QA5','QA6','QA7'];
const names = {Overall:'Overall rating', QA1:'Educational program and practice', QA2:"Children's health and safety", QA3:'Physical environment', QA4:'Staffing arrangements', QA5:'Relationships with children', QA6:'Collaborative partnerships', QA7:'Governance and leadership'};
const selection = {scope:'All services', mode:'gap'};
const finite = v => v !== null && v !== undefined && Number.isFinite(Number(v));
const fmt = (v,d=1) => finite(v) ? Number(v).toLocaleString('en-AU',{minimumFractionDigits:d,maximumFractionDigits:d}) : 'n/a';
const countFmt = v => Number(v||0).toLocaleString('en-AU');
function currentRows() { return QUALITY.filter(d => d.service_filter === selection.scope); }
function findCell(rows, state, measure) { return rows.find(d => d.state === state && d.measure === measure); }
function updateQuality() {
  const rows = currentRows();
  const matrix = states.map(state => measures.map(measure => {
    const d = findCell(rows,state,measure);
    return d ? (selection.mode === 'gap' ? d.gap_from_filter_national_pp : d.meeting_or_above_pct) : null;
  }));
  const text = matrix.map(row => row.map(v => finite(v) ? `${fmt(v)}${selection.mode === 'gap' ? ' pp' : '%'}` : ''));
  const custom = states.map(state => measures.map(measure => {
    const d = findCell(rows,state,measure);
    return d ? [d.state, d.measure, d.rated_services, d.meeting_or_above_pct, d.filter_national_meeting_or_above_pct, d.gap_from_filter_national_pp] : null;
  }));
  const gapScale = [[0,'#b83245'],[.5,'#f7f9fa'],[1,'#2a9d8f']];
  const rateScale = [[0,'#b83245'],[.48,'#f2cf5b'],[1,'#2a9d8f']];
  Plotly.react(gd, [{
    type:'heatmap', z:matrix, x:measures, y:states, text, texttemplate:'%{text}', textfont:{size:11}, customdata:custom,
    colorscale: selection.mode === 'gap' ? gapScale : rateScale,
    zmid: selection.mode === 'gap' ? 0 : undefined, zmin: selection.mode === 'gap' ? -15 : 70, zmax: selection.mode === 'gap' ? 15 : 100,
    colorbar:{title:{text: selection.mode === 'gap' ? 'Gap (pp)' : 'Meeting+ %'},thickness:13,len:.72,outlinewidth:0},
    hovertemplate:'<b>%{customdata[0]} | %{customdata[1]}</b><br>Meeting NQS or above: %{customdata[3]:.1f}%<br>National benchmark: %{customdata[4]:.1f}%<br>Gap: %{customdata[5]:+.1f} pp<br>Rated services: %{customdata[2]:,.0f}<extra></extra>'
  }], {
    height:600, margin:{l:60,r:28,t:22,b:54}, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    xaxis:{side:'top',tickfont:{size:12,color:'#334e68'},fixedrange:true},
    yaxis:{autorange:'reversed',tickfont:{size:12,color:'#334e68'},fixedrange:true},
    hoverlabel:{bgcolor:'#102a43',font:{color:'#fff'}}
  }, {responsive:true,displaylogo:false});
  const overall = rows.filter(d => d.measure === 'Overall');
  const national = overall[0]?.filter_national_meeting_or_above_pct;
  const lowest = overall.reduce((a,b) => !a || Number(b.meeting_or_above_pct) < Number(a.meeting_or_above_pct) ? b : a, null);
  const biggest = rows.reduce((a,b) => !a || Math.abs(Number(b.gap_from_filter_national_pp)) > Math.abs(Number(a.gap_from_filter_national_pp)) ? b : a, null);
  document.getElementById('quality-services').textContent = countFmt(rows[0]?.services_in_filter);
  document.getElementById('quality-national').textContent = `${fmt(national)}%`;
  document.getElementById('quality-lowest').textContent = lowest ? `${lowest.state} | ${fmt(lowest.meeting_or_above_pct)}%` : 'n/a';
  document.getElementById('quality-gap').textContent = biggest ? `${biggest.state} ${biggest.measure} | ${fmt(biggest.gap_from_filter_national_pp)} pp` : 'n/a';
  document.getElementById('quality-rated').textContent = lowest ? countFmt(lowest.rated_services) : 'n/a';
  const direction = Number(biggest?.gap_from_filter_national_pp) < 0 ? 'below' : 'above';
  document.getElementById('quality-analysis').textContent = lowest && biggest
    ? `${selection.scope}: ${lowest.state} has the lowest overall Meeting NQS-or-above rate (${fmt(lowest.meeting_or_above_pct)}%). The largest state-quality-area departure is ${biggest.state} ${biggest.measure}, ${fmt(Math.abs(biggest.gap_from_filter_national_pp))} percentage points ${direction} its national benchmark.`
    : 'No comparable quality observations are available for this selection.';
}
function showDetail(point) {
  const state = states[point.pointNumber[0]], measure = measures[point.pointNumber[1]];
  const d = findCell(currentRows(), state, measure);
  if (!d) return;
  const direction = Number(d.gap_from_filter_national_pp) >= 0 ? 'above' : 'below';
  document.getElementById('quality-detail').innerHTML = `<div><div class="detail-kicker">Selected comparison</div><div class="detail-title">${state} | ${names[measure]}</div></div>` +
    `<div class="detail-grid"><div class="detail-metric"><span>Meeting NQS or above</span><strong>${fmt(d.meeting_or_above_pct)}%</strong></div>` +
    `<div class="detail-metric"><span>National benchmark</span><strong>${fmt(d.filter_national_meeting_or_above_pct)}%</strong></div>` +
    `<div class="detail-metric"><span>Benchmark gap</span><strong>${fmt(Math.abs(d.gap_from_filter_national_pp))} pp ${direction}</strong></div>` +
    `<div class="detail-metric"><span>Rated denominator</span><strong>${countFmt(d.rated_services)} services</strong></div></div>`;
}
document.querySelectorAll('.segment').forEach(button => button.addEventListener('click', () => {
  const group = button.dataset.group; selection[group] = button.dataset.value;
  document.querySelectorAll(`[data-group="${group}"]`).forEach(b => {b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',b===button?'true':'false');});
  updateQuality();
}));
gd.on('plotly_click', e => { if (e.points?.[0]) showDetail(e.points[0]); });
updateQuality();
""".replace("__QUALITY_DATA__", quality_json)
    _write_page(
        output_path,
        "Quality geography",
        "Separate genuine geographic quality differences from differences in the mix of centre-based and family day care services.",
        "2 | Quality",
        kpis,
        controls,
        "quality-analysis",
        figure,
        "quality-heatmap",
        detail,
        script,
        note,
    )


def _population_coverage_page(tables, boundaries, output_path):
    coverage = tables["population_coverage_by_state_remoteness"].copy()
    coverage_columns = [
        "state",
        "remoteness_area",
        "remoteness_code",
        "child_population_0_13",
        "all_services",
        "centre_based_services",
        "centre_based_approved_places",
        "all_services_per_1000_children",
        "centre_based_services_per_1000_children",
        "approved_places_per_1000_children",
        "approved_places_per_centre_service",
        "national_approved_places_per_1000_children",
        "small_population_flag",
    ]
    coverage_json = _json_records(coverage, coverage_columns)
    areas = boundaries[
        boundaries["boundary_state_abbr"].isin(STATE_ORDER)
    ].to_crs("EPSG:4326").copy()
    areas["geometry"] = areas.geometry.simplify(0.02, preserve_topology=True)
    geojson = json.dumps(
        json.loads(areas[["RA_CODE21", "geometry"]].to_json()),
        separators=(",", ":"),
    ).replace("</", "<\\/")
    national = float(
        tables["population_coverage_national"].loc[
            0, "approved_places_per_1000_children"
        ]
    )

    figure = go.Figure(
        go.Choropleth(
            geojson=json.loads(geojson),
            locations=coverage["remoteness_code"],
            featureidkey="properties.RA_CODE21",
            z=coverage["approved_places_per_1000_children"],
            colorscale="RdYlGn",
            zmid=national,
            marker_line_color="white",
            marker_line_width=0.7,
            hoverinfo="skip",
            colorbar={"title": {"text": "Places / 1,000"}, "thickness": 14},
        )
    )
    figure.update_geos(fitbounds="locations", visible=False, bgcolor="#F8FAFB")
    figure.update_layout(
        height=650,
        margin={"l": 12, "r": 12, "t": 12, "b": 12},
        paper_bgcolor="#FFFFFF",
        geo_bgcolor="#F8FAFB",
    )
    controls = "".join(
        [
            _control_row(
                "Measure",
                _button_group(
                    "metric",
                    [
                        ("places", "Approved places / 1,000"),
                        ("centre", "Centre services / 1,000"),
                        ("all", "All services / 1,000"),
                    ],
                    "places",
                ),
            ),
            _control_row(
                "State",
                _button_group(
                    "state",
                    [("All", "Australia")] + [(state, state) for state in STATE_ORDER],
                    "All",
                ),
            ),
        ]
    )
    kpis = _kpis(
        [
            ("coverage-children", "Children aged 0-13", "-"),
            ("coverage-places", "Approved places", "-"),
            ("coverage-rate", "Places per 1,000", "-"),
            ("coverage-services", "Centre services per 1,000", "-"),
            ("coverage-low", "Below national benchmark", "-"),
        ]
    )
    detail = r"""
<section class="detail-band" id="coverage-detail">
  <div><div class="detail-kicker">Selected population geography</div><div class="detail-title">Click a remoteness area</div></div>
  <div class="detail-copy">Coverage divides current-register service supply by the ABS 2021 population aged 0-13. It is a broad supply-intensity indicator, not a vacancy or unmet-demand estimate.</div>
</section>
"""
    note = (
        "Numerator: current-register services or Centre-Based approved places. Denominator: ABS 2021 Census usual residents aged 0-13. "
        "Pan and zoom to inspect small Remote and Very Remote polygons; groups with fewer than 5,000 children are flagged in the detail view."
    )
    script = r"""
const COVERAGE = __COVERAGE_DATA__;
const GEOJSON = __GEOJSON__;
const NATIONAL = __NATIONAL__;
const gd = document.getElementById('coverage-map');
const selection = {metric:'places', state:'All'};
let current = [];
const metricConfig = {
  places:{field:'approved_places_per_1000_children',title:'Approved places / 1,000',scale:[[0,'#b83245'],[.5,'#f7f3c6'],[1,'#2a9d8f']],mid:NATIONAL},
  centre:{field:'centre_based_services_per_1000_children',title:'Centre services / 1,000',scale:'Blues'},
  all:{field:'all_services_per_1000_children',title:'All services / 1,000',scale:'Blues'}
};
const finite = value => value !== null && value !== undefined && Number.isFinite(Number(value));
const fmt = (value,digits=1) => finite(value) ? Number(value).toLocaleString('en-AU',{minimumFractionDigits:digits,maximumFractionDigits:digits}) : 'n/a';
const countFmt = value => Number(value||0).toLocaleString('en-AU');
const escapeHtml = value => String(value ?? 'Not available').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
function updateCoverage() {
  current = COVERAGE.filter(row => selection.state === 'All' || row.state === selection.state);
  const config = metricConfig[selection.metric];
  const z = current.map(row => Number(row[config.field]));
  const custom = current.map(row => [row.state,row.remoteness_area,row.child_population_0_13,row.centre_based_approved_places,row.centre_based_services,row.all_services,row.approved_places_per_1000_children,row.centre_based_services_per_1000_children,row.all_services_per_1000_children,row.approved_places_per_centre_service,row.small_population_flag]);
  Plotly.react(gd,[{
    type:'choropleth',geojson:GEOJSON,locations:current.map(row=>row.remoteness_code),featureidkey:'properties.RA_CODE21',z,customdata:custom,
    colorscale:config.scale,zmid:selection.metric==='places'?NATIONAL:undefined,marker:{line:{color:'#ffffff',width:.7}},
    colorbar:{title:{text:config.title},thickness:14,len:.72,outlinewidth:0},
    hovertemplate:'<b>%{customdata[0]} | %{customdata[1]}</b><br>Children 0-13: %{customdata[2]:,.0f}<br>Approved places: %{customdata[3]:,.0f}<br>Places / 1,000: %{customdata[6]:.1f}<br>Centre services / 1,000: %{customdata[7]:.2f}<br>All services / 1,000: %{customdata[8]:.2f}<extra></extra>'
  }],{
    height:650,margin:{l:12,r:12,t:12,b:12},paper_bgcolor:'#fff',geo:{fitbounds:'locations',visible:false,bgcolor:'#f8fafb'},
    hoverlabel:{bgcolor:'#102a43',font:{color:'#fff'}}
  },{responsive:true,displaylogo:false,scrollZoom:true});
  const children = current.reduce((sum,row)=>sum+Number(row.child_population_0_13||0),0);
  const places = current.reduce((sum,row)=>sum+Number(row.centre_based_approved_places||0),0);
  const centreServices = current.reduce((sum,row)=>sum+Number(row.centre_based_services||0),0);
  const rate = children ? places/children*1000 : null;
  const serviceRate = children ? centreServices/children*1000 : null;
  const below = current.filter(row=>Number(row.approved_places_per_1000_children)<NATIONAL).length;
  document.getElementById('coverage-children').textContent=countFmt(children);
  document.getElementById('coverage-places').textContent=countFmt(places);
  document.getElementById('coverage-rate').textContent=fmt(rate);
  document.getElementById('coverage-services').textContent=fmt(serviceRate,2);
  document.getElementById('coverage-low').textContent=`${below} of ${current.length} groups`;
  const lowest=current.reduce((a,b)=>!a||Number(b.approved_places_per_1000_children)<Number(a.approved_places_per_1000_children)?b:a,null);
  document.getElementById('coverage-analysis').textContent=lowest
    ? `${selection.state==='All'?'Australia':selection.state}: ${fmt(rate)} Centre-Based approved places per 1,000 children aged 0-13. ${lowest.state} ${lowest.remoteness_area} has the lowest State x Remoteness ratio (${fmt(lowest.approved_places_per_1000_children)}); inspect its child denominator before interpretation.`
    : 'No population-compatible geography is available for this selection.';
}
function showCoverage(point) {
  const row=current[point.pointNumber]; if(!row) return;
  const gap=Number(row.approved_places_per_1000_children)-NATIONAL;
  document.getElementById('coverage-detail').innerHTML=`<div><div class="detail-kicker">Selected population geography</div><div class="detail-title">${escapeHtml(row.state)} | ${escapeHtml(row.remoteness_area)}</div><span class="badge ${gap<0?'alert':''}">${gap<0?'Below':'At or above'} national benchmark</span>${row.small_population_flag?'<span class="badge alert">Small child denominator</span>':''}</div>`+
    `<div class="detail-grid"><div class="detail-metric"><span>Children 0-13</span><strong>${countFmt(row.child_population_0_13)}</strong></div>`+
    `<div class="detail-metric"><span>Approved places</span><strong>${countFmt(row.centre_based_approved_places)}</strong></div>`+
    `<div class="detail-metric"><span>Places per 1,000</span><strong>${fmt(row.approved_places_per_1000_children)}</strong></div>`+
    `<div class="detail-metric"><span>Gap from national</span><strong>${fmt(gap)} places</strong></div>`+
    `<div class="detail-metric"><span>Centre services</span><strong>${countFmt(row.centre_based_services)}</strong></div>`+
    `<div class="detail-metric"><span>Centre services / 1,000</span><strong>${fmt(row.centre_based_services_per_1000_children,2)}</strong></div>`+
    `<div class="detail-metric"><span>All services / 1,000</span><strong>${fmt(row.all_services_per_1000_children,2)}</strong></div>`+
    `<div class="detail-metric"><span>Places per centre</span><strong>${fmt(row.approved_places_per_centre_service)}</strong></div></div>`;
}
document.querySelectorAll('.segment').forEach(button=>button.addEventListener('click',()=>{
  const group=button.dataset.group;selection[group]=button.dataset.value;
  document.querySelectorAll(`[data-group="${group}"]`).forEach(item=>{item.classList.toggle('active',item===button);item.setAttribute('aria-pressed',item===button?'true':'false');});
  updateCoverage();
}));
gd.on('plotly_click',event=>{if(event.points?.[0])showCoverage(event.points[0]);});
updateCoverage();
""".replace("__COVERAGE_DATA__", coverage_json).replace(
        "__GEOJSON__", geojson
    ).replace("__NATIONAL__", f"{national:.8f}")
    _write_page(
        output_path,
        "Population-adjusted service coverage",
        "Compare registered service supply with the ABS 2021 population aged 0-13 using compatible State-specific Remoteness Areas.",
        "3 | Accessibility & coverage",
        kpis,
        controls,
        "coverage-analysis",
        figure,
        "coverage-map",
        detail,
        script,
        note,
    )


def _screening_page(tables, output_path):
    screening = tables["compound_disadvantage"].copy()
    screening = screening[screening["eligible_for_comparison"]].copy()
    columns = [
        "state",
        "remoteness_area",
        "services",
        "rated_services",
        "below_nqs_pct_of_rated",
        "below_nqs_ci95_lower_pct",
        "below_nqs_ci95_upper_pct",
        "median_nearest_transport_km",
        "median_approved_places",
        "median_annual_weekly_hours",
        "quality_disadvantage",
        "access_disadvantage",
        "capacity_disadvantage",
        "coverage_disadvantage",
        "screening_flag",
        "enhanced_screening_flag",
        "national_below_nqs_pct",
        "national_median_transport_km",
        "national_median_approved_places",
        "child_population_0_13",
        "approved_places_per_1000_children",
        "national_approved_places_per_1000_children",
    ]
    screening_json = _json_records(screening, columns)
    initial_national_quality = float(screening["national_below_nqs_pct"].dropna().iloc[0])
    initial_national_transport = float(
        screening["national_median_transport_km"].dropna().iloc[0]
    )

    figure = go.Figure(
        go.Scatter(
            x=screening["median_nearest_transport_km"],
            y=screening["below_nqs_pct_of_rated"],
            mode="markers",
            marker={"size": 12, "color": screening["approved_places_per_1000_children"], "colorscale": "Viridis"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_vline(x=initial_national_transport, line_dash="dash", line_color="#829AB1")
    figure.add_hline(y=initial_national_quality, line_dash="dash", line_color="#829AB1")
    figure.update_layout(
        height=620,
        margin={"l": 72, "r": 30, "t": 24, "b": 64},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFCFD",
        xaxis={"type": "log", "title": "Median distance to nearest station (km)", "gridcolor": "#E8EEF2"},
        yaxis={"title": "Below NQS (% of rated services)", "rangemode": "tozero", "gridcolor": "#E8EEF2"},
        hoverlabel={"bgcolor": "#102A43", "font": {"color": "white"}},
    )
    controls = "".join(
        [
            _control_row(
                "State",
                _button_group(
                    "state", [("All", "Australia")] + [(s, s) for s in STATE_ORDER], "All"
                ),
            ),
            _control_row(
                "Screening view",
                _button_group(
                    "view",
                    [("all", "All comparable groups"), ("flagged", "Flagged groups only")],
                    "all",
                ),
            ),
        ]
    )
    kpis = _kpis(
        [
            ("screen-groups", "Comparable groups", "-"),
            ("screen-flags", "Flagged groups", "-"),
            ("screen-quality", "National below NQS", "-"),
            ("screen-transport", "National station distance", "-"),
            ("screen-coverage", "National places / 1,000", "-"),
        ]
    )
    detail = r"""
<section class="detail-band" id="screen-detail">
  <div><div class="detail-kicker">Selected state-remoteness group</div><div class="detail-title">Click a bubble</div></div>
  <div class="detail-copy">An enhanced screening flag identifies groups that are simultaneously worse than national benchmarks for quality, transport access, median centre size and population-adjusted approved places. It is a prioritisation signal, not causal proof.</div>
</section>
"""
    note = (
        "Bubble size = rated services; colour = approved places per 1,000 children; whiskers = 95% Wilson interval for Below NQS. "
        "Dashed lines are national benchmarks. Population uses ABS 2021 ages 0-13; service supply is from the current register."
    )
    script = r"""
const GROUPS = __SCREEN_DATA__;
const gd = document.getElementById('screening-chart');
const selection = {state:'All', view:'all'};
let current = [];
const finite = v => v !== null && v !== undefined && Number.isFinite(Number(v));
const fmt = (v,d=1) => finite(v) ? Number(v).toLocaleString('en-AU',{minimumFractionDigits:d,maximumFractionDigits:d}) : 'n/a';
const countFmt = v => Number(v||0).toLocaleString('en-AU');
const escapeHtml = value => String(value ?? 'Not available').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
function updateScreening() {
  const stateRows = GROUPS.filter(d => selection.state === 'All' || d.state === selection.state);
  current = stateRows.filter(d => selection.view === 'all' || d.enhanced_screening_flag === true);
  const nationalQ = Number(GROUPS[0]?.national_below_nqs_pct);
  const nationalT = Number(GROUPS[0]?.national_median_transport_km);
  const nationalC = Number(GROUPS[0]?.national_median_approved_places);
  const nationalCoverage = Number(GROUPS[0]?.national_approved_places_per_1000_children);
  const sizes = current.map(d => Math.max(11, Math.min(48, 6 + Math.sqrt(Number(d.rated_services))*1.45)));
  const lineColors = current.map(d => d.enhanced_screening_flag ? '#b83245' : '#ffffff');
  const upper = current.map(d => Math.max(0, Number(d.below_nqs_ci95_upper_pct)-Number(d.below_nqs_pct_of_rated)));
  const lower = current.map(d => Math.max(0, Number(d.below_nqs_pct_of_rated)-Number(d.below_nqs_ci95_lower_pct)));
  const custom = current.map(d => [d.state,d.remoteness_area,d.services,d.rated_services,d.below_nqs_ci95_lower_pct,d.below_nqs_ci95_upper_pct,d.median_approved_places,d.median_annual_weekly_hours,d.enhanced_screening_flag,d.approved_places_per_1000_children,d.child_population_0_13]);
  const xValues = current.map(d=>Number(d.median_nearest_transport_km)).filter(v=>v>0);
  const xMin = Math.max(.05, Math.min(nationalT, ...xValues) * .68);
  const xMax = Math.max(nationalT * 2, ...xValues) * 1.45;
  const yMax = Math.min(100, Math.max(nationalQ * 2, ...current.map(d=>Number(d.below_nqs_ci95_upper_pct))) + 6);
  Plotly.react(gd, [{
    type:'scatter',mode:'markers',x:current.map(d=>d.median_nearest_transport_km),y:current.map(d=>d.below_nqs_pct_of_rated),customdata:custom,
    marker:{size:sizes,color:current.map(d=>d.approved_places_per_1000_children),colorscale:[[0,'#d1495b'],[.48,'#f2cf5b'],[1,'#2a9d8f']],cmin:80,cmax:450,opacity:.88,line:{color:lineColors,width:current.map(d=>d.enhanced_screening_flag?3:1)},colorbar:{title:{text:'Places / 1,000'},thickness:13,len:.68,outlinewidth:0}},
    error_y:{type:'data',symmetric:false,array:upper,arrayminus:lower,color:'rgba(98,125,152,.48)',thickness:1,width:2},
    hovertemplate:'<b>%{customdata[0]} | %{customdata[1]}</b><br>Below NQS: %{y:.1f}% (95% CI %{customdata[4]:.1f}-%{customdata[5]:.1f})<br>Median station distance: %{x:.1f} km<br>Services: %{customdata[2]:,.0f} | rated: %{customdata[3]:,.0f}<br>Median capacity: %{customdata[6]:.0f} places<br>Approved places / 1,000 children: %{customdata[9]:.1f}<br>Children 0-13: %{customdata[10]:,.0f}<extra></extra>'
  }], {
    height:620,margin:{l:72,r:30,t:24,b:64},paper_bgcolor:'#fff',plot_bgcolor:'#fafcfd',
    xaxis:{type:'log',range:[Math.log10(xMin),Math.log10(xMax)],title:'Median distance to nearest station (km)',gridcolor:'#e8eef2',zeroline:false,tickvals:[.1,.3,1,3,10,30,100,300,1000],ticktext:['0.1','0.3','1','3','10','30','100','300','1,000']},
    yaxis:{title:'Below NQS (% of rated services)',range:[0,yMax],gridcolor:'#e8eef2',zeroline:false},
    hoverlabel:{bgcolor:'#102a43',font:{color:'#fff'}},
    shapes:[
      {type:'rect',xref:'x',yref:'y',x0:nationalT,x1:xMax,y0:nationalQ,y1:yMax,fillcolor:'rgba(209,73,91,.055)',line:{width:0},layer:'below'},
      {type:'line',xref:'x',yref:'paper',x0:nationalT,x1:nationalT,y0:0,y1:1,line:{color:'#829ab1',width:1.2,dash:'dash'}},
      {type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:nationalQ,y1:nationalQ,line:{color:'#829ab1',width:1.2,dash:'dash'}}
    ],
    annotations:[{xref:'paper',yref:'paper',x:.99,y:.98,text:'Higher access and quality concern',showarrow:false,xanchor:'right',font:{size:11,color:'#a1283b'},bgcolor:'rgba(255,255,255,.78)',borderpad:4}]
  }, {responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']});
  const flags = stateRows.filter(d => d.enhanced_screening_flag).length;
  document.getElementById('screen-groups').textContent = countFmt(stateRows.length);
  document.getElementById('screen-flags').textContent = countFmt(flags);
  document.getElementById('screen-quality').textContent = `${fmt(nationalQ)}%`;
  document.getElementById('screen-transport').textContent = `${fmt(nationalT)} km`;
  document.getElementById('screen-coverage').textContent = fmt(nationalCoverage);
  if (!current.length) {
    document.getElementById('screening-analysis').textContent = 'No groups meet the current screening selection.';
  } else {
    const highest = current.reduce((a,b)=>Number(b.below_nqs_pct_of_rated)>Number(a.below_nqs_pct_of_rated)?b:a);
    const farthest = current.reduce((a,b)=>Number(b.median_nearest_transport_km)>Number(a.median_nearest_transport_km)?b:a);
    document.getElementById('screening-analysis').textContent = `${selection.state === 'All' ? 'Australia' : selection.state}: ${flags} of ${stateRows.length} comparable state-remoteness groups trigger all four disadvantage rules. ${highest.state} ${highest.remoteness_area} has the highest observed Below NQS rate; ${farthest.state} ${farthest.remoteness_area} has the greatest median station distance.`;
  }
}
function ruleBadge(label, active) { return `<span class="badge ${active?'alert':''}">${active?'Above concern threshold':'Within benchmark'} | ${label}</span>`; }
function showDetail(mark) {
  document.getElementById('screen-detail').innerHTML = `<div><div class="detail-kicker">Selected state-remoteness group</div><div class="detail-title">${escapeHtml(mark.state)} | ${escapeHtml(mark.remoteness_area)}</div><div>${ruleBadge('Quality',mark.quality_disadvantage)}${ruleBadge('Transport',mark.access_disadvantage)}${ruleBadge('Facility size',mark.capacity_disadvantage)}${ruleBadge('Population coverage',mark.coverage_disadvantage)}</div></div>` +
    `<div class="detail-grid"><div class="detail-metric"><span>Below NQS</span><strong>${fmt(mark.below_nqs_pct_of_rated)}%</strong></div>` +
    `<div class="detail-metric"><span>95% interval</span><strong>${fmt(mark.below_nqs_ci95_lower_pct)}-${fmt(mark.below_nqs_ci95_upper_pct)}%</strong></div>` +
    `<div class="detail-metric"><span>Station distance</span><strong>${fmt(mark.median_nearest_transport_km)} km</strong></div>` +
    `<div class="detail-metric"><span>Median capacity</span><strong>${fmt(mark.median_approved_places,0)} places</strong></div>` +
    `<div class="detail-metric"><span>Places per 1,000 children</span><strong>${fmt(mark.approved_places_per_1000_children)}</strong></div>` +
    `<div class="detail-metric"><span>Children 0-13</span><strong>${countFmt(mark.child_population_0_13)}</strong></div>` +
    `<div class="detail-metric"><span>Rated denominator</span><strong>${countFmt(mark.rated_services)}</strong></div>` +
    `<div class="detail-metric"><span>Screening result</span><strong>${mark.enhanced_screening_flag?'Priority flag':'Not flagged'}</strong></div></div>`;
}
document.querySelectorAll('.segment').forEach(button => button.addEventListener('click', () => {
  const group=button.dataset.group;selection[group]=button.dataset.value;
  document.querySelectorAll(`[data-group="${group}"]`).forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',b===button?'true':'false');});
  updateScreening();
}));
gd.on('plotly_click', e => { const index=e.points?.[0]?.pointNumber; if(index!==undefined && current[index]) showDetail(current[index]); });
updateScreening();
""".replace("__SCREEN_DATA__", screening_json)
    _write_page(
        output_path,
        "Geographic disadvantage screening",
        "Combine quality, transport access, centre size and population-adjusted approved places to identify state-remoteness groups that merit closer investigation.",
        "6 | Synthesis",
        kpis,
        controls,
        "screening-analysis",
        figure,
        "screening-chart",
        detail,
        script,
        note,
    )


def create_interactive_visualisations(data, tables, boundaries, output_dir):
    """Write the standalone explorers used in the HTML presentation."""
    paths = output_paths(output_dir)
    interactive_dir = paths["interactive"]
    for stale in interactive_dir.glob("*.html"):
        stale.unlink()

    _service_landscape_page(
        data,
        boundaries,
        interactive_dir / "01_service_landscape_explorer.html",
    )
    _quality_page(
        tables,
        interactive_dir / "02_quality_geography_explorer.html",
    )
    _population_coverage_page(
        tables,
        boundaries,
        interactive_dir / "03_population_coverage_explorer.html",
    )
    _screening_page(
        tables,
        interactive_dir / "04_geographic_screening_explorer.html",
    )
