// Hand-rolled SVG charts — no chart dependency, everything is in-repo data.

const W = 560, H = 180, PAD = 30;

function frame(series, w = W, h = H, pad = PAD) {
  const xs = series.map(([t]) => t);
  const ys = series.map(([, m]) => m);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  return {
    w, h, pad, x0, x1, y0, y1,
    sx: (t) => pad + ((t - x0) / Math.max(x1 - x0, 1e-9)) * (w - 2 * pad),
    sy: (m) => h - pad - ((m - y0) / Math.max(y1 - y0, 1e-9)) * (h - 2 * pad),
  };
}

const path = (series, f) =>
  series.map(([t, m], i) => `${i ? "L" : "M"}${f.sx(t).toFixed(1)},${f.sy(m).toFixed(1)}`).join(" ");

function AxisLabels({ f, xfmt = (v) => `${v.toFixed(0)}s`, yfmt = (v) => v.toFixed(2) }) {
  return (
    <>
      <text x={f.pad} y={14} fill="var(--dim)" fontSize="10">{yfmt(f.y1)}</text>
      <text x={f.pad} y={f.h - 6} fill="var(--dim)" fontSize="10">{yfmt(f.y0)}</text>
      <text x={f.w - f.pad} y={f.h - 6} fill="var(--dim)" fontSize="10" textAnchor="end">{xfmt(f.x1)}</text>
    </>
  );
}

export function MidChart({ series }) {
  if (!series?.length) return <p className="sub">run a sim to see the mid path</p>;
  const f = frame(series);
  return (
    <svg viewBox={`0 0 ${f.w} ${f.h}`} width="100%">
      <path d={path(series, f)} fill="none" stroke="var(--accent)" strokeWidth="1.4" />
      <AxisLabels f={f} yfmt={(v) => `$${v.toFixed(2)}`} />
    </svg>
  );
}

// Execution anatomy: mid path + exec fills (dots sized by qty) + decision marks
export function AnatomyChart({ anatomy }) {
  if (!anatomy?.mid_series?.length) return null;
  const f = frame(anatomy.mid_series);
  const maxQ = Math.max(...anatomy.fills.map((x) => x.qty), 1);
  return (
    <svg viewBox={`0 0 ${f.w} ${f.h}`} width="100%">
      <path d={path(anatomy.mid_series, f)} fill="none" stroke="var(--accent)" strokeWidth="1.2" opacity="0.7" />
      {anatomy.trace.map((r, i) =>
        r.mid == null ? null : (
          <line key={`d${i}`} x1={f.sx(r.t)} x2={f.sx(r.t)} y1={f.pad} y2={f.h - f.pad}
            stroke="var(--border)" strokeWidth="0.6" strokeDasharray="2 3" />
        )
      )}
      {anatomy.fills.map((x, i) => (
        <circle key={i} cx={f.sx(x.t)} cy={f.sy(x.price)}
          r={1.5 + 4 * Math.sqrt(x.qty / maxQ)} fill="var(--ask)" opacity="0.85" />
      ))}
      <AxisLabels f={f} yfmt={(v) => `$${v.toFixed(2)}`} />
      <text x={f.w - f.pad} y={14} fill="var(--ask)" fontSize="10" textAnchor="end">
        {anatomy.fills.length} fills
      </text>
    </svg>
  );
}

// Inventory trajectory vs the TWAP reference line
export function InventoryChart({ anatomy, totalQty, horizon }) {
  if (!anatomy?.trace?.length) return null;
  const inv = anatomy.trace.map((r) => [r.t, r.remaining]);
  inv.unshift([anatomy.trace[0].t - 0.001, totalQty]);
  const ref = [[0, totalQty], [horizon, 0]];
  const f = frame([...inv, ...ref]);
  return (
    <svg viewBox={`0 0 ${f.w} ${f.h}`} width="100%">
      <path d={path(ref, f)} fill="none" stroke="var(--dim)" strokeWidth="1"
        strokeDasharray="5 4" />
      <path d={path(inv, f)} fill="none" stroke="var(--accent)" strokeWidth="1.6" />
      {anatomy.trace.map((r, i) => (
        <circle key={i} cx={f.sx(r.t)} cy={f.sy(r.remaining)} r="2" fill="var(--accent)" />
      ))}
      <AxisLabels f={f} yfmt={(v) => `${(v / 1000).toFixed(0)}k`} />
      <text x={f.w - f.pad} y={14} fill="var(--dim)" fontSize="10" textAnchor="end">
        dashed = TWAP reference
      </text>
    </svg>
  );
}

// AC efficient frontier: E[cost] vs std[cost], with the selected lambda marked
export function FrontierChart({ curve, point }) {
  if (!curve?.expected?.length) return null;
  const pts = curve.expected.map((e, i) => [e, curve.std[i]]);
  const f = frame(pts);
  return (
    <svg viewBox={`0 0 ${f.w} ${f.h}`} width="100%">
      <path d={path(pts, f)} fill="none" stroke="var(--accent)" strokeWidth="1.6" />
      {pts.map(([e, s], i) => (
        <circle key={i} cx={f.sx(e)} cy={f.sy(s)} r="2.5" fill="var(--accent)" opacity="0.6" />
      ))}
      {point && (
        <circle cx={f.sx(point.expected)} cy={f.sy(point.std)} r="6"
          fill="none" stroke="var(--ask)" strokeWidth="2" />
      )}
      <AxisLabels f={f}
        xfmt={(v) => `$${(v / 1000).toFixed(0)}k`}
        yfmt={(v) => `$${(v / 1000).toFixed(0)}k`} />
      <text x={f.w - f.pad} y={14} fill="var(--dim)" fontSize="10" textAnchor="end">
        x = E[cost] → · y = std[cost] ↑
      </text>
    </svg>
  );
}

// AC optimal inventory schedule x(t) for a chosen lambda
export function ScheduleChart({ at }) {
  if (!at?.schedule_x?.length) return null;
  const pts = at.schedule_t.map((t, i) => [t, at.schedule_x[i]]);
  const f = frame(pts);
  return (
    <svg viewBox={`0 0 ${f.w} ${f.h}`} width="100%">
      <path d={path(pts, f)} fill="none" stroke="var(--bid)" strokeWidth="1.6" />
      <AxisLabels f={f} yfmt={(v) => `${(v / 1000).toFixed(0)}k`} />
      <text x={f.w - f.pad} y={14} fill="var(--dim)" fontSize="10" textAnchor="end">
        κ = {at.kappa} · remaining shares over time
      </text>
    </svg>
  );
}

// OFI scatter + fitted regression line
export function ScatterChart({ scatter, beta }) {
  if (!scatter?.length) return null;
  const f = frame(scatter);
  const [x0, x1] = [f.x0, f.x1];
  const yAt = (x) => {
    const ys = scatter.map(([, y]) => y);
    const my = ys.reduce((a, b) => a + b, 0) / ys.length;
    const mx = scatter.reduce((a, [x]) => a + x, 0) / scatter.length;
    return my + beta * (x - mx);
  };
  return (
    <svg viewBox={`0 0 ${f.w} ${f.h}`} width="100%">
      <line x1={f.sx(x0)} y1={f.sy(yAt(x0))} x2={f.sx(x1)} y2={f.sy(yAt(x1))}
        stroke="var(--ask)" strokeWidth="1.4" />
      {scatter.map(([x, y], i) => (
        <circle key={i} cx={f.sx(x)} cy={f.sy(y)} r="2" fill="var(--accent)" opacity="0.55" />
      ))}
      <AxisLabels f={f}
        xfmt={(v) => `OFI ${v.toFixed(0)}`}
        yfmt={(v) => `${v.toFixed(1)}t`} />
      <text x={f.w - f.pad} y={14} fill="var(--ask)" fontSize="10" textAnchor="end">
        Δmid = β·OFI, β={beta}
      </text>
    </svg>
  );
}

// autocorrelation bars (lags on x, rho on y, zero line)
export function AcfChart({ vals, title }) {
  if (!vals?.length) return null;
  const w = 560, h = 120, pad = 20;
  const y0 = Math.min(0, ...vals), y1 = Math.max(0.05, ...vals);
  const sy = (v) => h - pad - ((v - y0) / (y1 - y0)) * (h - 2 * pad);
  const bw = (w - 2 * pad) / vals.length;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%">
      <line x1={pad} x2={w - pad} y1={sy(0)} y2={sy(0)} stroke="var(--border)" />
      {vals.map((v, i) => (
        <rect key={i} x={pad + i * bw + 1} y={Math.min(sy(v), sy(0))}
          width={Math.max(bw - 2, 1)} height={Math.abs(sy(v) - sy(0))}
          fill="var(--accent)" opacity="0.75" />
      ))}
      <text x={pad} y={12} fill="var(--dim)" fontSize="10">{title}</text>
    </svg>
  );
}

// probe rows: predicted P(down/flat/up) as bars, realized class marked
export function ProbeTable({ rows }) {
  if (!rows?.length) return null;
  const colors = { down: "var(--ask)", flat: "var(--dim)", up: "var(--bid)" };
  return (
    <div className="tape">
      <table>
        <thead>
          <tr><th>t</th><th>event</th><th>P(down)</th><th>P(flat)</th><th>P(up)</th><th>actual</th></tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.t.toFixed(1)}</td>
              <td>{r.event_type}</td>
              {["down", "flat", "up"].map((k) => (
                <td key={k}>
                  <span className="probbar">
                    <span style={{
                      display: "inline-block", height: 9, width: `${r.pred[k] * 60}px`,
                      background: colors[k], borderRadius: 2, opacity: r.realized === k ? 1 : 0.55,
                    }} />
                    <span style={{ fontSize: 11, marginLeft: 4 }}>{r.pred[k].toFixed(2)}</span>
                  </span>
                </td>
              ))}
              <td style={{ color: colors[r.realized], fontWeight: 600 }}>
                {r.realized}{r.correct ? " ✓" : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DepthChart({ depth }) {
  if (!depth) return <p className="sub">no depth snapshot</p>;
  const n = Math.min(depth.bid_prices.length, depth.ask_prices.length, 10);
  const rows = [];
  const maxQ = Math.max(
    ...depth.bid_qtys.slice(0, n),
    ...depth.ask_qtys.slice(0, n),
    1
  );
  // asks above (reversed so best ask is lowest row of the ask block)
  for (let i = n - 1; i >= 0; i--) {
    rows.push({ side: "ask", price: depth.ask_prices[i], qty: depth.ask_qtys[i] });
  }
  for (let i = 0; i < n; i++) {
    rows.push({ side: "bid", price: depth.bid_prices[i], qty: depth.bid_qtys[i] });
  }
  return (
    <div>
      {rows.map((r, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, margin: "1px 0" }}>
          <span style={{ width: 60, color: r.side === "bid" ? "var(--bid)" : "var(--ask)", fontSize: 12 }}>
            {(r.price * 0.01).toFixed(2)}
          </span>
          <div
            style={{
              height: 12,
              width: `${(r.qty / maxQ) * 70}%`,
              background: r.side === "bid" ? "var(--bid)" : "var(--ask)",
              opacity: 0.75,
              borderRadius: 2,
            }}
          />
          <span style={{ color: "var(--dim)", fontSize: 11 }}>{r.qty}</span>
        </div>
      ))}
      <p style={{ color: "var(--dim)", fontSize: 11, marginTop: 8 }}>
        best {n} levels · t={depth.t.toFixed(1)}s
      </p>
    </div>
  );
}

export function Tape({ tape }) {
  if (!tape?.length) return <p className="sub">no events</p>;
  return (
    <div className="tape">
      <table>
        <thead>
          <tr><th>t</th><th>type</th><th>side</th><th>px</th><th>qty</th></tr>
        </thead>
        <tbody>
          {[...tape].reverse().map((e, i) => (
            <tr key={i}>
              <td>{e.t.toFixed(2)}</td>
              <td>{e.market ? "market" : e.type}</td>
              <td className={e.side === 1 ? "buy" : "sell"}>{e.side === 1 ? "BUY" : "SELL"}</td>
              <td>{e.price != null ? (e.price * 0.01).toFixed(2) : "—"}</td>
              <td>{e.qty}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
