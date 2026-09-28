// Hand-rolled SVG charts — no chart dependency, everything is in-repo data.

export function MidChart({ series }) {
  if (!series?.length) return <p className="sub">run a sim to see the mid path</p>;
  const w = 560, h = 180, pad = 28;
  const xs = series.map(([t]) => t);
  const ys = series.map(([, m]) => m);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (t) => pad + ((t - x0) / Math.max(x1 - x0, 1e-9)) * (w - 2 * pad);
  const sy = (m) => h - pad - ((m - y0) / Math.max(y1 - y0, 1e-9)) * (h - 2 * pad);
  const d = series.map(([t, m], i) => `${i ? "L" : "M"}${sx(t).toFixed(1)},${sy(m).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%">
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth="1.4" />
      <text x={pad} y={14} fill="var(--dim)" fontSize="10">${y1.toFixed(2)}</text>
      <text x={pad} y={h - 6} fill="var(--dim)" fontSize="10">${y0.toFixed(2)}</text>
      <text x={w - pad} y={h - 6} fill="var(--dim)" fontSize="10" textAnchor="end">{x1.toFixed(0)}s</text>
    </svg>
  );
}

export function DepthChart({ depth }) {
  if (!depth) return <p className="sub">no depth snapshot</p>;
  const n = Math.min(depth.bid_prices.length, depth.ask_prices.length, 10);
  const tick = depth.bid_prices.length ? Math.abs(depth.ask_prices[0] - depth.bid_prices[0]) || 1 : 1;
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
