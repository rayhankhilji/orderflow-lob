import { useEffect, useRef, useState } from "react";
import { DepthChart, MidChart, Tape } from "./components/Charts.jsx";

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function useCatalogs() {
  const [cat, setCat] = useState({ regimes: [], flows: [], algos: [] });
  useEffect(() => {
    Promise.all([api("/api/regimes"), api("/api/algos")])
      .then(([r, a]) => setCat({ regimes: r.regimes, flows: r.flows, algos: a.algos }))
      .catch(() => {});
  }, []);
  return cat;
}

function poll(jobId, onDone) {
  const id = setInterval(async () => {
    try {
      const j = await api(`/api/jobs/${jobId}`);
      if (j.status === "done" || j.status === "error") {
        clearInterval(id);
        onDone(j);
      }
    } catch {
      clearInterval(id);
    }
  }, 1000);
  return id;
}

function SimPanel({ onResult }) {
  const cat = useCatalogs();
  const [form, setForm] = useState({ regime: "normal", flow: "hawkes", horizon: 300, seed: 0 });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const run = async () => {
    setBusy(true); setErr(null);
    try {
      const { job_id } = await api("/api/sim", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, horizon: +form.horizon, seed: +form.seed }),
      });
      poll(job_id, (j) => {
        setBusy(false);
        if (j.status === "error") setErr(j.error);
        else onResult(j.result);
      });
    } catch (e) { setBusy(false); setErr(String(e)); }
  };
  return (
    <div className="panel">
      <h2>Simulation</h2>
      <label>regime</label>
      <select value={form.regime} onChange={(e) => setForm({ ...form, regime: e.target.value })}>
        {cat.regimes.map((r) => <option key={r}>{r}</option>)}
      </select>
      <label>flow model</label>
      <select value={form.flow} onChange={(e) => setForm({ ...form, flow: e.target.value })}>
        {cat.flows.map((f) => <option key={f}>{f}</option>)}
      </select>
      <label>horizon (s)</label>
      <input type="number" value={form.horizon} onChange={(e) => setForm({ ...form, horizon: e.target.value })} />
      <label>seed</label>
      <input type="number" value={form.seed} onChange={(e) => setForm({ ...form, seed: e.target.value })} />
      <button onClick={run} disabled={busy}>{busy ? "running…" : "run sim"}</button>
      {err && <p className="err">{err}</p>}
    </div>
  );
}

function ExecPanel({ onResult }) {
  const cat = useCatalogs();
  const [form, setForm] = useState({ algo: "twap", regime: "exec", total_qty: 100000, horizon: 1800, step: 10, n_episodes: 2, seed: 0 });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [result, setResult] = useState(null);
  const run = async () => {
    setBusy(true); setErr(null); setResult(null);
    try {
      const { job_id } = await api("/api/exec", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          total_qty: +form.total_qty,
          horizon: +form.horizon,
          step: +form.step,
          n_episodes: +form.n_episodes,
          seed: +form.seed,
        }),
      });
      poll(job_id, (j) => {
        setBusy(false);
        if (j.status === "error") setErr(j.error);
        else setResult(j.result);
      });
    } catch (e) { setBusy(false); setErr(String(e)); }
  };
  return (
    <div className="panel">
      <h2>Execution</h2>
      <label>algo</label>
      <select value={form.algo} onChange={(e) => setForm({ ...form, algo: e.target.value })}>
        {cat.algos.map((a) => <option key={a}>{a}</option>)}
      </select>
      <label>regime</label>
      <select value={form.regime} onChange={(e) => setForm({ ...form, regime: e.target.value })}>
        {cat.regimes.map((r) => <option key={r}>{r}</option>)}
      </select>
      <label>shares</label>
      <input type="number" value={form.total_qty} onChange={(e) => setForm({ ...form, total_qty: e.target.value })} />
      <label>horizon (s)</label>
      <input type="number" value={form.horizon} onChange={(e) => setForm({ ...form, horizon: e.target.value })} />
      <label>episodes</label>
      <input type="number" min="1" max="10" value={form.n_episodes} onChange={(e) => setForm({ ...form, n_episodes: e.target.value })} />
      <button onClick={run} disabled={busy}>{busy ? "running…" : "run exec"}</button>
      {err && <p className="err">{err}</p>}
      {result && (
        <div className="stats">
          <div className="stat"><b>{result.summary.mean_is_bps?.toFixed(2)}</b><span>mean IS (bps)</span></div>
          <div className="stat"><b>{result.summary.cvar95_is_bps?.toFixed(2)}</b><span>CVaR95</span></div>
          <div className="stat"><b>{(result.summary.mean_fill_rate * 100).toFixed(1)}%</b><span>fill</span></div>
          <div className="stat"><b>{(result.summary.mean_participation * 100).toFixed(1)}%</b><span>participation</span></div>
        </div>
      )}
    </div>
  );
}

function Leaderboard() {
  const [lb, setLb] = useState(null);
  useEffect(() => {
    api("/api/leaderboard").then(setLb).catch(() => {});
  }, []);
  return (
    <div className="panel wide">
      <h2>Leaderboard</h2>
      {lb?.leaderboard_md
        ? <pre className="leaderboard">{lb.leaderboard_md}</pre>
        : <p className="sub">no LEADERBOARD.md in the repo yet — run the benchmark pipeline</p>}
    </div>
  );
}

export default function App() {
  const [sim, setSim] = useState(null);
  return (
    <>
      <header>
        <h1>OrderFlow</h1>
        <span className="sub">limit-order-book research console</span>
      </header>
      <main>
        <SimPanel onResult={setSim} />
        <ExecPanel />
        <div className="panel">
          <h2>Mid price</h2>
          <MidChart series={sim?.mid_series} />
          {sim && (
            <div className="stats">
              <div className="stat"><b>{sim.summary.events}</b><span>events</span></div>
              <div className="stat"><b>{sim.summary.events_per_sec}/s</b><span>rate</span></div>
              <div className="stat"><b>{sim.summary.mean_spread_ticks}</b><span>mean spread (ticks)</span></div>
              <div className="stat"><b>{sim.summary.fills}</b><span>fills</span></div>
            </div>
          )}
        </div>
        <div className="panel">
          <h2>Book depth</h2>
          <DepthChart depth={sim?.depth} />
        </div>
        <div className="panel wide">
          <h2>Event tape</h2>
          <Tape tape={sim?.tape} />
        </div>
        <Leaderboard />
      </main>
    </>
  );
}
