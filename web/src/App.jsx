import { useEffect, useState } from "react";
import { api } from "./demo.js";
import {
  AcfChart,
  AnatomyChart,
  DepthChart,
  FrontierChart,
  InventoryChart,
  MidChart,
  ProbeTable,
  ScatterChart,
  ScheduleChart,
  Tape,
} from "./components/Charts.jsx";

const DEMO = import.meta.env.VITE_DEMO === "1";

function useCatalogs() {
  const [cat, setCat] = useState({ regimes: [], flows: [], algos: [], adversaries: [] });
  useEffect(() => {
    Promise.all([api("/api/regimes"), api("/api/algos"), api("/api/adversaries")])
      .then(([r, a, v]) =>
        setCat({ regimes: r.regimes, flows: r.flows, algos: a.algos, adversaries: v.adversaries })
      )
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
  }, DEMO ? 150 : 1000);
  return id;
}

function useJob() {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const submit = async (path, body, onDone) => {
    setBusy(true); setErr(null);
    try {
      const { job_id } = await api(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      poll(job_id, (j) => {
        setBusy(false);
        if (j.status === "error") setErr(j.error);
        else onDone(j.result);
      });
    } catch (e) { setBusy(false); setErr(String(e)); }
  };
  return { busy, err, submit };
}

function SimPanel({ onResult }) {
  const cat = useCatalogs();
  const [form, setForm] = useState({ regime: "normal", flow: "hawkes", horizon: 300, seed: 0 });
  const { busy, err, submit } = useJob();
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
      <button onClick={() => submit("/api/sim", { ...form, horizon: +form.horizon, seed: +form.seed }, onResult)}
        disabled={busy}>{busy ? "running…" : "run sim"}</button>
      {err && <p className="err">{err}</p>}
    </div>
  );
}

function ExecPanel({ onResult }) {
  const cat = useCatalogs();
  const [form, setForm] = useState({ algo: "ac", regime: "exec", total_qty: 100000, horizon: 1800, step: 10, n_episodes: 1, seed: 0 });
  const [advs, setAdvs] = useState([]);
  const [result, setResult] = useState(null);
  const { busy, err, submit } = useJob();
  const toggle = (a) => setAdvs((s) => (s.includes(a) ? s.filter((x) => x !== a) : [...s, a]));
  const run = () =>
    submit("/api/exec", {
      ...form,
      total_qty: +form.total_qty, horizon: +form.horizon, step: +form.step,
      n_episodes: +form.n_episodes, seed: +form.seed, adversaries: advs,
    }, (r) => { setResult(r); onResult(r); });
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
      <label>adversaries</label>
      <div className="advgrid">
        {cat.adversaries.map((a) => (
          <button key={a} type="button"
            className={`adv ${advs.includes(a) ? "on" : ""}`}
            onClick={() => toggle(a)}>{a}</button>
        ))}
      </div>
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

// Execution anatomy: what the algo actually did, decision by decision
function ExecAnatomy({ exec }) {
  const a = exec?.anatomy;
  if (!a?.trace?.length) return null;
  const last = a.trace[a.trace.length - 1];
  return (
    <div className="panel wide">
      <h2>Execution anatomy — {exec.algo}{exec.adversaries?.length ? ` vs ${exec.adversaries.join("+")}` : ""}</h2>
      <div className="grid2">
        <div>
          <p className="cap">mid path with the algo's fills (dot size = shares)</p>
          <AnatomyChart anatomy={a} />
        </div>
        <div>
          <p className="cap">inventory trajectory vs TWAP reference</p>
          <InventoryChart anatomy={a} totalQty={a.trace[0].remaining + a.trace[0].filled} horizon={a.trace[a.trace.length-1].t} />
        </div>
      </div>
      <div className="stats">
        <div className="stat"><b>{last.filled.toLocaleString()}</b><span>filled</span></div>
        <div className="stat"><b>{last.remaining.toLocaleString()}</b><span>left at deadline</span></div>
        <div className="stat"><b>{a.fills.length}</b><span>child fills</span></div>
        <div className="stat"><b>{a.trace.length}</b><span>decisions</span></div>
      </div>
      <details>
        <summary className="sub">decision log</summary>
        <div className="tape">
          <table>
            <thead><tr><th>t</th><th>mid</th><th>remaining</th><th>orders placed</th></tr></thead>
            <tbody>
              {a.trace.map((r, i) => (
                <tr key={i}>
                  <td>{r.t.toFixed(0)}</td>
                  <td>{r.mid == null ? "—" : `$${r.mid.toFixed(2)}`}</td>
                  <td>{r.remaining.toLocaleString()}</td>
                  <td>{r.children.map((c) => `${c.kind} ${c.qty}${c.price ? `@${c.price.toFixed(2)}` : ""}`).join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

// Almgren-Chriss frontier explorer — pure math, instant
function FrontierLab() {
  const [lam, setLam] = useState(-4);   // log10 lambda
  const [qty, setQty] = useState(100000);
  const [data, setData] = useState(null);
  useEffect(() => {
    const t = setTimeout(() => {
      api(`/api/frontier?total_qty=${qty}&lam=${(10 ** lam).toExponential(3)}`)
        .then(setData).catch(() => {});
    }, 120);
    return () => clearTimeout(t);
  }, [lam, qty]);
  return (
    <div className="panel wide">
      <h2>Almgren–Chriss frontier</h2>
      <p className="sub">
        x<sub>j</sub> = X·sinh κ(T−t<sub>j</sub>)/sinh κT, κ = acosh(1+λσ²τ²/2η̃)/τ —
        risk aversion λ trades expected cost against cost variance. σ, η from
        the sim-calibrated defaults (scripts/calibrate.py).
      </p>
      <div className="grid2">
        <div>
          <label>log₁₀ λ = {lam.toFixed(2)} (λ = {(10 ** lam).toExponential(1)})</label>
          <input type="range" min="-7" max="-3" step="0.1" value={lam}
            onChange={(e) => setLam(+e.target.value)} />
          <label>parent order: shares</label>
          <input type="number" value={qty} onChange={(e) => setQty(+e.target.value)} />
          {data?.at_lam && (
            <div className="stats">
              <div className="stat"><b>${data.at_lam.expected.toLocaleString()}</b><span>E[cost]</span></div>
              <div className="stat"><b>${data.at_lam.std.toLocaleString()}</b><span>std[cost]</span></div>
              <div className="stat"><b>{data.at_lam.kappa}</b><span>κ (urgency)</span></div>
            </div>
          )}
        </div>
        <div><FrontierChart curve={data} point={data?.at_lam} /></div>
      </div>
      <ScheduleChart at={data?.at_lam} />
    </div>
  );
}

// Stats lab: OFI regression + Kyle lambda + stylized facts on a fresh tape
function StatsLab() {
  const cat = useCatalogs();
  const [form, setForm] = useState({ regime: "normal", flow: "hawkes", horizon: 600, seed: 0 });
  const [res, setRes] = useState(null);
  const { busy, err, submit } = useJob();
  return (
    <div className="panel wide">
      <h2>Microstructure stats</h2>
      <div className="labrow">
        <select value={form.regime} onChange={(e) => setForm({ ...form, regime: e.target.value })}>
          {cat.regimes.map((r) => <option key={r}>{r}</option>)}
        </select>
        <select value={form.flow} onChange={(e) => setForm({ ...form, flow: e.target.value })}>
          {cat.flows.map((f) => <option key={f}>{f}</option>)}
        </select>
        <input type="number" style={{ width: 90 }} value={form.horizon}
          onChange={(e) => setForm({ ...form, horizon: e.target.value })} />
        <button onClick={() => submit("/api/stats", { ...form, horizon: +form.horizon, seed: +form.seed }, setRes)}
          disabled={busy}>{busy ? "fitting…" : "fit on fresh tape"}</button>
      </div>
      {err && <p className="err">{err}</p>}
      {res && (
        <>
          <div className="stats">
            <div className="stat"><b>{res.ofi.beta}</b><span>OFI β (ticks/sh)</span></div>
            <div className="stat"><b>{res.ofi.r2}</b><span>r² @1s</span></div>
            <div className="stat"><b>{res.ofi.t_stat}</b><span>t-stat</span></div>
            <div className="stat"><b>{res.ofi.windowed_r2}</b><span>r² @10s windows</span></div>
            <div className="stat"><b>{res.summary.kyle_lambda_ticks_per_1k ?? "—"}</b><span>Kyle λ /1k sh</span></div>
            <div className="stat"><b>{res.moments.excess_kurtosis}</b><span>xs kurtosis</span></div>
          </div>
          <div className="grid2">
            <div>
              <p className="cap">Δmid vs order-flow imbalance (1 s cadence)</p>
              <ScatterChart scatter={res.ofi.scatter} beta={res.ofi.beta} />
            </div>
            <div>
              <p className="cap">volatility clustering &amp; trade-sign persistence</p>
              <AcfChart vals={res.absret_acf} title="acf |ret|" />
              <AcfChart vals={res.sign_acf} title="acf trade sign" />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// Model lab: benchmark table + live probe against the trained checkpoint
function ModelLab() {
  const [models, setModels] = useState(null);
  const [probe, setProbe] = useState(null);
  const { busy, err, submit } = useJob();
  useEffect(() => { api("/api/models").then(setModels).catch(() => {}); }, []);
  const table = models?.pred_bench?.normal;
  const targets = table
    ? Object.keys(table.mlp || table.logistic || {}).filter((k) => k.endsWith("_nll")).map((k) => k.replace("_nll", ""))
    : [];
  return (
    <div className="panel wide">
      <h2>Prediction models</h2>
      {table ? (
        <div className="tape">
          <table>
            <thead>
              <tr><th>model</th>{targets.map((t) => <th key={t}>{t} NLL</th>)}</tr>
            </thead>
            <tbody>
              {Object.entries(table).map(([name, m]) => (
                <tr key={name}>
                  <td><b>{name}</b></td>
                  {targets.map((t) => {
                    const v = m[`${t}_nll`];
                    const best = Math.min(
                      ...Object.values(table).map((o) => o[`${t}_nll`] ?? Infinity)
                    );
                    return (
                      <td key={t} style={v === best ? { color: "var(--bid)" } : {}}>
                        {v == null ? "—" : v.toFixed(3)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="sub">no prediction bench results — run research/benchmark</p>
      )}
      <div className="labrow" style={{ marginTop: 14 }}>
        <p className="sub" style={{ margin: 0 }}>
          live probe — run a fresh episode, ask the trained MLP for P(mid move ↓/→/↑),
          then check what the tape actually did:
        </p>
        <button onClick={() => submit("/api/probe", { regime: "normal", flow: "hawkes", horizon: 120, n_probe: 24 }, setProbe)}
          disabled={busy}>{busy ? "probing…" : "probe the model"}</button>
      </div>
      {err && <p className="err">{err}</p>}
      {probe && (
        <>
          <div className="stats">
            <div className="stat"><b>{probe.checkpoint}</b><span>checkpoint</span></div>
            <div className="stat"><b>{(probe.probe_accuracy * 100).toFixed(0)}%</b><span>argmax acc on probe</span></div>
            <div className="stat"><b>{probe.n_events}</b><span>events probed</span></div>
          </div>
          <ProbeTable rows={probe.rows} />
        </>
      )}
    </div>
  );
}

function Leaderboard() {
  const [lb, setLb] = useState(null);
  useEffect(() => { api("/api/leaderboard").then(setLb).catch(() => {}); }, []);
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
  const [exec, setExec] = useState(null);
  return (
    <>
      <header>
        <h1>OrderFlow</h1>
        <span className="sub">limit-order-book research console{DEMO ? " · static demo (baked data)" : ""}</span>
      </header>
      <main>
        <SimPanel onResult={setSim} />
        <ExecPanel onResult={setExec} />
        <ExecAnatomy exec={exec} />
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
        <FrontierLab />
        <StatsLab />
        <ModelLab />
        <Leaderboard />
      </main>
    </>
  );
}
