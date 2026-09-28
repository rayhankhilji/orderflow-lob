// Demo-mode data layer. When built with VITE_DEMO=1 (the GitHub Pages
// static build), api() serves baked fixtures from ./demo/*.json — recorded
// from real local runs by scripts/bake_demo.py. POST endpoints return a
// synthetic job id that /api/jobs/* resolves to the fixture.

const DEMO = import.meta.env.VITE_DEMO === "1";
const BASE = import.meta.env.BASE_URL;

const fixture = async (name) => {
  const r = await fetch(`${BASE}demo/${name}.json`);
  if (!r.ok) throw new Error(`demo fixture ${name} missing`);
  return r.json();
};

// job_id -> fixture file
const JOB_FIXTURES = {
  "demo-sim": "sim",
  "demo-stats": "stats",
  "demo-probe": "probe",
};

export async function api(path, opts) {
  if (!DEMO) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  }

  const [route, qs] = path.split("?");
  if (route === "/api/regimes" || route === "/api/algos" || route === "/api/adversaries") {
    const cat = await fixture("catalog");
    const key = route.split("/").pop();
    return { [key]: cat[key] };
  }
  if (route === "/api/health") return { ok: true, demo: true };
  if (route === "/api/leaderboard") return fixture("leaderboard");
  if (route === "/api/models") return fixture("models");
  if (route === "/api/frontier") {
    const f = await fixture("frontier");
    const lam = +(new URLSearchParams(qs || "").get("lam") || 1e-4);
    const keys = Object.keys(f.schedules);
    const nearest = keys.reduce(
      (a, b) => (Math.abs(Number(b) - lam) < Math.abs(Number(a) - lam) ? b : a)
    );
    return { ...f.curve, at_lam: f.schedules[nearest], params: f.params };
  }
  if (route === "/api/sim" && opts?.method === "POST") return { job_id: "demo-sim" };
  if (route === "/api/stats" && opts?.method === "POST") return { job_id: "demo-stats" };
  if (route === "/api/probe" && opts?.method === "POST") return { job_id: "demo-probe" };
  if (route === "/api/exec" && opts?.method === "POST") {
    const body = JSON.parse(opts.body || "{}");
    return { job_id: body.adversaries?.length ? "demo-exec-adv" : "demo-exec" };
  }
  if (route.startsWith("/api/jobs/")) {
    const id = route.split("/").pop();
    const name =
      JOB_FIXTURES[id] || (id === "demo-exec-adv" ? "exec_twap_passive_adv" : "exec_ac");
    return { id, status: "done", result: await fixture(name) };
  }
  throw new Error(`demo: no fixture for ${path}`);
}
