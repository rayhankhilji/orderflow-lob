"""API smoke tests: job lifecycle for sim + exec endpoints."""

import time

import pytest
from fastapi.testclient import TestClient

from orderflow.api.app import app

client = TestClient(app)


def _wait(job_id: int, timeout: float = 120.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} still running")


def test_health_and_catalogs() -> None:
    assert client.get("/api/health").json()["ok"]
    regimes = client.get("/api/regimes").json()
    assert {"exec", "normal"} <= set(regimes["regimes"])
    assert "twap" in client.get("/api/algos").json()["algos"]


def test_sim_job_lifecycle() -> None:
    r = client.post(
        "/api/sim",
        json={"regime": "normal", "flow": "zi", "horizon": 20.0, "seed": 2},
    ).json()
    j = _wait(r["job_id"])
    assert j["status"] == "done", j.get("error")
    res = j["result"]
    assert res["summary"]["events"] > 0
    assert len(res["mid_series"]) > 5
    assert res["depth"]["bid_prices"]
    assert res["tape"]


@pytest.mark.slow
def test_exec_job_lifecycle() -> None:
    r = client.post(
        "/api/exec",
        json={
            "algo": "twap",
            "regime": "normal",
            "total_qty": 500,
            "horizon": 120.0,
            "step": 10.0,
            "n_episodes": 1,
        },
    ).json()
    j = _wait(r["job_id"])
    assert j["status"] == "done", j.get("error")
    assert j["result"]["summary"]["runs"] == 1.0


def test_unknown_inputs_400() -> None:
    assert client.post("/api/sim", json={"regime": "nope"}).status_code == 400
    assert (
        client.post(
            "/api/exec", json={"algo": "nope", "total_qty": 10, "horizon": 30}
        ).status_code
        == 400
    )
    assert client.get("/api/jobs/999999").status_code == 404
