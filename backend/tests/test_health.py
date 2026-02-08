"""Tests for health and root endpoints."""


def test_root_returns_platform_info(client):
    r = client.get("/")
    assert r.status_code == 200

    data = r.json()
    assert data["platform"] == "HarmonyRestorer v1"
    assert data["status"] == "ready"
    assert ".wav" in data["supported_formats"]


def test_health_reports_real_checks(client):
    r = client.get("/health")
    assert r.status_code == 200

    data = r.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["checks"]["disk_space"]["ok"] is True
    assert data["checks"]["directories"]["ok"] is True
    assert isinstance(data["checks"]["disk_space"]["free_mb"], int)


def test_health_includes_job_counts(client):
    r = client.get("/health")
    data = r.json()
    assert "jobs" in data
    assert data["jobs"] == {}  # no jobs yet
