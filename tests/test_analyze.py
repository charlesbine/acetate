"""The analytics tool: it must compute, and it must not be a file-read primitive."""

from __future__ import annotations

import json

import duckdb
import pytest

from acetate import agent, analyze, db, toolresult


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A tiny real database on disk, since DuckDB attaches a file, not a connection."""
    path = tmp_path / "acetate-test.db"
    conn = db.connect(path)
    db.init_db(conn)
    for i, plays in enumerate([1, 2, 3, 4, 100]):
        conn.execute(
            "INSERT INTO recording_stats(recording_key, rep_track_id, artist, title, plays, "
            "liked, homed) VALUES(?,?,?,?,?,1,0)",
            (f"artist\tt{i}", f"tid-{i}", "Artist", f"T{i}", plays),
        )
    conn.commit()
    conn.close()
    monkeypatch.setenv("ACETATE_DB", str(path))
    return path


def test_computes_the_statistics_sqlite_cannot(archive) -> None:
    result = analyze.run_analysis(
        "select median(plays) med, quantile_cont(plays, 0.9) p90, stddev_samp(plays) sd "
        "from recording_stats"
    )
    assert result["rows"][0][0] == 3.0  # median, not the 22.0 mean
    assert result["rows"][0][1] == pytest.approx(61.6)
    assert result["rows"][0][2] > 0


def test_table_names_need_no_catalog_prefix(archive) -> None:
    """The agent writes the same unqualified SQL it writes for query_db."""
    assert analyze.run_analysis("select count(*) from recording_stats")["rows"] == [[5]]


def test_works_as_a_calculator_over_literal_values(archive) -> None:
    result = analyze.run_analysis("select median(x) from (select unnest([4, 8, 15, 16]) as x)")
    assert result["rows"] == [[11.5]]


@pytest.mark.parametrize(
    "sql",
    [
        "select * from read_csv('/etc/passwd')",
        "select * from read_text('/etc/hostname')",
        "select * from read_csv('https://example.com/x.csv')",
        "select * from read_parquet('/data/anything.parquet')",
    ],
)
def test_cannot_read_the_filesystem_or_network(archive, sql) -> None:
    """These parse as SELECTs, so the guard waves them through — DuckDB's own
    disabled external access is what has to stop them."""
    with pytest.raises(duckdb.Error, match="Permission Error|Cannot access"):
        analyze.run_analysis(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "copy (select 1) to '/tmp/leak.csv'",
        "attach '/etc/passwd' as leak",
        "install httpfs",
    ],
)
def test_cannot_write_out_or_extend_itself(archive, sql) -> None:
    """Blocked at the guard here; DuckDB's permission layer backs it up."""
    with pytest.raises((ValueError, duckdb.Error)):
        analyze.run_analysis(sql)


def test_cannot_unlock_its_own_configuration(archive) -> None:
    """The lock is what stops a query from re-granting itself external access."""
    conn = analyze.analytic_connect()
    try:
        with pytest.raises(duckdb.Error, match="Cannot change configuration"):
            conn.execute("SET enable_external_access=true")
    finally:
        conn.close()


@pytest.mark.parametrize("sql", ["create table t(x int)", "select 1; drop table t", ""])
def test_rejects_anything_but_one_read_only_statement(archive, sql) -> None:
    with pytest.raises(ValueError):
        analyze.run_analysis(sql)


def test_the_archive_itself_is_attached_read_only(archive) -> None:
    conn = analyze.analytic_connect()
    try:
        with pytest.raises(duckdb.Error, match="read-only|read only"):
            conn.execute("insert into recording_stats(recording_key) values ('x')")
    finally:
        conn.close()


def test_sql_budget_is_shared_across_both_sql_tools(archive) -> None:
    """Spending it on one tool must not hand the model a fresh allowance on the other."""
    seen: dict = {}
    usage: dict[str, int] = {}
    sql = {"sql": "select 1"}
    for _ in range(agent.SQL_BUDGET):
        out, _ = agent.exec_tool(None, "analyze_db", dict(sql), seen, usage)
        assert "budget exhausted" not in out
        sql["sql"] += " "  # defeat the identical-call cache, not the budget

    out, _ = agent.exec_tool(None, "query_db", {"sql": "select 2"}, seen, usage)
    assert "budget exhausted" in out


# --- result packing -------------------------------------------------------


def test_oversized_results_stay_parseable_and_say_what_was_dropped() -> None:
    payload = {
        "columns": ["artist"],
        "rows": [[f"artist number {i}"] for i in range(500)],
        "row_count": 500,
        "truncated": False,
    }
    packed = toolresult.pack(payload, 600)

    assert len(packed) <= 600
    decoded = json.loads(packed)  # the whole point: still valid JSON
    assert decoded["truncated"] is True
    assert 0 < decoded["row_count"] < 500
    assert len(decoded["rows"]) == decoded["row_count"]
    assert "Showing" in decoded["note"]


def test_small_results_pass_through_untouched() -> None:
    payload = {"columns": ["n"], "rows": [[1]], "row_count": 1, "truncated": False}
    assert json.loads(toolresult.pack(payload, 6000)) == payload


def test_a_single_huge_cell_is_clipped_rather_than_losing_every_row() -> None:
    payload = {"columns": ["blob"], "rows": [["x" * 50_000]], "row_count": 1, "truncated": False}
    decoded = json.loads(toolresult.pack(payload, 2000))
    assert decoded["rows"], "clipping the cell should keep the row"
    assert "chars]" in decoded["rows"][0][0]


def test_non_tabular_payloads_degrade_to_a_labelled_envelope() -> None:
    decoded = json.loads(toolresult.pack({"deep": {"nested": ["x" * 10_000]}}, 500))
    assert decoded["truncated"] is True
    assert "partial" in decoded
