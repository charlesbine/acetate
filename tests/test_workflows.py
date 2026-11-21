from __future__ import annotations

from pathlib import Path

from acetate import agent, workflows


def test_workflow_registry_is_one_to_one() -> None:
    ids = [workflow.id for workflow in workflows.WORKFLOWS]
    tools = [workflow.tool_name for workflow in workflows.WORKFLOWS]
    agent_tools = [tool["function"]["name"] for tool in agent.TOOLS]

    assert len(ids) == len(set(ids))
    assert len(tools) == len(set(tools))
    assert agent_tools == tools
    assert all(workflow.web_path for workflow in workflows.WORKFLOWS)


def test_checked_in_workflow_markdown_matches_registry() -> None:
    path = Path(__file__).parent.parent / "docs" / "workflows.md"
    assert path.read_text() == workflows.render_markdown()
