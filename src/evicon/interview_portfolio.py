"""One-command, offline portfolio run for the current EviCon-Lab route.

This module is deliberately a presentation boundary, not a new experiment.
It composes the already-tested offline protocol checks and the public cascade
replay smoke into a small set of safe artifacts that can be shown in an
interview.  It never constructs a provider, reads credentials, joins private
truth, or executes the 12-request qualification study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .cascade_controller_smoke import run_smoke as run_controller_smoke
from .cascade_outcome_smoke import run_smoke as run_cascade_smoke
from .conformity_source_behavior_offline_simulation_v1 import (
    run_offline_smoke as run_factorization_smoke,
)
from .conformity_source_behavior_update_offline_simulation_v1 import (
    run_offline_smoke as run_stateful_update_smoke,
)


PORTFOLIO_ID: Final[str] = "evicon-interview-portfolio-v1"
PORTFOLIO_VERSION: Final[str] = "interview_portfolio.v1"
DEFAULT_OUTPUT_ROOT: Final[Path] = Path("outputs/interview-portfolio-v1")

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONTROLLER_POLICY_PATH: Final[Path] = (
    _PROJECT_ROOT / "configs" / "provenance_cascade" / "cascade_controller_policy.v1.toml"
)
_SCENARIO_DIR: Final[Path] = _PROJECT_ROOT / "configs" / "provenance_cascade" / "scenarios"
_COMPONENT_PATHS: Final[tuple[str, ...]] = (
    "src/evicon/interview_portfolio.py",
    "src/evicon/conformity_source_behavior_offline_simulation_v1.py",
    "src/evicon/conformity_source_behavior_update_offline_simulation_v1.py",
    "src/evicon/cascade_controller_smoke.py",
    "src/evicon/cascade_controller.py",
    "src/evicon/cascade_outcome_smoke.py",
    "src/evicon/cascade_outcome_runner.py",
    "src/evicon/cascade_replay.py",
    "src/evicon/cascade_outcome_replay.py",
    "configs/studies/provenance_cascade_pilot_preregistration.toml",
    "configs/provenance_cascade/scenarios/false_majority.toml",
    "configs/provenance_cascade/scenarios/true_minority_correction.toml",
    "configs/provenance_cascade/scenarios/independent_true_consensus.toml",
    "configs/provenance_cascade/scenarios/unresolved_disagreement.toml",
    "configs/provenance_cascade/cascade_controller_policy.v1.toml",
)
_SENSITIVE_MARKERS: Final[tuple[str, ...]] = (
    "api_key",
    "authorization",
    "ground_truth_label",
    "source_independence_label",
    "evaluator_private_truth",
    "full_response",
    "prompt_saved",
)


class InterviewPortfolioError(ValueError):
    """Stable, non-sensitive failure boundary for the portfolio command."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _safe_text(value: str, field_name: str) -> str:
    cleaned = " ".join(str(value).split())
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    lowered = cleaned.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        raise ValueError(f"{field_name} contains a restricted marker")
    return cleaned


class PortfolioSafety(BaseModel):
    """Explicit safety claims for the offline presentation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    network_enabled: Literal[False] = False
    api_key_read: Literal[False] = False
    real_provider_constructed: Literal[False] = False
    evaluator_private_truth_exposed: Literal[False] = False
    historical_results_used: Literal[False] = False
    prompts_saved: Literal[False] = False
    raw_model_responses_saved: Literal[False] = False
    request_ledger_written: Literal[False] = False
    behavior_effect_estimated: Literal[False] = False
    not_paper_result: Literal[True] = True
    no_causal_conclusion: Literal[True] = True


class PortfolioCheck(BaseModel):
    """One auditable engineering check, reduced to safe metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    status: Literal["passed", "blocked"]
    observed: str = Field(min_length=1)

    @field_validator("check_id", "label", "observed")
    @classmethod
    def safe_fields(cls, value: str, info: object) -> str:
        return _safe_text(value, getattr(info, "field_name", "field"))


class PortfolioStep(BaseModel):
    """A named stage in the end-to-end demonstration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: Literal["passed", "blocked"]
    evidence: str = Field(min_length=1)

    @field_validator("step_id", "name", "evidence")
    @classmethod
    def safe_fields(cls, value: str, info: object) -> str:
        return _safe_text(value, getattr(info, "field_name", "field"))


class PortfolioSummary(BaseModel):
    """Counts that describe the engineering run, not model behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factorization_requests: int = Field(ge=0)
    stateful_update_requests: int = Field(ge=0)
    cascade_run_count: int = Field(ge=0)
    cascade_scenario_count: int = Field(ge=0)
    cascade_condition_count: int = Field(ge=0)
    agent_count: int = Field(ge=0)
    round_count: int = Field(ge=0)
    replay_pass_count: int = Field(ge=0)
    application_replay_pass_count: int = Field(ge=0)
    total_exposure_events: int = Field(ge=0)
    controller_proposal_count: int = Field(ge=0)
    controller_fixture_count: int = Field(ge=0)
    controller_valid_proposal_count: int = Field(ge=0)
    distinct_trajectory_count: int = Field(ge=0)


class PortfolioReport(BaseModel):
    """Human-readable and machine-readable safe report body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    portfolio_id: Literal["evicon-interview-portfolio-v1"]
    portfolio_version: Literal["interview_portfolio.v1"]
    status: Literal["completed"]
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    technology_stack: tuple[str, ...] = Field(min_length=1)
    pipeline: tuple[PortfolioStep, ...] = Field(min_length=1)
    summary: PortfolioSummary
    checks: tuple[PortfolioCheck, ...] = Field(min_length=1)
    controller_examples: tuple[str, ...] = Field(min_length=1)
    what_is_demonstrated: tuple[str, ...] = Field(min_length=1)
    interpretation_boundary: tuple[str, ...] = Field(min_length=1)
    interview_pitch: str = Field(min_length=1)
    safety: PortfolioSafety

    @field_validator(
        "title",
        "purpose",
        "technology_stack",
        "what_is_demonstrated",
        "interpretation_boundary",
        "interview_pitch",
    )
    @classmethod
    def safe_text_fields(cls, value: str | tuple[str, ...], info: object) -> str | tuple[str, ...]:
        field_name = getattr(info, "field_name", "field")
        if isinstance(value, tuple):
            return tuple(_safe_text(item, field_name) for item in value)
        return _safe_text(value, field_name)


class PortfolioReceipt(BaseModel):
    """Small receipt binding the report to the code/configuration used."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    portfolio_id: Literal["evicon-interview-portfolio-v1"]
    portfolio_version: Literal["interview_portfolio.v1"]
    status: Literal["completed"]
    output_root: str
    output_files: tuple[str, ...] = Field(min_length=1)
    output_sha256: dict[str, str]
    component_sha256: dict[str, str]
    summary: PortfolioSummary
    safety: PortfolioSafety

    @field_validator("output_root")
    @classmethod
    def safe_root(cls, value: str) -> str:
        return _safe_text(value, "output_root")


@dataclass(frozen=True)
class _PortfolioRun:
    report: PortfolioReport
    trace_rows: tuple[dict[str, object], ...]
    component_sha256: dict[str, str]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except (OSError, UnicodeDecodeError) as exc:
        raise InterviewPortfolioError("component_missing") from exc


def _component_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in _COMPONENT_PATHS:
        path = _PROJECT_ROOT / relative
        hashes[relative] = _sha256_file(path)
    return hashes


def _offline_safety_ok(report: dict[str, object]) -> bool:
    safety = report.get("safety")
    if not isinstance(safety, dict):
        return False
    expected_false = (
        "real_provider_constructed",
        "api_key_read",
        "results_written",
        "prompt_saved",
        "full_response_saved",
        "historical_results_used",
        "evaluator_private_truth_exposed",
        "behavior_effect_estimated",
    )
    return safety.get("network") == "disabled" and all(
        safety.get(key) is False for key in expected_false
    )


def _check(
    check_id: str,
    label: str,
    passed: bool,
    observed: str,
) -> PortfolioCheck:
    return PortfolioCheck(
        check_id=check_id,
        label=label,
        status="passed" if passed else "blocked",
        observed=observed,
    )


def _safe_trace_rows(
    rows: Sequence[dict[str, object]],
    *,
    stage: str,
) -> tuple[dict[str, object], ...]:
    """Keep only IDs, counts, hashes, and replay statuses in the trace."""

    cascade_keys = (
        "scenario_id",
        "seed",
        "condition",
        "agent_count",
        "round_count",
        "event_count",
        "outcome_count",
        "directive_applied_count",
        "proposal_count",
        "trajectory_hash",
        "replay_status",
        "application_replay_status",
    )
    controller_keys = (
        "scenario_id",
        "condition_id",
        "action",
        "reason_codes",
        "used_content_count",
        "used_evidence_count",
        "used_root_count",
        "valid",
        "status",
    )
    keys = cascade_keys if stage == "cascade" else controller_keys
    return tuple({"stage": stage, **{key: row[key] for key in keys}} for row in rows)


def run_portfolio() -> _PortfolioRun:
    """Run every current offline stage and assemble a safe report."""

    try:
        factorization = run_factorization_smoke()
        stateful_update = run_stateful_update_smoke()
        controller_rows = run_controller_smoke(
            policy_path=_CONTROLLER_POLICY_PATH,
            scenario_dir=_SCENARIO_DIR,
        )
        cascade_rows = run_cascade_smoke()
        component_hashes = _component_hashes()
    except InterviewPortfolioError:
        raise
    except Exception as exc:
        raise InterviewPortfolioError("offline_stage_failed") from exc

    replay_pass_count = sum(row.get("replay_status") == "passed" for row in cascade_rows)
    application_pass_count = sum(
        row.get("application_replay_status") == "passed" for row in cascade_rows
    )
    scenario_count = len({str(row["scenario_id"]) for row in cascade_rows})
    condition_count = len({str(row["condition"]) for row in cascade_rows})
    agent_count = max((int(row["agent_count"]) for row in cascade_rows), default=0)
    round_count = max((int(row["round_count"]) for row in cascade_rows), default=0)
    controller_valid_count = sum(row.get("valid") is True for row in controller_rows)
    provenance_rows = [
        row
        for row in controller_rows
        if row.get("condition_id") == "provenance_aware_controller"
    ]
    false_majority_request = any(
        row.get("scenario_id") == "cascade-false-majority"
        and row.get("action") == "request_independent_source"
        and row.get("used_root_count") == 1
        for row in provenance_rows
    )
    independent_consensus_protected = any(
        row.get("scenario_id") == "cascade-independent-true-consensus"
        and row.get("action") == "abstain"
        and row.get("used_root_count") == 2
        for row in provenance_rows
    )
    minority_correction_protected = any(
        row.get("scenario_id") == "cascade-true-minority-correction"
        and row.get("action") == "abstain"
        and row.get("used_root_count") == 1
        for row in provenance_rows
    )
    controller_examples = tuple(
        "{scenario_id}: action={action}; roots={roots}; reasons={reasons}".format(
            scenario_id=row["scenario_id"],
            action=row["action"],
            roots=row["used_root_count"],
            reasons=",".join(str(reason) for reason in row["reason_codes"]),
        )
        for row in provenance_rows
    )
    summary = PortfolioSummary(
        factorization_requests=int(factorization["logical_request_count"]),
        stateful_update_requests=int(stateful_update["fake_logical_request_count"]),
        cascade_run_count=len(cascade_rows),
        cascade_scenario_count=scenario_count,
        cascade_condition_count=condition_count,
        agent_count=agent_count,
        round_count=round_count,
        replay_pass_count=replay_pass_count,
        application_replay_pass_count=application_pass_count,
        total_exposure_events=sum(int(row["event_count"]) for row in cascade_rows),
        controller_proposal_count=sum(int(row["proposal_count"]) for row in cascade_rows),
        controller_fixture_count=len(controller_rows),
        controller_valid_proposal_count=controller_valid_count,
        distinct_trajectory_count=len({str(row["trajectory_hash"]) for row in cascade_rows}),
    )

    checks = (
        _check(
            "factorization_design",
            "来源条件矩阵与解析器",
            factorization.get("status") == "offline_factorization_smoke_passed"
            and factorization.get("known_difference_recovered") is True
            and factorization.get("known_no_difference_recovered") is True,
            "预设差异和零差异均被离线检查恢复",
        ),
        _check(
            "stateful_update_design",
            "初始判断到更新的状态链",
            stateful_update.get("status") == "stateful_update_offline_smoke_passed"
            and stateful_update.get("known_difference_recovered") is True
            and stateful_update.get("order_shortcut_detected") is True,
            "共享初始状态、分支更新和顺序诊断均通过",
        ),
        _check(
            "public_replay",
            "公开暴露回放",
            bool(cascade_rows)
            and replay_pass_count == len(cascade_rows),
            f"{replay_pass_count}/{len(cascade_rows)} 条传播运行通过回放",
        ),
        _check(
            "application_replay",
            "干预应用回放",
            bool(cascade_rows)
            and application_pass_count == len(cascade_rows),
            f"{application_pass_count}/{len(cascade_rows)} 条运行通过应用审计",
        ),
        _check(
            "controller_policy_smoke",
            "来源关系规则演示",
            bool(controller_rows)
            and controller_valid_count == len(controller_rows)
            and false_majority_request
            and independent_consensus_protected
            and minority_correction_protected,
            "同源重复请求独立来源，独立来源共识和有支持的少数纠正均保留为可审计动作",
        ),
        _check(
            "offline_boundary",
            "离线与数据隔离边界",
            _offline_safety_ok(factorization)
            and _offline_safety_ok(stateful_update),
            "不联网、不读凭据、不构造真实服务、不暴露私有真值",
        ),
        _check(
            "component_integrity",
            "代码与配置指纹",
            len(component_hashes) == len(_COMPONENT_PATHS)
            and all(len(value) == 64 for value in component_hashes.values()),
            f"已绑定 {len(component_hashes)} 个代码或配置文件",
        ),
    )
    pipeline = (
        PortfolioStep(
            step_id="protocol",
            name="固定协议和结构化契约",
            status="passed",
            evidence="来源条件、前后状态和输出字段均有版本化校验",
        ),
        PortfolioStep(
            step_id="offline_validation",
            name="离线合成验证",
            status="passed",
            evidence="两个离线模拟阶段通过预设差异、零差异和顺序诊断",
        ),
        PortfolioStep(
            step_id="network_replay",
            name="多轮传播与实际暴露记录",
            status="passed" if replay_pass_count == len(cascade_rows) else "blocked",
            evidence=f"四类场景、四种条件共 {len(cascade_rows)} 次确定性运行",
        ),
        PortfolioStep(
            step_id="audit_report",
            name="回放审计与安全汇总",
            status="passed" if all(item.status == "passed" for item in checks) else "blocked",
            evidence="只输出安全计数、状态和哈希，不输出提示词或原始回复",
        ),
    )
    report = PortfolioReport(
        portfolio_id=PORTFOLIO_ID,
        portfolio_version=PORTFOLIO_VERSION,
        status="completed",
        title="EviCon-Lab 当前路线的离线端到端展示闭环",
        purpose="把来源结构、信息暴露、结构化判断、传播回放和安全汇总串成一个可复现演示。",
        technology_stack=(
            "Python 3.11+：核心 Agent 运行时、命令行入口和实验编排。",
            "Pydantic 2：严格数据契约、结构化输出、配置校验和不可变快照。",
            "自研 Agent 运行时：统一执行渲染、一次调用、解析、校验和安全审计。",
            "OpenAI 兼容接口适配：用统一 Provider 契约连接真实服务和本地 FakeProvider。",
            "来源图与来源根归并：区分同源转发、独立来源和公开证据关系。",
            "Exposure Ledger 与轮次快照：记录 Agent 实际看见的内容，并约束可传播信息。",
            "多轮传播、控制器和回放验证：检查时序、可见性、干预应用和结果引用。",
            "TOML、JSON 和 JSONL：管理版本化配置并输出脱敏报告、轨迹和回执。",
            "pytest、FakeProvider/FakeTransport 和 SHA-256：离线故障注入、契约测试和完整性绑定。",
        ),
        pipeline=pipeline,
        summary=summary,
        checks=checks,
        controller_examples=controller_examples,
        what_is_demonstrated=(
            "系统能够记录每轮实际暴露，并按公开来源关系进行回放。",
            "来源条件和前后状态协议能够在离线环境中做成对照检查。",
            "公开视图上的来源关系规则可以给出可解释的结构化提案。",
            "控制器与传播运行可以产生可审计的结构化记录。",
            "运行结果可以脱敏后绑定到代码和配置指纹。",
        ),
        interpretation_boundary=(
            "这是确定性工程演示和协议验证，不是模型行为效果结论。",
            "控制器案例展示的是预先写明的规则输出，不代表真实模型已经表现出同样行为。",
            "没有在这里估计因果效应，也没有合并旧 Pilot 或真人预试结果。",
            "下一步研究仍需单独完成真实模型行为实验和预注册分析。",
        ),
        interview_pitch=(
            "我把一个容易变成模型演示的研究想法，做成了可审计的实验基础设施："
            "同一条信息经过不同 Agent 传播时，系统能记录实际暴露、合并来源根、"
            "限制控制器只能看公开视图，并在运行后用回放验证时序和可见性。"
            "目前我先用完全离线的合成模拟把协议、状态更新和四类网络场景闭环，"
            "再把真实模型调用作为受控的后续阶段，而不是把工程 smoke 当成论文结果。"
        ),
        safety=PortfolioSafety(),
    )
    return _PortfolioRun(
        report=report,
        trace_rows=_safe_trace_rows(controller_rows, stage="controller")
        + _safe_trace_rows(cascade_rows, stage="cascade"),
        component_sha256=component_hashes,
    )


def _json_bytes(value: object, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _render_markdown(report: PortfolioReport, output_files: Sequence[str]) -> str:
    summary = report.summary
    lines = [
        "# EviCon-Lab：面试展示闭环",
        "",
        "> 本文件是离线工程演示报告，不是模型行为或因果效果结论。",
        "",
        "## 一句话介绍",
        "",
        report.interview_pitch,
        "",
        "## 运行链路",
        "",
    ]
    for index, step in enumerate(report.pipeline, start=1):
        lines.append(f"{index}. **{step.name}**：{step.evidence}（{step.status}）")
    lines.extend(
        [
            "",
            "## 本次离线运行",
            "",
            f"- 来源条件离线检查：{summary.factorization_requests} 个逻辑请求。",
            f"- 前后状态离线检查：{summary.stateful_update_requests} 个逻辑请求。",
            f"- 多轮传播运行：{summary.cascade_run_count} 次，覆盖 {summary.cascade_scenario_count} 类场景和 {summary.cascade_condition_count} 种条件。",
            f"- 回放审计：传播 {summary.replay_pass_count}/{summary.cascade_run_count}，干预应用 {summary.application_replay_pass_count}/{summary.cascade_run_count}。",
            f"- 记录的公开暴露事件：{summary.total_exposure_events}；传播运行中的控制器提案：{summary.controller_proposal_count}。",
            f"- 独立控制器规则案例：{summary.controller_fixture_count} 个，结构化提案有效 {summary.controller_valid_proposal_count} 个。",
            "",
            "## 一个可解释的来源对照",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report.controller_examples)
    lines.extend(["", "## 已实现的 Agent 技术栈", ""])
    lines.extend(f"- {item}" for item in report.technology_stack)
    lines.append(
        "- 边界说明：这是 Python + Pydantic 的自研 Agent 应用与实验基础设施；当前不包含模型训练或微调，也不把未实际使用的 Agent 框架写成项目依赖。"
    )
    lines.extend(["", "## 可以展示的工程能力", ""])
    lines.extend(f"- {item}" for item in report.what_is_demonstrated)
    lines.extend(["", "## 需要主动说明的边界", ""])
    lines.extend(f"- {item}" for item in report.interpretation_boundary)
    lines.extend(["", "## 生成文件", ""])
    lines.extend(f"- `{item}`" for item in output_files)
    lines.extend(
        [
            "",
            "## 建议的面试讲法",
            "",
            "先讲问题：消息数量不等于独立证据数量。再讲实现：我把来源图、实际暴露快照、结构化 Agent 输出和回放验证串起来。最后讲边界：当前闭环证明的是协议和审计链可运行，真实模型行为和论文结论仍单独验证。",
            "",
        ]
    )
    return "\n".join(lines)


def write_portfolio(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    overwrite: bool = False,
) -> PortfolioReceipt:
    """Run the portfolio and write only the four declared safe artifacts."""

    output_root = Path(output_root)
    if output_root.exists() and not overwrite:
        raise InterviewPortfolioError("output_exists_use_overwrite")
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InterviewPortfolioError("output_root_unavailable") from exc

    run = run_portfolio()
    output_files = (
        "portfolio_report.json",
        "portfolio_report.md",
        "portfolio_trace.jsonl",
        "portfolio_receipt.json",
    )
    report_bytes = _json_bytes(run.report.model_dump(mode="json"), indent=2)
    markdown = _render_markdown(run.report, output_files)
    markdown_bytes = markdown.encode("utf-8")
    trace_bytes = b"".join(_json_bytes(row) for row in run.trace_rows)
    output_sha256 = {
        "portfolio_report.json": _sha256_bytes(report_bytes),
        "portfolio_report.md": _sha256_bytes(markdown_bytes),
        "portfolio_trace.jsonl": _sha256_bytes(trace_bytes),
    }
    receipt = PortfolioReceipt(
        portfolio_id=PORTFOLIO_ID,
        portfolio_version=PORTFOLIO_VERSION,
        status="completed",
        output_root=str(output_root),
        output_files=output_files,
        output_sha256=output_sha256,
        component_sha256=run.component_sha256,
        summary=run.report.summary,
        safety=run.report.safety,
    )
    receipt_bytes = _json_bytes(receipt.model_dump(mode="json"), indent=2)
    payloads = {
        "portfolio_report.json": report_bytes,
        "portfolio_report.md": markdown_bytes,
        "portfolio_trace.jsonl": trace_bytes,
        "portfolio_receipt.json": receipt_bytes,
    }
    try:
        for name, payload in payloads.items():
            (output_root / name).write_bytes(payload)
    except OSError as exc:
        raise InterviewPortfolioError("portfolio_artifact_write_failed") from exc
    return receipt


def _blocked_payload(error_code: str) -> dict[str, object]:
    return {
        "status": "blocked",
        "error_code": error_code,
        "network_enabled": False,
        "api_key_read": False,
        "real_provider_constructed": False,
        "evaluator_private_truth_exposed": False,
        "behavior_effect_estimated": False,
        "not_paper_result": True,
        "no_causal_conclusion": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the offline EviCon-Lab interview portfolio artifacts."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination for the four safe portfolio artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the four declared artifacts if the destination exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks and print the report without writing files.",
    )
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            report = run_portfolio().report
            print(json.dumps(report.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
        else:
            receipt = write_portfolio(args.output_root, overwrite=args.overwrite)
            print(json.dumps(receipt.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))
        return 0
    except InterviewPortfolioError as exc:
        print(json.dumps(_blocked_payload(exc.code), ensure_ascii=True, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "InterviewPortfolioError",
    "PORTFOLIO_ID",
    "PORTFOLIO_VERSION",
    "PortfolioCheck",
    "PortfolioReceipt",
    "PortfolioReport",
    "PortfolioSafety",
    "PortfolioStep",
    "PortfolioSummary",
    "main",
    "run_portfolio",
    "write_portfolio",
]
