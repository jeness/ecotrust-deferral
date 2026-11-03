"""
Human-in-the-Loop Governance Layer — EcoTrust Pipeline

Escalation policy with full evaluation metrics:
  - escalation_rate     : % of decisions routed to human
  - error_coverage      : % of actual errors caught by escalation (recall of errors)
  - error_reduction     : accuracy improvement when human oracle corrects escalated cases
  - cp_coverage         : empirical coverage of CP prediction intervals
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EscalationConfig:
    ensemble_std_threshold: float = 0.08   # escalate if ensemble disagreement > this
    entropy_threshold: float = 0.40        # escalate if entropy > this
    risk_score_threshold: float = 0.60     # escalate if risk score > this (high-stakes)
    conflict_flag: bool = True             # always escalate on agent conflict


@dataclass
class DecisionRecord:
    date: object
    migration_risk: float
    ensemble_std: float
    entropy: float
    conflict: bool
    escalated: bool
    escalation_reasons: list
    ground_truth: Optional[int] = None
    model_pred: Optional[int] = None
    human_override: Optional[int] = None
    final_decision: Optional[int] = None

    @property
    def model_correct(self):
        if self.ground_truth is None or self.model_pred is None:
            return None
        return int(self.model_pred == self.ground_truth)

    @property
    def final_correct(self):
        if self.ground_truth is None or self.final_decision is None:
            return None
        return int(self.final_decision == self.ground_truth)


class HumanGovernanceLayer:
    """
    Applies escalation policy and computes comprehensive evaluation metrics.
    Human override is simulated with a perfect oracle (ground truth) to
    measure the upper bound benefit of human oversight.
    """

    def __init__(self, config: EscalationConfig = None):
        self.config = config or EscalationConfig()
        self.records: list[DecisionRecord] = []

    def evaluate(self, migration_out: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            migration_out: output of MigrationRiskAgent.predict()
                Must contain: migration_risk, ensemble_std, entropy, conflict,
                              ground_truth (for simulation)
        """
        cfg = self.config
        records = []

        for _, row in migration_out.iterrows():
            reasons = []
            escalated = False

            if row.get("ensemble_std", 0) > cfg.ensemble_std_threshold:
                reasons.append("ensemble_disagreement_high")
                escalated = True
            if row.get("entropy", 0) > cfg.entropy_threshold:
                reasons.append("entropy_high")
                escalated = True
            if row["migration_risk"] > cfg.risk_score_threshold:
                reasons.append("risk_score_high")
                escalated = True
            if cfg.conflict_flag and row.get("conflict", False):
                reasons.append("agent_conflict")
                escalated = True

            model_pred = int(row["migration_risk"] > 0.5)
            gt_val = row.get("ground_truth", None)
            gt = int(gt_val) if (gt_val is not None and not pd.isna(gt_val)) else None

            # Simulate perfect human oracle
            human_override = gt if (escalated and gt is not None) else None
            final_decision = human_override if human_override is not None else model_pred

            records.append(DecisionRecord(
                date=row["date"],
                migration_risk=row["migration_risk"],
                ensemble_std=row.get("ensemble_std", np.nan),
                entropy=row.get("entropy", np.nan),
                conflict=bool(row.get("conflict", False)),
                escalated=escalated,
                escalation_reasons=reasons,
                ground_truth=gt,
                model_pred=model_pred,
                human_override=human_override,
                final_decision=final_decision,
            ))

        self.records = records
        return pd.DataFrame([r.__dict__ for r in records])

    def summary(self) -> dict:
        if not self.records:
            return {}

        total = len(self.records)
        escalated_mask = [r.escalated for r in self.records]
        n_escalated = sum(escalated_mask)

        # Escalation rate
        escalation_rate = n_escalated / total

        # Error coverage: of all cases where model was WRONG,
        # what fraction were escalated? (recall of errors)
        errors = [r for r in self.records if r.model_correct == 0]
        n_errors = len(errors)
        errors_escalated = sum(r.escalated for r in errors)
        error_coverage = errors_escalated / n_errors if n_errors > 0 else float("nan")

        # Accuracy before and after escalation
        has_gt = [r for r in self.records if r.ground_truth is not None]
        if has_gt:
            acc_before = np.mean([r.model_correct for r in has_gt])
            acc_after  = np.mean([r.final_correct  for r in has_gt])
            error_reduction = acc_after - acc_before
        else:
            acc_before = acc_after = error_reduction = float("nan")

        # CP coverage: fraction of ground truth labels inside CP interval
        # (requires cp_in_high_risk from migration agent)
        cp_records = [r for r in self.records
                      if r.ground_truth is not None]
        # proxy: how often final_decision == ground_truth among escalated
        escalated_correct = [r for r in self.records
                             if r.escalated and r.final_correct == 1]
        escalation_precision = (
            len(escalated_correct) / n_escalated if n_escalated > 0 else float("nan")
        )

        # Reason breakdown
        reason_counts: dict = {}
        for r in self.records:
            for reason in r.escalation_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        return {
            "total_decisions":      total,
            "n_escalated":          n_escalated,
            "escalation_rate":      round(escalation_rate, 4),
            "n_model_errors":       n_errors,
            "errors_escalated":     errors_escalated,
            "error_coverage":       round(error_coverage, 4),   # recall of errors
            "accuracy_before":      round(acc_before, 4),
            "accuracy_after":       round(acc_after, 4),
            "error_reduction":      round(error_reduction, 4),  # Δ accuracy
            "escalation_precision": round(escalation_precision, 4),
            "conflict_cases":       sum(r.conflict for r in self.records),
            "reason_breakdown":     reason_counts,
        }
