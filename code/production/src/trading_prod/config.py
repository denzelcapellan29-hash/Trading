from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .domain import ExecutionMode


@dataclass(frozen=True)
class StrategyAllocation:
    account_weight: float
    enabled: bool


@dataclass(frozen=True)
class ProductionConfig:
    raw: dict

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode(self.raw["execution_mode"])

    @property
    def transmit_orders(self) -> bool:
        return bool(self.raw["transmit_orders"])

    @property
    def account_id(self) -> str:
        return str(self.raw["account"]["account_id"])

    @property
    def base_currency(self) -> str:
        return str(self.raw["account"]["base_currency"])

    @property
    def strategy_allocations(self) -> dict[str, StrategyAllocation]:
        out: dict[str, StrategyAllocation] = {}
        for k, v in self.raw["portfolio"]["strategies"].items():
            out[k] = StrategyAllocation(float(v["account_weight"]), bool(v["enabled"]))
        return out

    @property
    def fx_leverage(self) -> float:
        return float(self.raw["portfolio"]["fx_sleeve_leverage"])

    def validate(self) -> None:
        if self.fx_leverage != 1.0:
            raise ValueError("V1 production config is frozen to unlevered FX (1.0x)")
        if self.mode is ExecutionMode.LIVE and not self.transmit_orders:
            raise ValueError("LIVE mode requires transmit_orders=true")
        if self.mode is ExecutionMode.LIVE:
            if not self.account_id or self.account_id == "REPLACE_WITH_IBKR_ACCOUNT_ID":
                raise ValueError("LIVE mode requires explicit account_id")
            if self.raw["risk"].get("hard_limits_status") != "FROZEN":
                raise ValueError("LIVE mode requires frozen hard account risk limits")

        alloc = self.strategy_allocations
        enabled_sum = sum(v.account_weight for v in alloc.values() if v.enabled and v.account_weight > 0)
        if enabled_sum > 1.0000001:
            raise ValueError(f"enabled account strategy weights exceed 100%: {enabled_sum:.6f}")


def load_config(path: str | Path) -> ProductionConfig:
    raw = json.loads(Path(path).read_text())
    cfg = ProductionConfig(raw)
    cfg.validate()
    return cfg
