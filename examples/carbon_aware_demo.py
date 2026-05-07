#!/usr/bin/env python3
"""HarchOS Carbon-Aware Scheduling — Interactive Demo.

This script demonstrates the full carbon-aware scheduling workflow:

1. Fetches real-time carbon intensity for multiple zones
2. Ranks hubs by carbon intensity (greenest first)
3. Optimizes workload scheduling to minimize CO2
4. Shows carbon savings vs. baseline (worst hub)
5. Displays forecasts and green windows

Usage:
    python carbon_aware_demo.py [--api-key HSK_...] [--base-url http://localhost:8000/v1]

Requirements:
    pip install harchos rich
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone

# Try importing rich for pretty output; fall back to plain print
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _print(msg: str = "") -> None:
    print(msg)


def _header(text: str) -> None:
    if HAS_RICH:
        Console().rule(text, style="bold green")
    else:
        _print(f"\n{'=' * 60}")
        _print(f"  {text}")
        _print(f"{'=' * 60}\n")


def _zone_table(zones: list) -> None:
    """Print a formatted table of zone carbon intensities."""
    if HAS_RICH:
        console = Console()
        table = Table(title="Global Carbon Intensity", box=box.ROUNDED)
        table.add_column("Zone", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("gCO2/kWh", justify="right")
        table.add_column("Renewable %", justify="right", style="green")
        table.add_column("Status", style="bold")

        for z in sorted(zones, key=lambda x: x.carbon_intensity_gco2_kwh):
            status = "[green]GREEN[/green]" if z.is_green else "[red]DIRTY[/red]"
            table.add_row(
                z.zone,
                z.zone_name,
                f"{z.carbon_intensity_gco2_kwh:.0f}",
                f"{z.renewable_percentage:.1f}%",
                status,
            )
        console.print(table)
    else:
        _print(f"{'Zone':<12} {'Name':<25} {'gCO2/kWh':>10} {'Renewable%':>12} {'Status'}")
        _print("-" * 70)
        for z in sorted(zones, key=lambda x: x.carbon_intensity_gco2_kwh):
            status = "GREEN" if z.is_green else "DIRTY"
            _print(f"{z.zone:<12} {z.zone_name:<25} {z.carbon_intensity_gco2_kwh:>10.0f} {z.renewable_percentage:>11.1f}% {status}")


async def run_demo(api_key: str, base_url: str) -> None:
    """Run the interactive carbon-aware demo."""
    from harchos import HarchOSClient

    _header("HarchOS Carbon-Aware Scheduling Demo")

    async with HarchOSClient(api_key=api_key, base_url=base_url, carbon_aware=True) as client:
        # ------------------------------------------------------------------
        # Step 1: Health check
        # ------------------------------------------------------------------
        _header("Step 1: Health Check")
        try:
            health = await client.async_health()
            _print(f"  Status: {health.status}")
            _print(f"  Version: {health.version}")
        except Exception as e:
            _print(f"  ERROR: Cannot connect to HarchOS server: {e}")
            _print(f"  Make sure the server is running at {base_url}")
            return

        # ------------------------------------------------------------------
        # Step 2: Carbon intensity across zones
        # ------------------------------------------------------------------
        _header("Step 2: Global Carbon Intensity Map")
        try:
            zones_list = await client.carbon.async_list_intensities()
            _zone_table(zones_list.zones)
        except Exception as e:
            _print(f"  ERROR fetching intensities: {e}")

        # ------------------------------------------------------------------
        # Step 3: Morocco specific
        # ------------------------------------------------------------------
        _header("Step 3: Morocco Carbon Intensity")
        try:
            ma = await client.carbon.async_get_intensity("MA")
            _print(f"  Zone: {ma.zone} ({ma.zone_name})")
            _print(f"  Carbon Intensity: {ma.carbon_intensity_gco2_kwh:.0f} gCO2/kWh")
            _print(f"  Renewable: {ma.renewable_percentage:.1f}%")
            _print(f"  Data Source: {ma.source}")
            _print(f"  Green: {'YES' if ma.is_green else 'NO'} (> 200 gCO2/kWh threshold)")
        except Exception as e:
            _print(f"  ERROR: {e}")

        # ------------------------------------------------------------------
        # Step 4: Find optimal hub
        # ------------------------------------------------------------------
        _header("Step 4: Carbon-Optimal Hub Selection")
        try:
            optimal = await client.carbon.async_optimal_hub(
                region="europe",
                gpu_count=4,
                gpu_type="A100",
                carbon_max_gco2=200,
                priority="normal",
                defer_ok=True,
            )
            _print(f"  Recommended Hub: {optimal.recommended_hub_name or 'None'}")
            _print(f"  Hub Region: {optimal.hub_region}")
            _print(f"  Hub Zone: {optimal.hub_zone}")
            _print(f"  Carbon Intensity: {optimal.carbon_intensity_gco2_kwh:.0f} gCO2/kWh")
            _print(f"  Renewable %: {optimal.renewable_percentage:.1f}%")
            _print(f"  Available GPUs: {optimal.available_gpus}")
            _print(f"  Action: {optimal.action}")
            if optimal.is_deferred:
                _print(f"  Defer Hours: {optimal.defer_hours:.1f}h")
                _print(f"  Defer Reason: {optimal.defer_reason}")
            _print(f"  Estimated Carbon Saved: {optimal.estimated_carbon_saved_kg:.4f} kg CO2")

            if optimal.alternative_hubs:
                _print("\n  Alternative Hubs:")
                for alt in optimal.alternative_hubs[:5]:
                    _print(f"    - {alt.get('hub_name', 'N/A')}: "
                           f"{alt.get('carbon_intensity_gco2_kwh', 0):.0f} gCO2/kWh, "
                           f"{alt.get('available_gpus', 0)} GPUs")
        except Exception as e:
            _print(f"  ERROR: {e}")

        # ------------------------------------------------------------------
        # Step 5: Optimize a workload
        # ------------------------------------------------------------------
        _header("Step 5: Carbon-Aware Workload Optimization")
        workloads = [
            {"name": "llama-finetune", "gpus": 8, "type": "A100", "duration": 4.0, "max_co2": 100},
            {"name": "inference-api", "gpus": 2, "type": "T4", "duration": 24.0, "max_co2": 200},
            {"name": "batch-eval", "gpus": 1, "type": "V100", "duration": 0.5, "max_co2": None},
        ]

        for wl in workloads:
            _print(f"\n  Workload: {wl['name']}")
            _print(f"    GPUs: {wl['gpus']}x {wl['type']}, Duration: {wl['duration']}h")
            try:
                result = await client.carbon.async_optimize(
                    workload_name=wl["name"],
                    gpu_count=wl["gpus"],
                    gpu_type=wl["type"],
                    carbon_aware=True,
                    carbon_max_gco2=wl["max_co2"],
                    estimated_duration_hours=wl["duration"],
                )
                _print(f"    Action: {result.action}")
                _print(f"    Hub: {result.selected_hub_name or 'N/A'}")
                _print(f"    Carbon Intensity: {result.carbon_intensity_at_schedule_gco2_kwh:.0f} gCO2/kWh")
                _print(f"    Carbon Saved: {result.carbon_saved_kg:.4f} kg CO2")
                _print(f"    Baseline (worst hub): {result.baseline_carbon_kg:.4f} kg CO2")
                _print(f"    Actual (selected hub): {result.actual_carbon_kg:.4f} kg CO2")
                if result.carbon_savings_percentage > 0:
                    _print(f"    Savings: {result.carbon_savings_percentage:.1f}%")
                if result.deferred_hours > 0:
                    _print(f"    Deferred: {result.deferred_hours:.1f}h — {result.reason}")
            except Exception as e:
                _print(f"    ERROR: {e}")

        # ------------------------------------------------------------------
        # Step 6: Carbon forecast
        # ------------------------------------------------------------------
        _header("Step 6: Carbon Intensity Forecast (Sweden)")
        try:
            forecast = await client.carbon.async_get_forecast("SE", hours=12)
            _print(f"  Zone: {forecast.zone} ({forecast.zone_name})")
            _print(f"  Forecast Points: {len(forecast.forecast)}")
            green_count = sum(1 for p in forecast.forecast if p.is_green)
            _print(f"  Green Points: {green_count}/{len(forecast.forecast)}")
            _print(f"  Green Hours: {forecast.green_hours_count}h")

            if forecast.green_windows:
                _print("\n  Green Windows:")
                for gw in forecast.green_windows[:5]:
                    _print(f"    {gw.get('start', 'N/A')} -> {gw.get('end', 'N/A')} "
                           f"(~{gw.get('estimated_carbon_intensity_gco2_kwh', '?')} gCO2/kWh)")
            else:
                _print("  No distinct green windows found in the forecast period")
        except Exception as e:
            _print(f"  ERROR: {e}")

        # ------------------------------------------------------------------
        # Step 7: Platform carbon metrics
        # ------------------------------------------------------------------
        _header("Step 7: Platform Carbon Metrics")
        try:
            metrics = await client.carbon.async_get_metrics(period_days=30)
            _print(f"  Total CO2 Saved: {metrics.total_carbon_saved_kg:.4f} kg")
            _print(f"  Workloads Optimized: {metrics.total_workloads_optimized}")
            _print(f"  Workloads Deferred: {metrics.total_workloads_deferred}")
            _print(f"  Average Carbon Intensity: {metrics.average_carbon_intensity_gco2_kwh:.0f} gCO2/kWh")
            _print(f"  Best Hub: {metrics.best_hub_name} ({metrics.best_hub_carbon_intensity:.0f} gCO2/kWh)")
            _print(f"  Worst Hub Intensity: {metrics.worst_hub_carbon_intensity:.0f} gCO2/kWh")
        except Exception as e:
            _print(f"  ERROR: {e}")

    _header("Demo Complete!")
    _print("  HarchOS: The Only GPU Orchestration Platform with Native Carbon-Aware Scheduling")
    _print("  No competitor offers this. This is our competitive moat.")


def main() -> None:
    parser = argparse.ArgumentParser(description="HarchOS Carbon-Aware Demo")
    parser.add_argument(
        "--api-key", default="hsk_test_development_key_12345",
        help="HarchOS API key (default: test key)",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000/v1",
        help="HarchOS API base URL (default: http://localhost:8000/v1)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_demo(args.api_key, args.base_url))
    except KeyboardInterrupt:
        _print("\n\n  Demo interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
