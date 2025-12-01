"""
Verification script to check if all expected outputs are generated correctly.
"""
import os
import json
from pathlib import Path

def verify_outputs():
    """Verify all expected outputs exist and are valid."""
    
    print("="*70)
    print("PROJECT OUTPUT VERIFICATION")
    print("="*70)
    print()
    
    checks = {}
    issues = []
    
    # Note: This script should be run from project root, not results/ directory
    # 1. Check models directory
    print("1. Checking models/ directory...")
    if os.path.exists("models"):
        model_files = list(Path("models").glob("*.zip"))
        if model_files:
            checks["models"] = True
            print(f"   [OK] Found {len(model_files)} model file(s)")
            # Check file sizes
            for mf in model_files[:5]:  # Check first 5
                size_mb = mf.stat().st_size / (1024 * 1024)
                if size_mb > 0:
                    print(f"      - {mf.name}: {size_mb:.2f} MB")
                else:
                    issues.append(f"   [WARNING] {mf.name} has zero size")
        else:
            checks["models"] = False
            issues.append("   [ERROR] No model files found in models/")
    else:
        checks["models"] = False
        issues.append("   [ERROR] models/ directory not found")
    
    # 2. Check runs directory
    print("\n2. Checking results/runs/ directory (TensorBoard logs)...")
    if os.path.exists("results/runs"):
        run_dirs = [d for d in Path("results/runs").iterdir() if d.is_dir()]
        if run_dirs:
            checks["runs"] = True
            print(f"   [OK] Found {len(run_dirs)} TensorBoard run directory(ies)")
            # Check for event files
            for rd in run_dirs[:3]:  # Check first 3
                event_files = list(rd.glob("events.out.tfevents.*"))
                if event_files:
                    print(f"      - {rd.name}: {len(event_files)} event file(s)")
                else:
                    issues.append(f"   [WARNING] {rd.name} has no event files")
        else:
            checks["runs"] = False
            issues.append("   [ERROR] No run directories found in runs/")
    else:
        checks["runs"] = False
        issues.append("   [ERROR] runs/ directory not found")
    
    # 3. Check training_results.json
    print("\n3. Checking results/training_results.json...")
    if os.path.exists("results/training_results.json"):
        try:
            with open("results/training_results.json", 'r') as f:
                data = json.load(f)
                checks["training_results"] = True
                print("   [OK] training_results.json exists and is valid JSON")
                # Check for expected keys
                if "training_summary" in data or "evaluation" in data or "training_stats" in data:
                    print("      - Contains training/evaluation data")
                else:
                    issues.append("   [WARNING] training_results.json missing expected keys")
        except json.JSONDecodeError as e:
            checks["training_results"] = False
            issues.append(f"   [ERROR] training_results.json is not valid JSON: {e}")
        except Exception as e:
            checks["training_results"] = False
            issues.append(f"   [ERROR] Error reading training_results.json: {e}")
    else:
        checks["training_results"] = False
        issues.append("   [ERROR] training_results.json not found")
    
    # 4. Check plots directory
    print("\n4. Checking results/plots/ directory...")
    if os.path.exists("results/plots"):
        plot_files = list(Path("results/plots").glob("*.png"))
        if plot_files:
            checks["plots"] = True
            print(f"   [OK] Found {len(plot_files)} plot file(s)")
            for pf in plot_files:
                size_kb = pf.stat().st_size / 1024
                print(f"      - {pf.name}: {size_kb:.2f} KB")
        else:
            checks["plots"] = False
            issues.append("   [ERROR] No plot files found in plots/")
    else:
        checks["plots"] = False
        issues.append("   [ERROR] plots/ directory not found")
    
    # 5. Check documentation reports
    print("\n5. Checking documentation reports...")
    report_files = [
        "TRAINING_REPORT.md",
        "TRAINING_RESULTS_EXPLANATION.md",
        "TRAINING_METRICS.md"
    ]
    found_reports = []
    for report in report_files:
        if os.path.exists(report):
            found_reports.append(report)
            size_kb = Path(report).stat().st_size / 1024
            print(f"   [OK] {report} exists ({size_kb:.2f} KB)")
        else:
            issues.append(f"   [WARNING] {report} not found")
    
    if found_reports:
        checks["reports"] = True
    else:
        checks["reports"] = False
        issues.append("   [ERROR] No documentation reports found")
    
    # 6. Check config files (inputs, but should exist)
    print("\n6. Checking config/ directory (inputs)...")
    config_files = [
        "configs/agent_params.json",
        "configs/environment_params.yaml",
        "configs/hyper_params.yaml",
        # Note: Reward functions are now in configs/rewards/
        # Hyperparameters are in configs/hyperparams/
        "configs/reward_function.py"
    ]
    found_configs = []
    for config in config_files:
        if os.path.exists(config):
            found_configs.append(config)
        else:
            issues.append(f"   ⚠️  {config} not found")
    
    if len(found_configs) == len(config_files):
        checks["configs"] = True
        print(f"   [OK] All {len(config_files)} config files exist")
    else:
        checks["configs"] = False
        print(f"   [WARNING] Only {len(found_configs)}/{len(config_files)} config files found")
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    required_checks = ["models", "runs", "training_results"]
    recommended_checks = ["plots", "reports"]
    
    passed_required = sum(checks.get(k, False) for k in required_checks)
    passed_recommended = sum(checks.get(k, False) for k in recommended_checks)
    total_required = len(required_checks)
    total_recommended = len(recommended_checks)
    
    print(f"\nRequired Outputs: {passed_required}/{total_required} [OK]")
    for check in required_checks:
        status = "[OK]" if checks.get(check, False) else "[ERROR]"
        print(f"  {status} {check}")
    
    print(f"\nRecommended Outputs: {passed_recommended}/{total_recommended} [OK]")
    for check in recommended_checks:
        status = "[OK]" if checks.get(check, False) else "[WARNING]"
        print(f"  {status} {check}")
    
    if issues:
        print(f"\n[WARNING] Issues Found ({len(issues)}):")
        for issue in issues:
            print(issue)
    
    if passed_required == total_required:
        print("\n[SUCCESS] All required outputs are present!")
        if passed_recommended == total_recommended:
            print("[SUCCESS] All recommended outputs are also present!")
            return True
        else:
            print("[INFO] Some recommended outputs are missing (optional)")
            return True
    else:
        print(f"\n[ERROR] Missing {total_required - passed_required} required output(s)")
        return False

if __name__ == "__main__":
    success = verify_outputs()
    exit(0 if success else 1)

