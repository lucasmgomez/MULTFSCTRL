#!/usr/bin/env python3
"""
Test Runner for Nilearn GLM Analysis and Evaluation

This script runs the complete Nilearn GLM analysis pipeline:
1. GLM fitting with Nilearn FirstLevelModel
2. Performance evaluation and diagnostics
3. Comparison with original implementation (if available)

Usage:
python run_nilearn_analysis.py --subj sub-01 --task ctxdm --run_evaluation
"""

import argparse
import subprocess
import sys
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Run complete Nilearn GLM analysis pipeline")
    parser.add_argument("--subj", default="sub-01", help="Subject ID")
    parser.add_argument("--task", default="ctxdm", help="Task name")
    parser.add_argument("--ses", default="ses-001", help="Session (optional, will process all if not specified)")
    parser.add_argument("--run", default=None, help="Run (optional, will process all if not specified)")
    parser.add_argument("--include_types", nargs="+", default=["encoding", "delay"], 
                       help="Event types to include")
    parser.add_argument("--correct_only", action="store_true", help="Use only correct trials")
    parser.add_argument("--run_evaluation", action="store_true", help="Run evaluation after GLM analysis")
    parser.add_argument("--compare_original", action="store_true", help="Compare with original GLM implementation")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    return parser.parse_args()

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Command completed successfully!")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    args = get_args()
    
    scripts_dir = Path(__file__).parent
    
    print("🧠 Nilearn GLM Analysis Pipeline")
    print(f"Subject: {args.subj}")
    print(f"Task: {args.task}")
    print(f"Session: {args.ses}")
    print(f"Run: {args.run if args.run else 'all'}")
    print(f"Event types: {', '.join(args.include_types)}")
    print(f"Correct trials only: {args.correct_only}")
    
    # Step 1: Run Nilearn GLM analysis
    glm_cmd = [
        "python", str(scripts_dir / "glm_analysis_nilearn.py"),
        "--subj", args.subj,
        "--tasks", args.task
    ]
    
    if args.ses:
        # Note: The script will process all sessions by default, but we can filter in the future
        pass
    
    glm_cmd.extend(["--include_types"] + args.include_types)
    
    if args.correct_only:
        glm_cmd.append("--correct_only")
    
    if args.overwrite:
        glm_cmd.append("--overwrite")
    
    success = run_command(glm_cmd, "Running Nilearn GLM Analysis")
    
    if not success:
        print("\n❌ GLM analysis failed. Stopping pipeline.")
        sys.exit(1)
    
    # Step 2: Run evaluation (if requested)
    if args.run_evaluation:
        # Determine which runs to evaluate
        data_root = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/nilearn_data/trial_level_betas")
        subj_dir = data_root / args.subj
        
        if not subj_dir.exists():
            print(f"⚠️ No results found in {subj_dir}")
            return
        
        # Find GLM result files
        pattern = f"{args.subj}_*_task-{args.task}_*_nilearn_betas.h5"
        glm_files = list(subj_dir.glob(f"**/func/{pattern}"))
        
        if not glm_files:
            print(f"⚠️ No GLM result files found matching pattern: {pattern}")
            return
        
        print(f"\n📊 Found {len(glm_files)} GLM result files to evaluate")
        
        evaluation_success = True
        for glm_file in glm_files:
            # Parse file components
            filename = glm_file.stem.replace("_nilearn_betas", "")
            parts = filename.split("_")
            
            if len(parts) >= 4:
                file_subj = parts[0]
                file_ses = parts[1]  
                file_task = parts[2].replace("task-", "")
                file_run = parts[3]
                
                # Skip if specific run requested and doesn't match
                if args.run and file_run != args.run:
                    continue
                
                # Run evaluation
                eval_cmd = [
                    "python", str(scripts_dir / "evaluate_nilearn_glm.py"),
                    "--subj", file_subj,
                    "--task", file_task,
                    "--ses", file_ses,
                    "--run", file_run
                ]
                
                if args.compare_original:
                    eval_cmd.append("--compare_original")
                
                eval_success = run_command(
                    eval_cmd, 
                    f"Evaluating GLM results: {file_subj} {file_ses} {file_task} {file_run}"
                )
                
                if not eval_success:
                    print(f"⚠️ Evaluation failed for {filename}")
                    evaluation_success = False
        
        if evaluation_success:
            print("\n✅ All evaluations completed successfully!")
        else:
            print("\n⚠️ Some evaluations failed. Check output above for details.")
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎯 PIPELINE SUMMARY")
    print('='*60)
    print("✅ Nilearn GLM analysis completed")
    
    if args.run_evaluation:
        print("✅ Performance evaluation completed")
    
    output_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/data/nilearn_data")
    print(f"\n📁 Results saved to: {output_dir}")
    
    if args.run_evaluation:
        eval_dir = Path("/project/def-pbellec/xuan/fmri_dataset_project/results/nilearn_evaluation")
        print(f"📊 Evaluation results: {eval_dir}")
    
    print(f"\n🔍 To check results:")
    print(f"   ls -la {output_dir}/trial_level_betas/{args.subj}/")
    
    if args.run_evaluation:
        print(f"   ls -la {eval_dir}/")

if __name__ == "__main__":
    main()