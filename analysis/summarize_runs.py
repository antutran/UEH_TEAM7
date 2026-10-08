#!/usr/bin/env python3
"""
UEH CRC 2026 Simulation Run Summarizer.

Reads all measured run CSV files and metadata JSON files from analysis/data/,
computes descriptive statistics across all runs using pure standard library (no extra dependencies),
produces:
  - analysis/data/run_summary.csv
  - analysis/data/summary_statistics.csv
and displays a formatted ASCII summary table.
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys


def summarize(data_dir):
    data_dir = os.path.abspath(data_dir)
    csv_files = sorted(glob.glob(os.path.join(data_dir, 'run_*.csv')))
    csv_files = [f for f in csv_files if not f.endswith('run_summary.csv') and not f.endswith('summary_statistics.csv')]

    if not csv_files:
        print(f"No run CSV files found in {data_dir}")
        return

    run_records = []

    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        run_id = os.path.splitext(basename)[0]
        meta_file = os.path.join(data_dir, f"{run_id}_metadata.json")

        metadata = {}
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: could not load metadata {meta_file}: {e}")

        # Read CSV data
        rows = []
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        if not rows:
            continue

        duration_sec = metadata.get('duration_sec')
        if duration_sec is None:
            duration_sec = float(rows[-1]['elapsed_sec'])

        minutes = int(duration_sec // 60)
        seconds = duration_sec % 60
        duration_min = f"{minutes:02d}:{seconds:05.2f}"

        cum_dist = metadata.get('cumulative_distance_m')
        if cum_dist is None:
            cum_dist = float(rows[-1]['cumulative_distance_m'])

        lin_speeds = [float(r['linear_x']) for r in rows if 'linear_x' in r]
        ang_speeds = [abs(float(r['angular_z'])) for r in rows if 'angular_z' in r]

        mean_lin_spd = statistics.mean(lin_speeds) if lin_speeds else 0.0
        max_lin_spd = max(lin_speeds) if lin_speeds else 0.0
        mean_ang_spd = statistics.mean(ang_speeds) if ang_speeds else 0.0
        max_ang_spd = max(ang_speeds) if ang_speeds else 0.0

        completed = metadata.get('completed', True)
        timeout = metadata.get('timeout', False)
        notes = metadata.get('notes', 'Normal valid autonomous run' if completed else 'Incomplete')

        record = {
            'run_id': run_id,
            'completed': bool(completed),
            'duration_sec': round(float(duration_sec), 4),
            'duration_min': duration_min,
            'cumulative_distance_m': round(float(cum_dist), 4),
            'mean_linear_speed_mps': round(mean_lin_spd, 4),
            'max_linear_speed_mps': round(max_lin_spd, 4),
            'mean_abs_angular_speed_rps': round(mean_ang_spd, 4),
            'max_abs_angular_speed_rps': round(max_ang_spd, 4),
            'timeout': bool(timeout),
            'notes': str(notes),
        }
        run_records.append(record)

    run_summary_path = os.path.join(data_dir, 'run_summary.csv')
    fieldnames = [
        'run_id', 'completed', 'duration_sec', 'duration_min',
        'cumulative_distance_m', 'mean_linear_speed_mps', 'max_linear_speed_mps',
        'mean_abs_angular_speed_rps', 'max_abs_angular_speed_rps', 'timeout', 'notes'
    ]
    with open(run_summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_records)
    print(f"Saved run summary to: {run_summary_path}\n")

    valid_runs = [r for r in run_records if r['completed']]
    total_attempted = len(run_records)
    valid_count = len(valid_runs)
    success_rate = (valid_count / total_attempted * 100.0) if total_attempted > 0 else 0.0

    # Print Table
    print("=" * 86)
    print(f"{'Run ID':<10} {'Completed':<10} {'Duration (s)':<14} {'Duration (min)':<16} {'Distance (m)':<14} {'Notes':<18}")
    print("-" * 86)
    for r in run_records:
        comp_str = "YES" if r['completed'] else "NO"
        print(f"{r['run_id']:<10} {comp_str:<10} {r['duration_sec']:<14.2f} {r['duration_min']:<16} {r['cumulative_distance_m']:<14.2f} {r['notes']:<18}")
    print("=" * 86)

    if valid_count > 0:
        durations = [r['duration_sec'] for r in valid_runs]
        distances = [r['cumulative_distance_m'] for r in valid_runs]

        mean_dur = statistics.mean(durations)
        median_dur = statistics.median(durations)
        std_dur = statistics.stdev(durations) if valid_count > 1 else 0.0
        min_dur = min(durations)
        max_dur = max(durations)

        mean_dist = statistics.mean(distances)
        std_dist = statistics.stdev(distances) if valid_count > 1 else 0.0

        def fmt_min(sec):
            m = int(sec // 60)
            s = sec % 60
            return f"{m:02d}:{s:05.2f}"

        stats = [
            {'metric': 'Total Runs Attempted', 'value': str(total_attempted)},
            {'metric': 'Valid Completed Runs', 'value': str(valid_count)},
            {'metric': 'Success Rate (%)', 'value': f"{success_rate:.1f}%"},
            {'metric': 'Mean Completion Time (s)', 'value': f"{mean_dur:.2f} ({fmt_min(mean_dur)})"},
            {'metric': 'Median Completion Time (s)', 'value': f"{median_dur:.2f} ({fmt_min(median_dur)})"},
            {'metric': 'Std Deviation Completion Time (s)', 'value': f"{std_dur:.2f}"},
            {'metric': 'Fastest Run (s)', 'value': f"{min_dur:.2f} ({fmt_min(min_dur)})"},
            {'metric': 'Slowest Run (s)', 'value': f"{max_dur:.2f} ({fmt_min(max_dur)})"},
            {'metric': 'Mean Cumulative Distance (m)', 'value': f"{mean_dist:.2f}"},
            {'metric': 'Std Deviation Distance (m)', 'value': f"{std_dist:.2f}"},
        ]

        stats_csv_path = os.path.join(data_dir, 'summary_statistics.csv')
        with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
            writer.writeheader()
            writer.writerows(stats)
        print(f"\nSaved summary statistics to: {stats_csv_path}\n")

        print("=" * 60)
        print("                  SUMMARY STATISTICS                  ")
        print("=" * 60)
        for s in stats:
            print(f"  {s['metric']:<36}: {s['value']}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Summarize UEH CRC 2026 Simulation Runs")
    parser.add_argument('--data-dir', type=str, default='analysis/data', help="Directory containing run CSVs")
    args = parser.parse_args()
    summarize(args.data_dir)


if __name__ == '__main__':
    main()
