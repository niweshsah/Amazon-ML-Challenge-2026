"""Challenge-compatible command line validator. IDs are checked by default."""
import argparse
from pathlib import Path
from entity_resolution.submission import validate

parser = argparse.ArgumentParser()
parser.add_argument('--matching', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--test-dir', type=Path, required=True)
parser.add_argument('--check-ids', action='store_true', help='Accepted for compatibility; always enabled')
args = parser.parse_args()
if args.matching.name != 'matching_results.tsv' or args.candidate.name != 'candidate_pairs.tsv' or args.matching.parent != args.candidate.parent:
    parser.error('Provide the two standard filenames in the same output directory')
try:
    print(validate(args.test_dir, args.matching.parent)['status'])
except (ValueError, OSError, KeyError) as exc:
    parser.exit(1, f'FAIL: {exc}\n')
