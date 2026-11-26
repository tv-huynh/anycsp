from argparse import ArgumentParser
from pathlib import Path


def read_output(path):
    """Read the final status and final assignment line."""
    status = None
    assignment = None

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)

            if not parts:
                continue

            if parts[0] == "s":
                status = parts[1].strip() if len(parts) == 2 else ""

            elif parts[0] == "v":
                assignment = parts[1].strip() if len(parts) == 2 else ""

    return status, assignment


def validate_assignment(assignment, path):
    if not assignment:
        raise ValueError(f"{path}: missing or empty assignment")

    if any(bit not in "01" for bit in assignment):
        raise ValueError(f"{path}: expected a compact Boolean bitstring after 'v'")


def main():
    parser = ArgumentParser()
    parser.add_argument("--neural_dir",type=Path,required=True,help="Directory containing neural validation .out files")
    parser.add_argument("--bnb_dir",type=Path,required=True,help="Directory containing corresponding optimal BnB .out files",)
    parser.add_argument("--accuracy_mode",choices=["bit", "exact"],default="exact",help="Compare individual bits or complete assignments (default: exact)")
    args = parser.parse_args()

    if not args.neural_dir.is_dir():
        parser.error(f"Directory not found: {args.neural_dir}")

    if not args.bnb_dir.is_dir():
        parser.error(f"Directory not found: {args.bnb_dir}")

    neural_files = sorted(args.neural_dir.glob("*.out"))

    if not neural_files:
        parser.error("No neural .out files found.")

    num_compared = 0
    num_unknown = 0
    num_exact_matches = 0
    num_matching_bits = 0
    num_total_bits = 0

    for neural_path in neural_files:
        neural_status, neural_assignment = read_output(neural_path)

        if neural_status == "UNKNOWN":
            num_unknown += 1
            continue

        if neural_status != "SATISFIABLE":
            raise ValueError(f"{neural_path}: unexpected status {neural_status!r}")

        bnb_path = args.bnb_dir / neural_path.name

        if not bnb_path.is_file():
            raise FileNotFoundError(f"Missing corresponding BnB output: {bnb_path}")

        bnb_status, bnb_assignment = read_output(bnb_path)

        if bnb_status != "OPTIMUM FOUND":
            raise ValueError(f"{bnb_path}: expected 's OPTIMUM FOUND', found {bnb_status!r}")

        validate_assignment(neural_assignment, neural_path)
        validate_assignment(bnb_assignment, bnb_path)

        if len(neural_assignment) != len(bnb_assignment):
            raise ValueError(f"{neural_path.name}: assignment lengths differ: neural={len(neural_assignment)}, BnB={len(bnb_assignment)}")

        num_compared += 1

        num_exact_matches += int(neural_assignment == bnb_assignment)

        num_matching_bits += sum(neural_bit == bnb_bit for neural_bit, bnb_bit in zip(neural_assignment, bnb_assignment))
        num_total_bits += len(bnb_assignment)

    if num_compared == 0:
        raise ValueError("No feasible neural assignments found to compare.")

    print(f"Neural output files: {len(neural_files)}")
    print(f"Skipped UNKNOWN: {num_unknown}")
    print(f"Compared feasible assignments: {num_compared}")

    if args.accuracy_mode == "exact":
        accuracy = num_exact_matches / num_compared
        print(f"Exact matches: {num_exact_matches}/{num_compared}")
        print(f"Exact-match accuracy: {accuracy:.2%}")
    else:
        accuracy = num_matching_bits / num_total_bits
        print(f"Matching bits: {num_matching_bits}/{num_total_bits}")
        print(f"Per-bit accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()