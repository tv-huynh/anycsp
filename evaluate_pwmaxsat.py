from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from src.model.model import ANYCSP
from src.csp.csp_data import CSP_Data
from src.data.dataset import File_Dataset


def extract_best_feasible_assignment(data):
    """
    Return the lowest-cost feasible assignment across all parallel
    runs and recorded steps, or None if none was found.
    """
    assignments = data.all_assignments.detach().cpu()
    costs = data.all_num_unsat.detach().cpu()
    hard_counts = data.all_hard_unsat.detach().cpu()
    variable_batch = data.batch.detach().cpu()

    runs, steps = torch.where(hard_counts == 0)

    if runs.numel() == 0:
        return None

    best = costs[runs, steps].argmin().item()
    run = runs[best].item()
    step = steps[best].item()

    return {
        "assignment": assignments[
            variable_batch == run, step
        ].clone(),
        "soft_cost": costs[run, step].item(),
        "run": run,
        "step": step,
    }

def write_solution(output_path, solution):
    """Write the requested compact Boolean-assignment format."""
    if solution is None:
        output_path.write_text("s UNKNOWN\n", encoding="utf-8")
        return

    assignment = solution["assignment"]

    if not torch.all((assignment == 0)|(assignment == 1)).item():
        raise ValueError(f"Cannot write {output_path}: assignment contains non-Boolean domain values.")

    bitstring = "".join(
        str(int(value))
        for value in assignment.tolist()
    )

    # This is a feasible solution, not an optimality certificate.
    output_path.write_text(
        f"s SATISFIABLE\n"
        f"o {solution['soft_cost']:g}\n"
        f"v {bitstring}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Model directory")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--output_dir",type=str,default=None,help="Directory to save the assignments, not saved if None",)
    parser.add_argument("--checkpoint_name", type=str, default='best', help="Checkpoint to be used")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--network_steps", type=int, default=100000, help="Number of network steps during evaluation")
    parser.add_argument("--num_boost", type=int, default=1, help="Number of parallel evaluate runs")
    parser.add_argument("--verbose", action='store_true', default=False, help="Output intermediate optima")
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout in seconds")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ANYCSP.load_model(args.model_dir,args.checkpoint_name,)
    model.eval()
    model.to(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = File_Dataset(args.data_path)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    num_feasible = 0
    total_time = 0.0
    num_total = len(dataset)

    for data in dataset:
        file = Path(data.path)
        filename = file.name
        if args.output_dir is not None:
            output_name = f"{file.stem}.out"
            output_path = output_dir / output_name

        if args.num_boost > 1:
            data = CSP_Data.collate([data for _ in range(args.num_boost)])

        data.to(device)

        if args.verbose:
            print(f"Solving {file}:")

        with torch.inference_mode():
            data = model(
                data,
                args.network_steps,
                return_all_assignments=True,
                return_log_probs=False,
                # Continue searching after reaching feasibility.
                stop_early=False,
                return_all_unsat=True,
                verbose=args.verbose,
                keep_time=True,
                timeout=args.timeout,
            )

        solution = extract_best_feasible_assignment(data)
        if args.output_dir is not None:
            write_solution(output_path, solution)

        total_time += data.opt_time

        if solution is None:
            print(
                f"{filename}: no feasible assignment found, "
                f"Steps: {data.num_steps}, "
                f"Time: {data.total_time:.2f}s"
            )
        else:
            num_feasible += 1
            print(
                f"{filename}: feasible, "
                f"Soft cost: {solution['soft_cost']:g}, "
                f"Step: {solution['step']}, "
                f"Run: {solution['run']}, "
                f"Time: {data.total_time:.2f}s"
            )

    print(
        f"Feasible: {100 * num_feasible / num_total:.2f}%, Average time (if feasible): {total_time / num_total:.2f}s")
