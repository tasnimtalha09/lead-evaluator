# importing the necessary libraries
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# importing the predict_leads function from the pipeline module
from pipeline import predict_leads


# parsing the command-line arguments for the scoring script
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description = "Score Swan Chemical leads using the saved Lead Evaluator artifacts.")

    # positional argument for the input CSV file path
    parser.add_argument(
        "input_csv",
        type = str,
        help = "Path to the input CSV file containing new leads.",
    )

    # optional argument to save output
    parser.add_argument(
        "--out",
        type = str,
        default = None,
        help = "Optional path to save the scored output as CSV.",
    )

    return parser.parse_args()


# creating the main function to load leads, score them, and display/save results
def main() -> None:

    # parsing command-line arguments
    args = parse_args()

    # resolving the input CSV path
    input_path = Path(args.input_csv).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # loading the new leads from the CSV file
    new_leads = pd.read_csv(filepath_or_buffer = input_path)

    # if the file contains a "lead_score" column, rename it to preserve for comparison
    if "lead_score" in new_leads.columns:
        new_leads = new_leads.rename(columns = {"lead_score": "actual_lead_score"})

    # predicting lead buckets, scores, and actions
    results = predict_leads(new_leads)

    # building the output table
    out = pd.DataFrame({"Company Name": results["company_name"]})

    # including actual bucket if it exists
    if "lead_bucket" in results.columns:
        out["Actual Lead Bucket"] = results["lead_bucket"]

    # adding predicted bucket
    out["Predicted Lead Bucket"] = results["pred_bucket"]

    # calculating bucket match/mismatch if actual buckets exist
    if "Actual Lead Bucket" in out.columns:
        out.insert(
            loc = out.columns.get_loc("Predicted Lead Bucket") + 1,
            column = "Status",
            value = np.where(
                out["Actual Lead Bucket"] == out["Predicted Lead Bucket"],
                "Match",
                "Mismatch"
            )
        )

    # including actual lead score if it exists
    if "actual_lead_score" in results.columns:
        out["Actual Lead Score"] = results["actual_lead_score"]

    # adding predicted lead score
    out["Predicted Lead Score"] = results["lead_score"]

    # calculating score difference if actual scores exist
    if "Actual Lead Score" in out.columns:
        out["Score Difference"] = (
            out["Actual Lead Score"] - out["Predicted Lead Score"]
        ).abs().round(decimals = 1)

    # adding the recommended action
    out["Action"] = results["action"]

    # printing the results table
    print(out.to_string(index = False))

    # computing bucket accuracy if actual labels exist
    if len(out) > 1 and "Actual Lead Bucket" in out.columns:
        matches = (out["Actual Lead Bucket"].values == out["Predicted Lead Bucket"].values)
        correct = int(matches.sum())
        total = int(len(out))
        acc = 100.0 * correct / total
        print(f"\nBucket Accuracy: {acc:.0f}% ({correct}/{total})")

    # computing average score difference if actual scores exist
    if len(out) > 1 and "Score Difference" in out.columns:
        avg_diff = float(out["Score Difference"].mean())
        print(f"Average Score Difference: {avg_diff:.1f} points")

    # saving output to CSV if requested
    if args.out is not None:
        out_path = Path(args.out).expanduser().resolve()
        out.to_csv(out_path, index = False)
        print(f"\nSaved scored output to: {out_path}")


# running the main function when the script is executed directly
if __name__ == "__main__":
    main()