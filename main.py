import argparse
import os
from omegaconf import OmegaConf
from src.run_federated import run_full_experiment
from src.run_centralized import run_centralized
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Federated Waveform Inversion experiments.")

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the OmegaConf YAML configuration file."
    )

    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=['marmousi', 'overthrust', 'foothill', 'bpsalt', 'combined'],
        help="Family: ['marmousi', 'overthrust', 'foothill', 'bpsalt', 'combined']"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['federated', 'centralized'],
        help="Mode: 'federated' for FL experiments, 'centralized' for centralized baseline"
    )

    # Add batch size for centralized runs (supports integer or 'max')
    parser.add_argument(
        "--batch_size",
        type=str,
        required=False,
        help="Batch size for centralized runs; use an integer or 'max' to use all instances"
    )
    # Also accept hyphenated form for convenience
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=str,
        required=False,
        help=argparse.SUPPRESS
    )

    # Parse known args; extra positional args are OmegaConf dotlist overrides
    # e.g. experiment.reg_lambda=0.5 federated.local_lr=0.005
    args, overrides = parser.parse_known_args()

    # Load base config and merge CLI overrides
    config = OmegaConf.load(args.config_path)
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, cli_conf)
        print(f"Applied CLI overrides: {overrides}")

    target_families_env = os.environ.get("RFL_TARGET_FAMILIES")
    target_instances_env = os.environ.get("RFL_TARGET_INSTANCES")

    # Order must match scripts/create_combined_dataset.py FAMILIES list
    COMBINED_FAMILIES = ['foothill', 'marmousi', 'overthrust', 'bpsalt']

    if args.family == 'combined':
        families_to_run = COMBINED_FAMILIES
    else:
        families_to_run = [args.family]

    if args.mode == 'federated':
        resolved_target_instances = (
            [int(x) for x in target_instances_env.split(',')]
            if target_instances_env else None
        )

        if args.family == 'combined':
            # Batched mode: one FL simulation for all families
            run_full_experiment(
                config_override=config,
                target_families=families_to_run,
                target_instances=resolved_target_instances,
                batched=True,
            )
        else:
            # Single-family: sequential mode (one simulation per family)
            for fam in families_to_run:
                fam_config = OmegaConf.merge(config, OmegaConf.create({
                    "path": {
                        "velocity_data_path": f"dataset/{fam}/velocity_model",
                        "client_seismic_data_path": f"dataset/{fam}/seismic_data/{config.experiment.scenario_flag}",
                        "gt_seismic_data_path": f"dataset/{fam}/seismic_data/gt",
                        "output_path": f"experiment_result/{fam}/",
                    }
                }))

                run_full_experiment(
                    config_override=fam_config,
                    target_families=[fam],
                    target_instances=resolved_target_instances,
                )

    elif args.mode == 'centralized':
        for fam in families_to_run:
            fam_config = OmegaConf.merge(config, OmegaConf.create({
                "path": {
                    "velocity_data_path": f"dataset/{fam}/velocity_model",
                    "client_seismic_data_path": f"dataset/{fam}/seismic_data/{config.experiment.scenario_flag}",
                    "gt_seismic_data_path": f"dataset/{fam}/seismic_data/gt",
                    "output_path": f"experiment_result/{fam}/",
                }
            }))

            run_centralized(
                config_override=fam_config,
                family=fam,
                batch_size=args.batch_size,
            )
