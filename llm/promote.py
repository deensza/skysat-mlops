"""Make a registered version the champion: python -m llm.promote 2   (make llm-promote VERSION=2)

Only versions that passed the gate are registered (llm/train_lora.py), so this is the release step for a version that
was trained with --no-promote. The API notices the new alias within MODEL_RELOAD_SECONDS and switches adapters.
"""
import argparse

MODEL_NAME = "skysat-reviews"
ALIAS = "champion"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("version", help="the registered version to make champion, e.g. 2")
    args = p.parse_args()

    import mlflow
    from mlflow.exceptions import MlflowException
    client = mlflow.MlflowClient()
    print(f"MLflow: {mlflow.get_tracking_uri()}")
    client.get_model_version(MODEL_NAME, args.version)  # fails loudly if the version does not exist
    try:
        before = str(client.get_model_version_by_alias(MODEL_NAME, ALIAS).version)
    except MlflowException:
        before = "none"
    if before == args.version:
        print(f"Version {args.version} is already the {ALIAS}")
        return
    client.set_registered_model_alias(MODEL_NAME, ALIAS, args.version)
    print(f"{MODEL_NAME}@{ALIAS}: version {before} -> version {args.version}")


if __name__ == "__main__":
    main()
