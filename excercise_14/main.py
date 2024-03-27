import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    steps_to_execute=config["main"]["execute_steps"]
    print(steps_to_execute)
    # if isinstance(config["main"]["execute_steps"], str):
    #     # This was passed on the command line as a comma-separated list of steps
    #     steps_to_execute = config["main"]["execute_steps"].split(",")
    # else:
    #     assert isinstance(config["main"]["execute_steps"], list)
    #     steps_to_execute = config["main"]["execute_steps"]

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            "main",
            parameters={
                "input_artifact": "raw_data.parquet:latest",
                "artifact_name": "clean_data",
                "artifact_type": "preprocess_data",
                "artifact_description": "Data after cleaning step",
            }
        )

    if "check_data" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "reference_artifact": "clean_data:latest",
                "sample_artifact": "test_data",
                "ks_alpha": 0.05
                }
        )

    if "segregate" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "input_artifact":"clean_data:latest",
                "artifact_root":"data",
                "artifact_type":"segregated_data",
                "test_size":0.2,
                "random_state": 42,
                "stratify":"null"
            }
        )

    if "random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "train_data": "data_train.csv:latest",
                "model_config":model_config,
                "export_artifact":config["random_forest_pipeline"]["export_artifact"],
                "random_seed":42,
                "val_size":0.2,
                "stratify": config["data"]["stratify"]
            }
        )

    if "evaluate" in steps_to_execute:

        _ =  mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "model_export":f"{config['random_forest_pipeline']['export_artifact']}:latest",
                "test_data": "data_test.csv:latest",
            }
        )


if __name__ == "__main__":
    go()
