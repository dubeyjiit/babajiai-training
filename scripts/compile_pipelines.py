import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from kfp import compiler

from pipelines.training_pipeline  import training_pipeline
from pipelines.inference_pipeline import inference_pipeline

COMPILED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "compiled_pipelines")


def compile_pipelines(upload: bool = False, gcs_prefix: str = "") -> None:
    os.makedirs(COMPILED_DIR, exist_ok=True)

    pipelines_to_compile = [
        ("training_pipeline.yaml",  training_pipeline),
        ("inference_pipeline.yaml", inference_pipeline),
    ]

    compiled_paths = []
    for filename, pipeline_func in pipelines_to_compile:
        out_path = os.path.join(COMPILED_DIR, filename)
        # KFP v2 @pipeline wraps functions into GraphComponent — use .name not __name__
        pipeline_name = getattr(pipeline_func, "name", None) or getattr(pipeline_func, "__name__", filename)
        print(f"Compiling {pipeline_name} → {out_path}")
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=out_path)
        compiled_paths.append((filename, out_path))
        print(f"  OK: {out_path}")

    if upload and gcs_prefix:
        from google.cloud import storage
        bucket_name = gcs_prefix.replace("gs://", "").split("/")[0]
        prefix      = "/".join(gcs_prefix.replace("gs://", "").split("/")[1:])
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        for filename, local_path in compiled_paths:
            blob_path = f"{prefix}/{filename}" if prefix else filename
            bucket.blob(blob_path).upload_from_filename(local_path)
            print(f"Uploaded → gs://{bucket_name}/{blob_path}")

    print("\nAll pipelines compiled successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload",     action="store_true", help="Upload to GCS after compiling")
    parser.add_argument("--gcs-prefix", default="",          help="GCS prefix for upload (e.g. gs://bucket/compiled)")
    args = parser.parse_args()
    compile_pipelines(upload=args.upload, gcs_prefix=args.gcs_prefix)