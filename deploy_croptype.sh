# exit when any command fails
set -e

export TAG=us-central1-docker.pkg.dev/bsos-geog-harvest1/crop-type/crop-type
export BUCKET=crop-type-earthengine
export URL="https://crop-type-grxg7bzh2a-uc.a.run.app"
export MODELS=$(
        python -c \
        "from pathlib import Path; \
        print(' '.join([p.stem for p in Path('crop_classification/models').glob('*.pt')]))"
)

docker build -f Dockerfile.croptype . --build-arg MODELS="$MODELS" -t $TAG
docker push $TAG
gcloud run deploy crop-type --image ${TAG}:latest \
        --cpu=4 \
        --memory=8Gi \
        --platform=managed \
        --region=us-central1 \
        --allow-unauthenticated \
        --concurrency 10 \
        --port 8080

gcloud run deploy crop-type-management-api --image ${TAG}:latest \
        --memory=4Gi \
        --platform=managed \
        --region=us-central1 \
        --allow-unauthenticated \
        --port 8081

gcloud functions deploy trigger-croptype \
    --source=crop_classification/trigger_inference_gcp \
    --trigger-bucket=$BUCKET \
    --allow-unauthenticated \
    --runtime=python39 \
    --entry-point=hello_gcs \
    --set-env-vars MODELS="$MODELS",INFERENCE_HOST="$URL" \
    --timeout=300s