PROJECT_ID=alfredlabs-model-training
sudo docker build -t asia.gcr.io/$PROJECT_ID/bitplanes-build:latest .
sudo gcloud docker -- push asia.gcr.io/$PROJECT_ID/bitplanes-build:latest
