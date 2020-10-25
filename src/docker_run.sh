# 要求：Secret Managerに値を設定
IEX_TOKEN=$(gcloud secrets versions access 1 --secret IEX_TOKEN)
SLACK_TOKEN=$(gcloud secrets versions access 1 --secret SLACK_TOKEN)

docker run --rm -it --env IEX_TOKEN=$IEX_TOKEN --env SLACK_TOKEN=$SLACK_TOKEN minervini_screening:latest --country ja