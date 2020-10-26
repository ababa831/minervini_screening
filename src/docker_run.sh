# 要求：Secret Managerに値を設定
IEX_TOKEN=$(gcloud secrets versions access 1 --secret IEX_TOKEN)
SLACK_TOKEN=$(gcloud secrets versions access 1 --secret SLACK_TOKEN)

ja_searchtime_start=$(date "+%Y-%m-%d 16:00:00")
ja_searchtime_end=$(date "+%Y-%m-%d 23:59:59")
us_searchtime_start=$(date "+%Y-%m-%d 09:00:00")
us_searchtime_end=$(date "+%Y-%m-%d 15:59:59")

datenow=$(date "+%Y-%m-%d %H:%M:%S")

country="None"
if [[ "$ja_searchtime_start" < "$datenow" ]] && [[ "$ja_searchtime_end" > "$datenow" ]]; then
    country="ja"
elif [[ "$us_searchtime_start" < "$datenow" ]] && [[ "$us_searchtime_end" > "$datenow" ]]; then
    country="us"
fi

echo $country

if [ "$country" != "None" ]; then
    sudo docker run --rm --name minervini_screening --env IEX_TOKEN=$IEX_TOKEN --env SLACK_TOKEN=$SLACK_TOKEN minervini_screening:latest --country $country
    #sudo docker logs minervini_screening:latest
fi