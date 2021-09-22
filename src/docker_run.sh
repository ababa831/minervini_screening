# 要求：Secret Managerに値を設定
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
filename_chart="chart.pkl"
bucket="stock-dwh-lake"
directory="stock-batch"
destination="gs://$bucket/$directory/$country/"
if [ "$country" != "None" ]; then
    sudo docker run --rm --name minervini_screening --env SLACK_TOKEN=$SLACK_TOKEN -v $PWD:/home minervini_screening:latest --country $country --filename_chart $filename_chart 
    # sudo docker logs minervini_screening:latest
    gsutil cp $(cd $(dirname ${BASH_SOURCE:-$0}); pwd)/$filename_chart  $destination
fi
