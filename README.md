# minervini_screening
NOTE: 自分用のためドキュメントは最小限

ミネルヴィニのテンプレートに基づいた自動銘柄選定プログラム
dockerをビルドしてdocker_run -> Slackに投資推奨銘柄スクリーニング結果(CSV)が投稿される．

## 準備

1. ~~IEX CloudのAPI & IEXFinance（Pythonクライアントライブラリ）を使えるようにする~~ -> 不要にした．
    - Tickerリストを取得する用途でしか利用していなかったので，NASDAQ公式から直接取れるようにした．
2. GCPのSecret managerで SlackのSecret tokenを設定
3. GCEのインスタンスを適当に用意して，スタートアップスクリプト設定欄に `/mnt/c/Users/tabie/workspace/minervini_screening/src/startup_script.txt` の内容を登録
4. Cloud SchedulerにGCEの起動・停止タスクを仕掛ける
