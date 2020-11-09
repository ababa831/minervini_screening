# minervini_screening
ミネルヴィニのテンプレートに基づいた自動銘柄選定プログラム

NOTE: 自分用のためドキュメントは最小限

## 準備

- ~~IEX CloudのAPI & IEXFinance（Pythonクライアントライブラリ）を使えるようにする~~ -> 不要にした
- GCPのSecret managerで Slackと IEX CloudのSecret tokenを設定する
- dockerをビルドしてdocker_runする
- あとは適当なサーバでtask schedulingするなり好きに使う