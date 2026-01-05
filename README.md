# shallow-water-enkf

## 概要
カスケード沈み込み帯でのシナリオ津波に対して,EnKFを用いて観測データを同化し,津波のリアルタイム予測を行う.

## フォルダの説明
### scripts
実験に使用するファイル.

## 手順
1. 以下のコマンドを入力する.
~~~
pip3 install -r requirements.txt
~~~

2. `scripts/make_data`上で以下のコマンドを順に入力する.
~~~
python3 make_H.py
~~~
~~~
python3 make_eta0.py
~~~
3. `scripts/model`上で以下のコマンドを入力する.
~~~
python3 python3 run_enkf.py
~~~
