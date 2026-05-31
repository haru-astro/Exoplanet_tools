# Exoplanet_tools

## sample_transit_fit.ipynb
同一惑星の複数トランジットを同時フィッティングするファイル。k, ln_a, T14, u1, u2, 各tcをMCMC法により推定する。
- 観測波長はr, iバンドに対応
- 2バンド同時観測にも対応
- 設定セクションにて、ファイル名・バンド・tcの初期期・ユリウス日を追加するだけで同時フィッティングできる

解析の流れは以下の通り
1. トランジット曲線はpytransitを利用
1. トランジット曲線に0次, 1次, 2次関数をかけたものをscipy.optimize.minimizeにより最適化
1. 2のうち、最もBICが低いものをトランジット曲線として採用
1. k, ln_a, T14, 各u1, 各u2, 各tc, 各jitterをパラメータとしemceeを用いて推定する

このファイルで惑星のパラメータ（bやk等）も推定できる。

## grid_search_jnkepler.py
3rd_bodyが存在するかもしれない系に対して、3rd_bodyのパラメータ空間をgridで区切り、jnkeplerを用いて各grid内での最も優れたchi^2を計算する。
コード内のconfigで探索するgridの大きさを指定できる。

## target_selection.py
grid_search_jnkepler.pyで選ばれたパラメータセットに対して、実際にMCMCを実行する際の初期値として用いるパラメータセットを選ぶ。
具体的には、1つの周期binでは1つのパラメータセットを選び、各パラメータがgridの端にくっついていないもののみを選ぶ。

## run_hmc.py
target_selection.pyで選定したパラメータを初期値として、MCMCを実行する。各パラメータの上限値、下限値も設定できる。
実行時間例：TOI-560、dt=0.4、20000stepsで48時間弱
