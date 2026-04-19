# Informational Health Simulator: Social Desirability Bias & The Illusion of Massive Data

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本リポジトリは、アンケートや大規模調査における「社会的望ましさバイアス（Social Desirability Bias / 忖度）」が、いかにしてマイノリティの警告シグナルを構造的かつ非線形に抹殺するかを数理的に証明するシミュレーション・コードです。

「大数の法則」に過剰適応した現代のデータサイエンスにおいて、データ生成プロセスそのものに潜む「情報の不健康状態（Informational Ill-health）」を可視化し、「情報の健康診断」という新たなパラダイムを提唱するための計算論的根拠を提供します。

## 📌 背景と問題意識 (Background)

ビッグデータ時代において、「サンプルサイズが大きければ誤差は消える」という素朴な信仰が蔓延しています。しかし、回答者の意思決定プロセスに「社会的圧力（忖度）」という重力が作用している場合、巨大なデータは真実を映し出すのではなく、むしろ「歪んだ合意」を数学的な真理へとロンダリングする装置として機能します。

本プロジェクトのシミュレーションは、以下の事実を定量的に証明します。

1. **シグナルの崖（Phase Transition）**: バイアスが一定の臨界点を超えた瞬間、マイノリティの「小さな声」はなだらかに減少するのではなく、非線形に「蒸発」する。
2. **統計的有意性の欺瞞（Epistemic Injustice）**: 平均値の線形な改善の裏で、情報の構造的な歪み（KLダイバージェンス）が非線形に爆発する。
3. **大数の法則のパラドックス（Illusion of Stability）**: サンプルサイズ（$N=10^9$）の極大化は、真実への到達を保証するのではなく、忖度に汚染された「偽の安定」への確信を強める。

## 🧮 数理モデル (Mathematical Model)

個人の最終的な効用 $U$ を、内発的な「本音の効用」と外発的な「忖度の効用」の線形結合として定義し、Softmax関数を通じて選択確率を算出します。

$$U_{total} = (1 - v_2)U_{true} + v_2U_{target}$$

* $v_2$: 社会的望ましさ（忖度）の重み。0で完全な本音、1で完全な社会的同調。
* $\beta$: 逆温度係数（確信度）。Softmax関数の鋭敏さを制御。

## 📊 出力される分析結果 (Outputs)

スクリプトを実行すると、`comprehensive_simulation_results` ディレクトリが作成され、以下の6つの高解像度グラフ（PNG）と対応する生データ（CSV）が生成されます。

* **Fig 1: Survival Curve** - バイアス増大に伴う警告シグナル（評価1）の急激な墜落（崖）。
* **Fig 2: Structural Alteration** - 5つのバイアス・シナリオにおける分布の全体主義的収束プロセス。
* **Fig 3: Mean vs KL Divergence** - 表面的な平均値の改善と、裏側で進行する情報の歪み（KL距離）の対比。
* **Fig 4: Power Cliff** - 異常検知の統計的検出力（Power）が臨界点で機能不全に陥る証明。
* **Fig 5: Convergence to False Stability** - $N=10^3$ から $10^9$ への拡大がもたらす「偽の安定」のバイオリンプロット。
* **Fig 6: The Violence of Absolute Numbers** - $N=10^9$ において、わずか5%の割合の中に不可視化される5,000万人規模の「SOS」の絶対数可視化。

## 🚀 実行方法 (Usage)

本コードは **Google Colaboratory** での実行に最適化されています。

1. `informational_health_sim.py`（またはJupyter Notebook形式）を Google Colab にアップロードします。
2. 全てのセルを実行します。
3. 実行完了後、グラフとCSVが格納された `comprehensive_simulation_archive.zip` がブラウザ経由で自動的にダウンロードされます。

### ローカル環境での実行に関する注意
ローカルのPython環境で実行する場合は、スクリプト先頭の `from google.colab import files` および、スクリプト末尾の `files.download()` の行をコメントアウトして実行してください。ZIPファイルはカレントディレクトリに生成されます。

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# スクリプトの実行
python informational_health_sim.py
