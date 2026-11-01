# RoboOps Basketball：人間のデモ3本から学ぶシュートロボット

Physical AI 応用編1 最終課題です。授業の「PushTで学ぶRoboOps入門」（データセットの管理・検証 → LeRobot ACTの学習 → シミュレータ評価 → GitHub Actions）の流れはそのまま使い、タスクを**自作の2Dバスケットボール**に置き換えました。さらに、データ収集から自分で行っています。

| | 授業（PushT） | この課題（Basketball） |
|---|---|---|
| データ | Hugging Faceの収集済みDataset | **人間がトラックパッドで操作したデモ3本** + 人間の癖を真似た合成デモ400本 |
| 観測 | agent位置、ブロックの8 keypoints | 手先位置、握っているか、経過時間、ボールの位置と速度、リング位置 |
| 行動 | 目標座標 (x, y) | 目標座標 (x, y) + **握る/離す** |
| 成功条件 | T字ブロックと目標の重なりが95%以上 | ボールが上からリングを通過する |

## パイプライン

```bash
uv sync --frozen && export PYTHONPATH=src

python src/teleop_basketball.py              # 1. 人間がデモを記録（Mac本体。スペースで握り、離すとシュート）
python src/generate_basketball.py            # 2. 人間の癖を測り、人間らしい合成デモを生成
python src/validate_basketball.py --min-human 3  # 3. Datasetの検査
python src/train.py --config configs/train_basketball.yaml       # 4. ACTを学習
python src/eval_basketball.py --config configs/eval_basketball.yaml  # 5. 固定した20 seedで評価
python src/replay_basketball.py              # 6. デモをreplayして動画にし、再現性を確認
```

GitHub Actionsでは、PushTに加えて手順2〜5（Dataset生成 → 検査 → 小規模な学習 → 評価）を自動で実行します。

## 工夫した点

- **人間らしいデータ拡張:** 人間のデモから、待ち時間、構え、投げる角度、加速時間、手ブレ、フォロースルーを測り（`measure_human_style`）、その分布からサンプルした合成デモを作ります。手ブレはOU過程で表現し、リリース速度にも誤差を入れます。成功した合成デモだけを学習に使います。
- **リリースのタイミングを学習させる:** temporal ensembleを使うと「離す」指令が平均化され、早く離しすぎて成功率が0%になりました。ensembleを外して毎step予測するように変えています。
- **経過時間を観測に加える:** 待ち時間がランダムだと、ポリシーが動き出せなくなります。そこで経過時間を観測に加えました。
- **管理:** 人間のデモ（`data/human/`）はseedとactionをGitで管理しています。replayで記録時の状態を誤差0.001以下で再現できます。Datasetのrevisionは内容のhashです。

## 結果（固定した20 seed）

| 試行 | 設定 | リリース率 | 成功率 |
|---|---|---|---|
| 1 | 授業と同じACT（temporal ensembleあり） | 100% | 0% |
| 2 | ensembleなし | 25% | 20% |
| 3 | ensembleなし + 経過時間を観測に追加（合成デモのみ） | 95% | 70% |
| 4 | 3と同じ設定 + 人間のデモ3本と、人間の癖に合わせた合成デモ | 100% | **50%** |

試行4の合成デモは、人間の低い弾道（約43°）を真似ているため、エキスパート自身の成功率も59%に下がっています。

---

# （ベース）PushTで学ぶRoboOps入門

ロボット学習における**管理・検証・自動化**を120分で一周するハンズオン教材です。PushTを題材に、Dataset取得、品質検証、学習、Simulator評価、GitHub Actionsの流れを体験します。

## 受講前の準備

アカウント作成、Dockerのインストール、事前動作確認は [`PREPARATION.md`](PREPARATION.md) にまとめています。講義前に実施してください。

## 教材の構成

```text
.
├── notebooks/01_roboops_workshop.ipynb  # 講義本編
├── src/                                  # CIで使う短い検証処理
├── tests/                                # codeとDatasetのtest
├── configs/                              # CI用の学習・評価条件
├── .github/workflows/ci.yml              # GitHub Actions
├── Dockerfile / compose.yaml             # 固定した実行環境
└── PREPARATION.md                        # 受講前の準備
```

## 起動する

教材を展開したfolderで、次を実行します。

```bash
docker compose up --build
```

起動後、次のURLを開きます。

[http://localhost:8888/lab?token=roboops-workshop](http://localhost:8888/lab?token=roboops-workshop)

JupyterLabで [`notebooks/01_roboops_workshop.ipynb`](notebooks/01_roboops_workshop.ipynb) を開きます。

## 終了する

起動したTerminalで `Ctrl+C` を押し、次を実行します。

```bash
docker compose down
```
