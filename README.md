# PushTで学ぶRoboOps入門

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
