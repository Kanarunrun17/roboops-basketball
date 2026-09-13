# RoboOps入門 事前準備

この演習ではDockerとGit等を使用します。講義当日のdownload待ちや環境差による問題を避けるため、以下を事前に完了してください。

## 1. アカウントとアプリを準備する

### Docker

お使いのOSに合わせてDockerをインストールし、起動してください。

- **macOS:** [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/)をインストールします。Apple SiliconまたはIntelの該当するinstallerを選びます。
- **Windows:** [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)をインストールし、WSL 2 backendを有効にします。
- **Ubuntu:** [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)とDocker Compose pluginをインストールします。Docker Desktopは不要です。

Ubuntuでは、以降のcommandを `sudo` なしで実行するため、[Linuxのインストール後の手順](https://docs.docker.com/engine/install/linux-postinstall/)も行います。

### Git系

#### GitHubアカウント

GitHubアカウントがない場合は、[アカウントを作成](https://github.com/signup)します。

#### Git

お使いのOSに合わせてGitをインストールします。

- **macOS:** [Homebrew](https://brew.sh/ja/)をインストールし、Terminalで `brew install git` を実行します。
- **Windows:** [Git for Windows](https://git-scm.com/install/windows)からinstallerをdownloadして実行します。
- **Ubuntu:** Terminalで `sudo apt update`、続けて `sudo apt install git` を実行します。

#### GitHub CLI

お使いのOSに合わせてGitHub CLIをインストールします。

- **macOS:** Terminalで `brew install gh` を実行します。
- **Windows:** PowerShellで `winget install --id GitHub.cli --source winget` を実行します。
- **Ubuntu:** [Ubuntu向け手順](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)に従ってpackage repositoryを登録し、GitHub CLIをインストールします。

インストール後、次を実行してブラウザからGitHubへログインします。

```bash
gh auth login
```

認証先や接続方法など、画面に表示される選択肢は[GitHub CLIのクイックスタート](https://docs.github.com/ja/github-cli/github-cli/quickstart)を確認してください。

## 2. インストールを確認する

### Docker

Dockerのインストールと動作を確認します。Terminal（WindowsではPowerShell）を開き、次を1行ずつ実行します。

```bash
docker --version
docker compose version
docker run --rm hello-world
```

次のように表示されれば、Dockerを実行できます。version番号やbuild IDは環境によって異なります。

| command | 表示例 |
|---|---|
| `docker --version` | `Docker version xx.x.x, build xxxxxxx` |
| `docker compose version` | `Docker Compose version v2.x.x` |
| `docker run --rm hello-world` | `Hello from Docker!` |

### Git系

GitとGitHub CLIのインストール、GitHubへのログインを確認します。

```bash
git --version
gh --version
gh auth status
```

version番号は環境によって異なります。

| command | 表示例 |
|---|---|
| `git --version` | `git version 2.x.x` |
| `gh --version` | `gh version 2.x.x` |
| `gh auth status` | `Logged in to github.com account YOUR_NAME` |

## 3. 教材を事前に起動する

講師から案内された教材ファイルをdownloadし、ZIPファイルを展開します。展開した教材folderでTerminal（WindowsではPowerShell）を開き、次を実行します。

```bash
docker compose up --build
```

初回は必要なpackageをdownloadするため、数分から数十分かかることがあります。起動後、ブラウザで次を開きます。

[http://localhost:8888/lab?token=roboops-workshop](http://localhost:8888/lab?token=roboops-workshop)

JupyterLabの左側に `notebooks/01_roboops_workshop.ipynb` が表示されれば、事前確認は完了です。

---

port 8888を使用できない場合は、別のportを指定して起動します。

Ubuntu・macOSの場合：

```bash
JUPYTER_PORT=9999 docker compose up --build
```

Windows PowerShellの場合：

```powershell
$env:JUPYTER_PORT=9999
docker compose up --build
```

この場合は `http://localhost:9999/lab?token=roboops-workshop` を開きます。

---

確認後は、起動したTerminalで `Ctrl+C` を押してContainerを停止します。Terminalへ入力できる状態に戻ったら、次を実行してContainerを削除します。

```bash
docker compose down
```

## 困ったとき

動かない、errorが出る、commandの意味が分からないなど困った場合は、[ChatGPT](https://chatgpt.com/)などの生成AIへ相談するのも一つの方法です。

質問するときは、次の情報をまとめて伝えると原因を調べやすくなります。

- 使用しているOS
- 実行したcommand
- 表示されたerror messageの全文
- どの手順で発生したか

```text
Windows 11でRoboOps入門の事前準備をしています。
docker compose up --buildを実行したところ、次のerrorが出ました。

（error messageの全文）

考えられる原因と、安全に確認できる手順を順番に教えてください。
```

生成AIの回答が常に正しいとは限りません。内容を確認してから実行し、password、access token、個人情報、公開したくないcodeやDatasetは入力しないでください。
