# 從零開始：把本地專案初始化成 Git repo 並推上 GitHub

這篇會帶你走一遍完整流程：本地資料夾裡已經有程式碼，但本地還沒有 Git repo，GitHub 上也還沒有 repo，最後要做到本地和 GitHub 連在一起，之後可以直接 `git push`。

文中的範例是一個叫 `circle_detection` 的專案，裡面只有幾個 Jupyter notebook。

---

## 開始前的狀態

```
circle_detection/
├── 01_estimate_radius.ipynb
├── 02_detect_circles.ipynb
├── 03_refine_circles.ipynb
└── 04_neighbor_distances.ipynb
```

- 本地：只是普通資料夾，還沒 `git init`
- GitHub：還沒有這個 repo

---

## 步驟 1：確認環境

```powershell
git --version
git config user.name
git config user.email
```

如果 `user.name` 或 `user.email` 沒有輸出任何東西，要先設定。commit 的作者資訊就是從這裡來的：

```powershell
git config --global user.name "你的名字"
git config --global user.email "你的 email"
```

> 補充：如果之後想用命令列建 GitHub repo，可以順便看看有沒有安裝 GitHub CLI（`gh --version`）。沒有也沒關係，下面會用網頁建立 repo。

---

## 步驟 2：本地初始化 Git repo

在專案資料夾裡執行：

```powershell
git init -b main
```

`-b main` 會把預設分支命名為 `main`，跟 GitHub 目前的預設分支名稱一致，可以避免之後 `master` / `main` 對不上的問題。

---

## 步驟 3：加上 `.gitignore`

有些檔案不應該進版本控制，例如快取、虛擬環境、機密設定。在專案根目錄建立 `.gitignore`：

```gitignore
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.venv/
venv/
.env
.vscode/
.DS_Store
Thumbs.db
```

用 Jupyter notebook 的專案特別要注意 `.ipynb_checkpoints/`，這個資料夾是 Jupyter 自動產生的，沒有必要進版本控制。

---

## 步驟 4：在 GitHub 建立空的 repo

1. 打開 <https://github.com/new>
2. Repository name 填 `circle_detection`
3. 選擇 Public 或 Private
4. **不要勾選** README、.gitignore、license

第 4 點很重要。本地已經有內容了，如果 GitHub 上也先產生了 README，兩邊的歷史就不一樣，第一次 push 會被拒絕，還要另外處理合併。保持遠端是空的，就能直接推上去。

建好之後會拿到 repo 網址，例如：

```
https://github.com/houzeyu2683/circle_detection
```

---

## 步驟 5：第一次 commit、連接遠端、推送

```powershell
git add -A
git commit -m "Initial commit"
git remote add origin https://github.com/houzeyu2683/circle_detection.git
git push -u origin main
```

每一行在做什麼：

| 指令 | 作用 |
|---|---|
| `git add -A` | 把所有檔案（會排除 `.gitignore` 裡列出的）加入暫存區 |
| `git commit -m "..."` | 建立第一個 commit |
| `git remote add origin <網址>` | 把 GitHub repo 註冊成名叫 `origin` 的遠端 |
| `git push -u origin main` | 推送 `main` 分支，`-u` 會讓本地 `main` 追蹤 `origin/main`，之後只要打 `git push` 就好 |

成功的話會看到類似這樣的輸出：

```
To https://github.com/houzeyu2683/circle_detection.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.
```

### 關於 `LF will be replaced by CRLF` 警告

在 Windows 上 `git add` 時可能會看到：

```
warning: in the working copy of 'xxx', LF will be replaced by CRLF the next time Git touches it
```

這是 Git 在處理 Windows（CRLF）和 Unix（LF）換行符號的轉換，屬於正常提示，不影響使用，可以忽略。

---

## 之後的日常流程

```powershell
git add -A
git commit -m "說明這次改了什麼"
git push
```

---

## 延伸：內網 Git 伺服器也是一樣的做法

Git 是分散式的，GitHub 只是其中一種遠端主機。對 Git 來說，remote 就是一個網址或路徑。所以換成公司內網的 Git 伺服器時，**本地的步驟完全相同**，不同的只有遠端 repo 怎麼建，以及網址長什麼樣子：

| 內網常見做法 | 遠端 repo 怎麼建 | remote 網址範例 |
|---|---|---|
| 自架 GitLab / Gitea / Bitbucket Server / Azure DevOps Server | 跟 GitHub 一樣在網頁上建空的 repo | `https://gitlab.company.local/team/circle_detection.git` |
| 一台 Linux 主機（走 SSH） | 登入主機執行 `git init --bare circle_detection.git` | `ssh://user@10.0.0.5/srv/git/circle_detection.git` |
| 共用資料夾（網路磁碟） | 在共用資料夾執行 `git init --bare` | `//fileserver/share/circle_detection.git` |

內網環境常遇到的問題：

- **驗證方式**：可能要用 SSH key，或用帳號加 Personal Access Token，不一定能用密碼登入。
- **自簽憑證**：HTTPS 出現 `SSL certificate problem` 時，正確的做法是 `git config http.sslCAInfo <公司 CA 憑證路徑>`，不要直接關掉 SSL 驗證。
- **Proxy**：公司有 proxy 的話，可能要設定 `http.proxy`，或把內網網址排除在 proxy 之外。
- **遠端 repo 保持空的**：跟 GitHub 一樣，建 repo 時不要先產生 README。

---

## 快速總結

```powershell
# 1. 本地初始化
git init -b main

# 2. 建立 .gitignore（內容見上文）

# 3. 在 GitHub（或內網伺服器）建立「空的」repo

# 4. commit、連接、推送
git add -A
git commit -m "Initial commit"
git remote add origin <repo 網址>
git push -u origin main
```
