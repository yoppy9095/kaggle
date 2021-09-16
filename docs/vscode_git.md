# Visual Studio CodeでのGithub基本操作まとめ
gitはあらかじめ設定できているものとする
gitの初期設定は[ここを参照](https://prog-8.com/docs/git-env-win)
vscodeでのgitの基本操作まとめのQiitaは[こちら](https://qiita.com/y-tsutsu/items/2ba96b16b220fb5913be)
## 1.リポジトリのクローン
この操作は環境構築を行う際に、最初に連携用のディレクトリを作成するために用いる

vscodeでのクローンの方法としてはソース管理よりリポジトリの複製を選択し、ローカルリポジトリを作成
（リポジトリの初期化となっている場合はまずそれをするとリポジトリの複製が選べる）

![vscode_git2](https://user-images.githubusercontent.com/68320871/133446264-1c91454b-01db-4a22-a540-32d89ab2e4da.png)

![vscode_git3](https://user-images.githubusercontent.com/68320871/133446823-552a978a-ebf7-412c-86bc-17572c918438.png)


## リポジトリのpull 
この作業はほかの人が共有した内容を自分のローカルリポジトリに反映させるために行う（セーブデータをサーバからロードするイメージ）
ファイルの編集を行う前に最初に行うのがベスト
ソース管理より「・・・」を押下し、プルを選択
この操作により、最新のリポジトリの内容が自分のローカルリポジトリに反映される。
![vscode_git4](https://user-images.githubusercontent.com/68320871/133447681-f5cfce9d-99eb-4e10-8a8f-e78f82ccc2e4.png)

## 編集済みファイルのstaging、commitおよびpush
この作業は自分が編集したファイルや、追加したファイルなどをリモートリポジトリに反映させるためのものである
編集及び追加ファイルに関しては、ソース管理より変更部分に表示される
ここで、どのファイルをpushするのかを「＋」マークより選択する
![vsscode_git6](https://user-images.githubusercontent.com/68320871/133448784-95c978ec-5f7e-4efa-9be4-92bf0a13d912.png)

選択したファイルはステージされている変更部分に表示が移動する
pushしたいファイルをすべて選択したことを確認し、pushしたことをほかの人に伝えるためのメッセージを記入する
![vscode_git7](https://user-images.githubusercontent.com/68320871/133450315-09a04bc6-ed09-4f14-88f9-fbbfd43e2547.png)

メッセージを記入したのち✓マークを押下することで変更の履歴がリモートリポジトリに保存される（これをcommitと言い、ゲームデータをセーブするイメージ）
![vscode_git9](https://user-images.githubusercontent.com/68320871/133451408-459d9248-ada5-4d59-804a-5d037555f82d.png)

記入されたメッセージは矢印の部分に反映される
![vscode_git8](https://user-images.githubusercontent.com/68320871/133450398-431d4541-df08-451b-af8a-b8fb2bd82b05.png)

最後にステージングしたファイルをpushすることにより、リモートリポジトリに反映される（セーブデータをサーバにアップロードするイメージ）
![vscode_git5](https://user-images.githubusercontent.com/68320871/133451787-20c05552-9100-4e99-88e7-55894add6311.png)