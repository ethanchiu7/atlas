# branch

    # 查看本地分支
    git branch
    # 查看远端分支
    git branch -r
    # 创建并切换分支
    git checkout -b <branchName>

    # 推送本地分支到远端
    git push origin <branchName>

    # 删除远端分支
    git push origin --delete <branchName>
    
# merge

# cherry-pick
    
    # 获取其他分支某个提交
    git cherry-pick commitId
    
# submodule

    # 拉取子项目代码
    git submodule init
    git submodule update
    
    # 遇到 +Subproject commit
    git submodule update --init
    
    # 更新子项目
    cd iflow_proto/proto
    git checkout master
    git fetch
    git merge origin/master
    cd ..
    scons install