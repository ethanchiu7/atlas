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
    
    # set tracking information for this branch
    git branch --set-upstream-to=origin/<branch>
    
# merge

# HTTP免密码

    # 1.配置保存密码【长期】
    git config --global credential.helper store
    # 2.配置保存密码【短期】
    # 默认15分钟
    git config --global credential.helper cache
    git config credential.helper 'cache --timeout=3600'
    
    # 3.增加远程地址的时候带上密码
    git clone http://yourname:password@git.oschina.net/name/project.git

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
    
# 忘记新建分支

    git stash
    git fetch
    git merge origin/master
    git checkout -b new_branch_name
    git stash pop
    

# 提交错了分支

    git reset HEAD^
    git stash
    git fetch
    git merge origin/master
    git checkout -b new_branch_name
    git stash pop