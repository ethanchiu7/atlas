# https://zhuanlan.zhihu.com/p/44405596
# 远端服务器设置
jupyter notebook --generate-config
# 设置密码
jupyter notebook password

# 修改 .jupyter/jupyter_notebook_config.py

c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.local_hostnames = ['ethanserver']
c.NotebookApp.password_required = True
c.NotebookApp.port = 8888