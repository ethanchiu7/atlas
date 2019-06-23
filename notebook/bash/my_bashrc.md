# Ethan Jo bashrc

# env
export PATH="/Users/zhaoxin/bin:$PATH"

# some more ls aliases
alias ll='ls -alF'
alias lh='ls -alFh'
alias la='ls -A'
alias l='ls -CF'

# find
alias findcpp='find  -depth -type f -iname "*.cpp" | xargs grep --color -nE -ir'
alias findh='find  -depth -type f -iname "*.h" | xargs grep --color -nE -ir'
alias findpy='find  -depth -type f -iname "*.py" | xargs grep --color -nE -ir'
alias findsh='find  -depth -type f -iname "*.sh" | xargs grep --color -nE -ir'

# ssh key tencent
alias ten104="sshpass -p xxx ssh -p xx server_user@xx"
alias ten104_key="ssh -i private_key_file_path server_user@xx.xxx.xxx.xxx"

# sshpass
PASSWORD="xx"
alias xx="sshpass -p ${PASSWORD} ssh -p 2001 private_user@role_user@1.1.1.1@9922@xx.xx.cn"

# --- TensorFlow Config ---
# if you're on Unix. Tensorflow is working fine anyway, but you won't see these annoying warnings
export TF_CPP_MIN_LOG_LEVEL=2

#  add tensorflow models to Python path
export PYTHONPATH="$PYTHONPATH:/Users/zhaoxin/PycharmProjects/models"

# ------ GNU COMMAND ------
alias date='gdate'
alias readlink="greadlink"
