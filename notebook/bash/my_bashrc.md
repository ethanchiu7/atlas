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
alias ten104="sshpass -p xxx ssh -p xx xx@xx"
ssh -i "xx/xx" user@xx.xxx.xxx.xxx
