#!/bin/bash
set -euo pipefail

# 一次性配置：从 npu1-23 免密 SSH 到三台远程 + 远程 sudo docker 免密。
# 用法：bash setup_ssh_sudo_nopasswd.sh
#
# 每台远程大概需要输两次密码：
#   1) ssh-copy-id 时提示的登录密码
#   2) 写 sudoers 时提示的 [sudo] password

HOSTS=(10.120.72.25 10.120.72.31 10.120.72.32)
KEY="$HOME/.ssh/id_rsa.pub"
SUDOERS_FILE=/etc/sudoers.d/chengyetai-docker

if [ ! -f "$KEY" ]; then
  echo "缺少公钥: $KEY"
  exit 1
fi

for H in "${HOSTS[@]}"; do
  echo
  echo "===== 1) SSH 免密 -> ${H}（如提示 password，请输入该机登录密码） ====="
  ssh-copy-id -i "$KEY" "$H"

  echo "===== 2) sudo docker 免密 -> ${H}（如提示 [sudo] password，请输入 sudo 密码） ====="
  ssh -t "$H" "
    D=\$(command -v docker || echo /usr/bin/docker)
    echo \"chengyetai ALL=(ALL) NOPASSWD: \${D}\" | sudo tee ${SUDOERS_FILE} >/dev/null
    sudo chmod 440 ${SUDOERS_FILE}
  "

  echo "===== 3) 验证 ${H} 的 sudo docker 免密 ====="
  ssh -o BatchMode=yes "$H" \
    'sudo -n docker version --format "sudo docker ok ({{.Client.Version}})"'
done

echo
echo "===== 验证结果 ====="
for H in "${HOSTS[@]}"; do
  printf '%-16s : ' "$H"
  ssh -o BatchMode=yes -o ConnectTimeout=6 "$H" \
    'printf "ssh=ok "; sudo -n docker version --format "sudo_docker=ok({{.Client.Version}})"' \
    2>&1 | tr '\n' ' '
  echo
done
