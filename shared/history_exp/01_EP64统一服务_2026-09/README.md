# GLM-5.2 四机 EP64 decode 部署

这是 chengyetai 自己的四机 EP64 decode 启动/停止脚本。

容器名：

```text
glm52-ep64-decode-chengyetai
```

## 启动

```bash
cd /home/chengyetai/workspace/glm52_ep64_chengyetai
bash start_ep64_decode.sh
```

## 查看状态

```bash
ssh 10.120.72.23 'sudo docker ps --filter name=glm52-ep64-decode-chengyetai --format "{{.Names}} {{.Status}}"'
```

## 健康检查

```bash
ssh 10.120.72.23 'curl -s -m 5 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:30000/health'
```

## 停止

```bash
cd /home/chengyetai/workspace/glm52_ep64_chengyetai
bash stop_ep64_decode.sh
```

## 文件说明

- `start_ep64_decode.sh`：把启动脚本传到四台机器并启动。
- `serve_glm52_ep64_decode.sh`：单台机器的实际启动脚本。
- `stop_ep64_decode.sh`：停止四台机器上的容器。
