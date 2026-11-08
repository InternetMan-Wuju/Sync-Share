"""Wait for this experiment's server, then validate before any performance suite."""
import subprocess,time,urllib.request
s=time.monotonic()
while time.monotonic()-s<1800:
    try:
        with urllib.request.urlopen('http://127.0.0.1:30080/health',timeout=5) as r:
            if r.status==200:break
    except Exception:pass
    time.sleep(10)
else:raise RuntimeError('Server did not become healthy in 30 minutes')
for argv in [
 ['--mode','off','--name','warmup_off_0'],
 ['--mode','shadow','--profile','--verify','--name','validate_shadow_0'],
 ['--mode','active_cold','--profile','--verify','--name','validate_cold_0']]:
    subprocess.run(['python3','/runexp/scripts/client.py',*argv],check=True)
