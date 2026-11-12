"""Выкладка: пути службы — в одном несекретном файле, его подключают все юниты, deploy.sh его ставит.
Сеть: Streamlit слушает LAN (к нему ходит cloudflared из другого контейнера), заглушка LMS — только 127.0.0.1."""
from __future__ import annotations

import ipaddress
import re

from app import common
from tests.conftest import ROOT

DEPLOY = ROOT / "deploy"
PATH_VARS = ("CURATOR_INDEX_DIR", "CURATOR_JOURNAL", "CURATOR_EXPORTS_DIR", "LMS_URL")


def _env(path: str) -> dict[str, str]:
    lines = (DEPLOY / path).read_text(encoding="utf-8").splitlines()
    pairs = [line.split("=", 1) for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return {k.strip(): v.strip() for k, v in pairs}


def _exec_start(unit: str) -> str:
    lines = (DEPLOY / unit).read_text(encoding="utf-8").splitlines()
    starts = [line for line in lines if line.startswith("ExecStart=")]
    assert len(starts) == 1, unit
    return starts[0]


def _description(unit: str) -> str:
    text = (DEPLOY / unit).read_text(encoding="utf-8")
    found = re.search(r"^Description=(.*)$", text, re.MULTILINE)
    assert found is not None, unit
    return found.group(1)


def test_paths_file_has_all_service_paths():
    env = _env("ai-curator.paths.env")
    assert set(PATH_VARS) <= set(env)
    assert env["CURATOR_JOURNAL"] == "/var/lib/ai-curator/journal.db"
    assert not any(k in env for k in ("OPENAI_API_KEY", "USER_HASH_SALT"))  # секреты — только в /etc/ai-curator.env


def test_units_load_paths_file_instead_of_inline_paths():
    for unit in ("ai-curator.service", "ai-curator-cleanup.service", "ai-curator-lms.service"):
        text = (DEPLOY / unit).read_text(encoding="utf-8")
        assert "EnvironmentFile=/etc/ai-curator.paths.env" in text, unit
        assert not any(f"Environment={var}=" in text for var in PATH_VARS), unit


def test_deploy_script_installs_paths_file():
    text = (DEPLOY / "deploy.sh").read_text(encoding="utf-8")
    install = text.index("install -m 644 $APP/deploy/ai-curator.paths.env /etc/ai-curator.paths.env")
    assert install < text.index("systemctl restart ai-curator-lms ai-curator")


def test_deploy_package_has_every_local_package_tests_import():
    """Тесты гоняются на сервере на копии из архива: каждый пакет проекта, который они импортируют, должен в архив
    попасть, иначе выкладка остановится на сборе тестов (так было с eval)."""
    text = (DEPLOY / "deploy.sh").read_text(encoding="utf-8")
    shipped = " ".join(line for line in text.splitlines()
                       if re.match(r"\s*(?:cp|mkdir)\b", line) and "$TMP/pkg" in line)
    local = {p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".") and any(p.glob("*.py"))}
    imported: set[str] = set()
    for path in (ROOT / "tests").glob("*.py"):
        for found in re.finditer(r"^\s*(?:from|import)\s+(\w+)", path.read_text(encoding="utf-8"), re.MULTILINE):
            if found.group(1) in local:
                imported.add(found.group(1))
    assert {"curator", "app", "eval"} <= imported
    missing = [name for name in sorted(imported) if not re.search(rf"(?<![\w/$]){name}\b", shipped)]
    assert not missing, f"deploy.sh не кладёт в архив пакеты, которые импортируют тесты: {missing}"


def test_deploy_package_has_directories_app_pages_read():
    """Страницы app/ читают файлы с диска (`ROOT / "docs" / "diagram.svg"` на «О проекте»): каждый такой каталог
    должен ехать в архиве. Без docs схема на демо молча пропадала, а тест ссылок на сервере пропускался."""
    text = (DEPLOY / "deploy.sh").read_text(encoding="utf-8")
    shipped = " ".join(line for line in text.splitlines()
                       if re.match(r"\s*(?:cp|mkdir)\b", line) and "$TMP/pkg" in line)
    read_dirs: set[str] = set()
    for path in [*(ROOT / "app").glob("*.py"), *(ROOT / "app" / "views").glob("*.py")]:
        read_dirs |= set(re.findall(r'ROOT\s*/\s*"([\w.-]+)"', path.read_text(encoding="utf-8")))
    assert "docs" in read_dirs
    missing = [d for d in sorted(read_dirs) if not re.search(rf"(?<![\w/$]){re.escape(d)}\b", shipped)]
    assert not missing, f"deploy.sh не кладёт в архив каталоги, которые читают страницы: {missing}"


def test_health_check_inside_container_does_not_need_curl():
    """Внутри контейнера (heredoc pct exec) проверка готовности идёт через python3: curl в образе LXC может не быть,
    и тогда выкладка падала с «не поднялся» при живых службах. curl нужен только на хосте Proxmox."""
    text = (DEPLOY / "deploy.sh").read_text(encoding="utf-8")
    inside = text[text.index('pct exec $CT -- bash -s" <<EOF'):text.index("\nEOF\n")]
    assert "curl" not in "\n".join(line for line in inside.splitlines() if not line.lstrip().startswith("#"))
    assert "urllib.request" in inside and "/_stcore/health" in inside and ":8502/health" in inside


def test_streamlit_listens_on_lan_lms_only_on_loopback():
    """cloudflared (другой контейнер) ходит к Streamlit по LAN — на 127.0.0.1 туннель не достучится.
    Заглушку LMS наружу не открываем: куратор ходит к ней изнутри контейнера."""
    streamlit = _exec_start("ai-curator.service")
    assert "--server.address 0.0.0.0 --server.port 8501" in streamlit
    assert "127.0.0.1" not in streamlit
    assert "0.0.0.0:8501" in _description("ai-curator.service")
    lms = _exec_start("ai-curator-lms.service")
    assert "--host 127.0.0.1 --port 8502" in lms and "0.0.0.0" not in lms
    assert "127.0.0.1:8502" in _description("ai-curator-lms.service")
    assert _env("ai-curator.paths.env")["LMS_URL"] == "http://127.0.0.1:8502"


def test_trusted_proxies_in_paths_file():
    """Адрес cloudflared — в общем файле, который юнит Streamlit подключает через EnvironmentFile. Только адреса
    (каждый принимает app.common) и не loopback: cloudflared подключается по LAN, с 127.0.0.1 его не бывает."""
    raw = _env("ai-curator.paths.env")["CURATOR_TRUSTED_PROXIES"]
    entries = [part.strip() for part in raw.split(",") if part.strip()]
    assert entries
    assert len(common.parse_trusted_proxies(raw)) == len(entries)
    assert not any(ipaddress.ip_address(e).is_loopback or ipaddress.ip_address(e).is_unspecified for e in entries)
    unit = (DEPLOY / "ai-curator.service").read_text(encoding="utf-8")
    assert "EnvironmentFile=/etc/ai-curator.paths.env" in unit
    for name in ("ai-curator.service", "ai-curator-lms.service", "ai-curator-cleanup.service"):
        assert "CURATOR_TRUSTED_PROXIES=" not in (DEPLOY / name).read_text(encoding="utf-8"), name
    # uvicorn внутри Streamlit берёт адрес соединения из X-Forwarded-For только от FORWARDED_ALLOW_IPS (по умолчанию
    # 127.0.0.1, ::1). Расширить её — и клиент из LAN подделает сам адрес соединения, проверка прокси станет пустой.
    for name in ("ai-curator.service", "ai-curator.paths.env"):
        assert "FORWARDED_ALLOW_IPS" not in (DEPLOY / name).read_text(encoding="utf-8"), name


def test_deploy_checks_ports_from_proxmox_host():
    """После перезапуска deploy.sh проверяет с хоста Proxmox по LAN-адресу контейнера (адрес — из hostname -I,
    не зашит): 8501 отвечает, 8502 — нет. Для 8502 любой HTTP-ответ — ошибка, поэтому curl без -f."""
    text = (DEPLOY / "deploy.sh").read_text(encoding="utf-8")
    inside = text.index("http://127.0.0.1:8501/_stcore/health")
    heredoc_end = text.index("\nEOF\n", inside)
    host_part = text[heredoc_end:]
    assert 'ssh "$HOST" "pct exec $CT -- hostname -I"' in host_part
    lines = host_part.splitlines()
    ok = next(line for line in lines if "http://$CT_IP:8501/_stcore/health" in line)
    assert _runs_on_host(ok) and "curl -fsS" in ok
    closed = next(line for line in lines if "http://$CT_IP:8502/" in line)
    assert _runs_on_host(closed) and "curl -sS " in closed and " -f" not in closed.split("curl", 1)[1]
    assert re.search(r"^\s*0\) .*exit 1", host_part, re.MULTILINE)  # ответил — ошибка
    assert re.search(r"^\s*\*\) .*exit 1", host_part, re.MULTILINE)  # не удалось проверить — тоже ошибка
    assert not re.search(r"192\.168\.\d+\.\d+", text)


def _runs_on_host(line: str) -> bool:
    return line.lstrip().startswith('ssh "$HOST" "curl ')
