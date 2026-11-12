#!/usr/bin/env bash
# Выкладка ИИ-куратора в LXC-контейнер на pve (по образцу FAQ-ассистента PEcf11). Запускать из Git Bash:
#   DEPLOY_CT=173 bash deploy/deploy.sh                      # обновить код, базу и индекс, прогнать тесты, перезапустить
#   DEPLOY_CT=173 bash deploy/deploy.sh --secrets path/.env  # первый запуск: взять из файла строку OPENAI_API_KEY=
# Номер контейнера обязателен (DEPLOY_CT): выкладка в чужой контейнер хуже, чем остановка с ошибкой.
# Ключ идёт потоком прямо в контейнер и в выводе не печатается. Соль кода сессии (USER_HASH_SALT) создаётся
# на сервере один раз и при обновлениях не меняется. Индекс собирается локально (python -m curator.ingest) и едет
# в архиве: на сервере эмбеддинги базы не пересчитываются. Перед упаковкой индекс сверяется с data/kb.
# Порядок: тесты гоняются на отдельной копии, и только если они прошли, код заменяется и службы перезапускаются.
# Сеть: Streamlit слушает 0.0.0.0:8501 — cloudflared живёт в другом контейнере (LXC 101) и ходит к службе по LAN;
# заглушка LMS — только 127.0.0.1:8502. После перезапуска обе стороны проверяются с хоста Proxmox по LAN-адресу
# контейнера: 8501 должен отвечать, 8502 — нет.
set -euo pipefail

HOST="${DEPLOY_HOST:-pve-lan}"
CT="${DEPLOY_CT:?задайте номер контейнера: DEPLOY_CT=<номер> bash deploy/deploy.sh}"
APP=/opt/ai-curator
STATE=/var/lib/ai-curator
STAGE=/opt/ai-curator.stage
SECRETS=""
if [ "${1:-}" = "--secrets" ]; then
  SECRETS="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"   # до cd ниже: относительный путь не потеряется
  grep -q '^OPENAI_API_KEY=.' "$SECRETS" || { echo "в $SECRETS нет строки OPENAI_API_KEY="; exit 1; }
fi
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi

"$PY" -m curator.ingest --check || { echo 'индекс не совпадает с data/kb: сначала python -m curator.ingest'; exit 1; }
INDEX="$("$PY" -c 'from curator.config import load_settings; print(load_settings().index_dir)')"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/pkg/data" "$TMP/pkg/index"
# docs — целиком: страница «О проекте» показывает docs/diagram.svg, а тест ссылок проверяет все пути docs/… страницы.
cp -r curator app tests deploy docs .streamlit requirements.txt requirements-dev.txt pyproject.toml "$TMP/pkg/"
cp -r data/kb data/lms data/resources data/fewshot "$TMP/pkg/data/"
# Код оценки едет только ради тестов на сервере (tests/test_eval_*.py его импортируют); наборы и прогоны — нет.
mkdir -p "$TMP/pkg/eval" && cp eval/*.py "$TMP/pkg/eval/"
rm -rf "$TMP/pkg/data/lms/exports"
cp -r "$INDEX"/. "$TMP/pkg/index/"
TGZ="$TMP/ai-curator.tgz"
tar czf "$TGZ" --owner=0 --group=0 --exclude=__pycache__ -C "$TMP/pkg" .
LOCAL_SUM="$(sha256sum "$TGZ" | cut -d' ' -f1)"
scp -q "$TGZ" "$HOST:/tmp/ai-curator.tgz"
ssh "$HOST" "pct push $CT /tmp/ai-curator.tgz /tmp/ai-curator.tgz && rm -f /tmp/ai-curator.tgz"

if [ -n "$SECRETS" ]; then
  grep -E '^OPENAI_API_KEY=' "$SECRETS" | tr -d '\r' | \
    ssh "$HOST" "pct exec $CT -- sh -c 'umask 077; cat > /root/.curator-openai-key'"
fi

ssh "$HOST" "pct exec $CT -- bash -s" <<EOF
set -euo pipefail
UV=/usr/local/bin/uv   # pct exec запускает голый процесс: PATH без /usr/local/bin
# Ключ из --secrets не должен остаться в /root, чем бы ни кончился запуск.
trap 'rm -f /root/.curator-openai-key /tmp/ai-curator.tgz' EXIT

id curator >/dev/null 2>&1 || useradd --system --home-dir $STATE --shell /usr/sbin/nologin curator
if [ -s /root/.curator-openai-key ]; then
  umask 077
  SALT="\$(grep '^USER_HASH_SALT=' /etc/ai-curator.env 2>/dev/null || true)"
  [ -n "\$SALT" ] || SALT="USER_HASH_SALT=\$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
  { cat /root/.curator-openai-key; echo "\$SALT"; } > /etc/ai-curator.env.new
  mv /etc/ai-curator.env.new /etc/ai-curator.env
  umask 022
fi
test -f /etc/ai-curator.env || { echo 'нет /etc/ai-curator.env: запустите с --secrets'; exit 1; }
chown root:curator /etc/ai-curator.env && chmod 640 /etc/ai-curator.env

test "\$(sha256sum /tmp/ai-curator.tgz | cut -d' ' -f1)" = "$LOCAL_SUM" || { echo 'архив доехал битым'; exit 1; }
test -x \$UV || { echo 'нет uv в /usr/local/bin'; exit 1; }

# 1. Тесты на отдельной копии: рабочий код и его окружение пока не тронуты. Тестам нужны только фикстуры.
rm -rf $STAGE && install -d -m 755 $STAGE
tar xzf /tmp/ai-curator.tgz --no-same-owner -C $STAGE
\$UV venv --quiet --python /usr/bin/python3 $STAGE/.venv
\$UV pip install --quiet --python $STAGE/.venv/bin/python -r $STAGE/requirements.txt -r $STAGE/requirements-dev.txt
cd $STAGE
if ! PYTHONDONTWRITEBYTECODE=1 $STAGE/.venv/bin/python -m pytest -q -p no:cacheprovider tests > /tmp/curator-tests.log 2>&1; then
  tail -40 /tmp/curator-tests.log; cd /; rm -rf $STAGE; echo 'тесты не прошли: рабочий код не тронут'; exit 1
fi
tail -1 /tmp/curator-tests.log
cd /
rm -rf $STAGE

# 2. Тесты прошли: меняем код, данные и индекс, перезапускаем службы. Журнал в $STATE не трогается.
install -d -o root -g root -m 755 $APP
install -d -o curator -g curator -m 750 $STATE
find $APP -mindepth 1 -maxdepth 1 ! -name .venv -exec rm -rf {} +
tar xzf /tmp/ai-curator.tgz --no-same-owner -C $APP
rm -rf $STATE/index && mv $APP/index $STATE/index && chown -R curator:curator $STATE/index
chown -R root:root $APP
[ -x $APP/.venv/bin/python ] || \$UV venv --quiet --python /usr/bin/python3 $APP/.venv
\$UV pip install --quiet --python $APP/.venv/bin/python -r $APP/requirements.txt

# Пути службы — общий несекретный файл для всех юнитов и ручных команд (отчёт, journal stats).
install -m 644 $APP/deploy/ai-curator.paths.env /etc/ai-curator.paths.env
for unit in ai-curator.service ai-curator-lms.service ai-curator-cleanup.service ai-curator-cleanup.timer; do
  install -m 644 $APP/deploy/\$unit /etc/systemd/system/\$unit
done
systemctl daemon-reload
systemctl enable --quiet ai-curator-lms ai-curator
systemctl enable --quiet --now ai-curator-cleanup.timer
systemctl restart ai-curator-lms ai-curator
# Изнутри контейнера: обе службы поднялись (127.0.0.1 отвечает и у Streamlit на 0.0.0.0, и у заглушки LMS).
ok=0
for i in \$(seq 1 60); do
  # python3, а не curl: в образе контейнера curl может не быть (в LXC 173 его нет), python3 есть всегда.
  if python3 -c 'import sys, urllib.request as u; [u.urlopen(x, timeout=3).read() for x in sys.argv[1:]]' \
      http://127.0.0.1:8502/health http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
    echo "ai-curator: ok"; ok=1; break
  fi
  sleep 1
done
[ \$ok = 1 ] || { echo "ai-curator: не поднялся"; journalctl -u ai-curator -u ai-curator-lms -n 40 --no-pager; exit 1; }
EOF

# 3. Снаружи контейнера — с хоста Proxmox по LAN-адресу контейнера: этим путём к Streamlit ходит cloudflared
# (LXC 101). Адрес не зашит: первый IPv4 из hostname -I контейнера.
CT_IP="$(ssh "$HOST" "pct exec $CT -- hostname -I" | tr ' \r' '\n\n' | grep -m1 -E '^[0-9]+(\.[0-9]+){3}$' || true)"
[ -n "$CT_IP" ] || { echo "не узнать IPv4-адрес контейнера $CT (hostname -I)"; exit 1; }
ssh "$HOST" "curl -fsS --max-time 5 -o /dev/null http://$CT_IP:8501/_stcore/health" || {
  echo "ai-curator: с хоста $HOST не отвечает http://$CT_IP:8501 — туннель до службы не достучится"; exit 1; }
# Заглушка LMS наружу не открывается. Без -f: любой HTTP-ответ, даже 404, — значит, порт открыт.
# 7 — соединение отклонено, 28 — нет ответа за отведённое время: порт закрыт, как и должно быть.
rc=0
ssh "$HOST" "curl -sS --max-time 3 -o /dev/null http://$CT_IP:8502/health" 2>/dev/null || rc=$?
case $rc in
  7|28) echo "ai-curator: снаружи $CT_IP:8501 отвечает, заглушка LMS ($CT_IP:8502) закрыта" ;;
  0) echo "заглушка LMS отвечает по LAN-адресу $CT_IP:8502 — она должна слушать только 127.0.0.1"; exit 1 ;;
  *) echo "не удалось проверить $CT_IP:8502 с хоста $HOST (код $rc)"; exit 1 ;;
esac
