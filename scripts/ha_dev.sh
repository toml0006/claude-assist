#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_DIR="${ROOT_DIR}/dev"
CONFIG_DIR="${DEV_DIR}/ha_config"

ha_compose() {
  docker compose -f "${DEV_DIR}/docker-compose.yml" "$@"
}

ha_exec() {
  docker exec -i ai-subscription-assist-ha "$@"
}

wait_for_ha() {
  # Wait until HA is responding to HTTP.
  local url="http://localhost:8123/"

  for _ in {1..120}; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "Home Assistant did not become ready at ${url}" >&2
  return 1
}

bootstrap_onboarding() {
  local user="$1"
  local pass="$2"

  local status_code
  status_code="$(curl -sS -o /tmp/ha_onboarding.json -w '%{http_code}' http://localhost:8123/api/onboarding || true)"

  # If onboarding endpoints are not present, HA is already onboarded.
  if [[ "${status_code}" == "404" ]]; then
    return 0
  fi

  if [[ "${status_code}" != "200" ]]; then
    echo "Unexpected /api/onboarding status code: ${status_code}" >&2
    cat /tmp/ha_onboarding.json >&2 || true
    return 1
  fi

  local user_done
  user_done="$(
    python3 - <<'PY'
import json
steps = json.load(open("/tmp/ha_onboarding.json"))
print("true" if any(s.get("step") == "user" and s.get("done") for s in steps) else "false")
PY
  )"

  if [[ "${user_done}" == "true" ]]; then
    # We can't programmatically finish the remaining onboarding steps without a token
    # unless we start from a clean instance.
    echo "Onboarding 'user' step is already done; run './scripts/ha_dev.sh reset' then bootstrap for a fresh instance." >&2
    return 1
  fi

  # 1) Create the initial admin/owner user via the onboarding API (no auth required).
  local auth_code
  auth_code="$(
    curl -fsS \
      -X POST \
      -H 'Content-Type: application/json' \
      -d "{\"name\":\"Dev\",\"username\":\"${user}\",\"password\":\"${pass}\",\"client_id\":\"http://localhost:8123/\",\"language\":\"en\"}" \
      http://localhost:8123/api/onboarding/users \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["auth_code"])'
  )"

  # 2) Exchange auth code for tokens so we can complete the remaining onboarding steps.
  local access_token
  access_token="$(
    curl -fsS \
      -X POST \
      -H 'Content-Type: application/x-www-form-urlencoded' \
      --data-urlencode 'grant_type=authorization_code' \
      --data-urlencode "code=${auth_code}" \
      --data-urlencode 'client_id=http://localhost:8123/' \
      http://localhost:8123/auth/token \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])'
  )"

  local authz="Authorization: Bearer ${access_token}"

  # 3) Complete the onboarding steps. These require auth.
  curl -fsS -X POST -H "${authz}" -H 'Content-Type: application/json' -d '{}' \
    http://localhost:8123/api/onboarding/core_config >/dev/null
  curl -fsS -X POST -H "${authz}" -H 'Content-Type: application/json' -d '{}' \
    http://localhost:8123/api/onboarding/analytics >/dev/null
  curl -fsS -X POST -H "${authz}" -H 'Content-Type: application/json' \
    -d '{"client_id":"http://localhost:8123/","redirect_uri":"http://localhost:8123/"}' \
    http://localhost:8123/api/onboarding/integration >/dev/null

  # Verify onboarding now reports all steps done.
  curl -fsS http://localhost:8123/api/onboarding >/dev/null
}

cmd="${1:-}"

case "${cmd}" in
  up)
    ha_compose up -d
    ;;
  down)
    ha_compose down
    ;;
  restart)
    ha_compose restart
    ;;
  logs)
    # Pass through any extra args (e.g. --since=5m) while keeping a useful default.
    ha_compose logs -f --tail=200 "${@:2}"
    ;;
  reset)
    # Stop containers and wipe runtime state so the next `up` is a clean instance.
    ha_compose down
    rm -rf \
      "${CONFIG_DIR}/.cloud" \
      "${CONFIG_DIR}/.storage" \
      "${CONFIG_DIR}/blueprints" \
      "${CONFIG_DIR}/deps" \
      "${CONFIG_DIR}/tts"
    rm -f \
      "${CONFIG_DIR}/.HA_VERSION" \
      "${CONFIG_DIR}/.ha_run.lock" \
      "${CONFIG_DIR}/known_devices.yaml" \
      "${CONFIG_DIR}/home-assistant.log" \
      "${CONFIG_DIR}/home-assistant.log.1" \
      "${CONFIG_DIR}/home-assistant.log.fault" \
      "${CONFIG_DIR}/home-assistant_v2.db" \
      "${CONFIG_DIR}/home-assistant_v2.db-shm" \
      "${CONFIG_DIR}/home-assistant_v2.db-wal"
    ;;
  ensure-user)
    # Ensure a known username/password exists for local dev.
    # Usage: ./scripts/ha_dev.sh ensure-user [username] [password]
    user="${2:-dev}"
    pass="${3:-devpassword}"

    existing="$(
      ha_exec hass --script auth -c /config list \
        | awk 'NF==1 {print $1}' \
        | tr -d '\r'
    )"

    if printf '%s\n' "${existing}" | grep -qx "${user}"; then
      ha_exec hass --script auth -c /config change_password "${user}" "${pass}"
    else
      ha_exec hass --script auth -c /config add "${user}" "${pass}"
    fi
    ;;
  bootstrap)
    # Bring HA into a ready-to-use development state:
    # - start container
    # - complete onboarding via the onboarding HTTP API
    # Usage: ./scripts/ha_dev.sh bootstrap [username] [password]
    user="${2:-dev}"
    pass="${3:-devpassword}"

    ha_compose up -d
    wait_for_ha

    bootstrap_onboarding "${user}" "${pass}"
    ;;
  *)
    echo "Usage: $0 {up|down|restart|logs|reset|ensure-user|bootstrap}" >&2
    exit 2
    ;;
esac
