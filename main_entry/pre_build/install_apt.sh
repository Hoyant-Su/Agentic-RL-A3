#!/usr/bin/env bash
# Online install for sandbox-style Debian/Ubuntu images (requires root, public network).
# Uses distro packages and go install only (no literal download URLs in this script).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export TZ="${TZ:-Etc/UTC}"

apt-get update -y

# Curated apt seed list for bash/sandbox-style workloads (XML/CSV/yaml/sqlite, editors, network utils, build + Python).
# yq / helm: install from distro (enable Ubuntu "universe" if apt reports not found).
PACKAGES=(
  libghc-yaml-dev xmlstarlet bc bsdextrautils libxml2-utils dos2unix sqlite3 moreutils csvtool
  jq ripgrep fd-find less vim nano tree tmux rsync zip unzip file patch shellcheck strace lsof
  procps psmisc iproute2 dnsutils netcat-openbsd iputils-ping apt-file sudo wget curl git
  build-essential pkg-config python3-pip python3-venv python3-dev ca-certificates openssh-client
  golang-go yq helm
)

apt-get install -y --no-install-recommends "${PACKAGES[@]}"

install -d /usr/local/bin

YQ_BIN="$(command -v yq)"

write_wrapper() {
  local out="$1"
  local input_format="$2"
  cat >"${out}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec ${YQ_BIN} -p=${input_format} "\$@"
EOF
  chmod 0755 "${out}"
}

write_wrapper /usr/local/bin/xq xml
write_wrapper /usr/local/bin/tomlq toml

CSVTK_VERSION="${CSVTK_VERSION:-0.31.0}"
GOBIN=/usr/local/bin go install "github.com/shenwei356/csvtk/cmd/csvtk@v${CSVTK_VERSION}"

missing=()
for cmd in json2yaml yaml2json xmlstarlet bc helm column xmllint dos2unix sqlite3 sponge csvtool xq tomlq csvtk yq; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    missing+=("${cmd}")
  fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: required commands are still missing after online install: ${missing[*]}" >&2
  exit 3
fi

echo "OK: online apt install finished"
