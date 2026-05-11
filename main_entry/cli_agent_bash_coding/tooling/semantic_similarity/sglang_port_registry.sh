SGLANG_REGISTRY_BASENAME="sglang_port"

sglang_registry_file_for_port() {
  local log_dir="${1:?}"
  local port="${2:?}"
  printf '%s/%s_%s.registry' "$log_dir" "$SGLANG_REGISTRY_BASENAME" "$port"
}

sglang_read_model_tag_from_registry() {
  local reg_file="${1:?}"
  if [ ! -f "$reg_file" ]; then
    return 0
  fi
  local line
  line="$(grep -m1 '^model_tag=' "$reg_file" 2>/dev/null || true)"
  if [ -n "$line" ]; then
    printf '%s' "${line#model_tag=}"
  fi
}
