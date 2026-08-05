#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s TARGET_DB SOURCE_DB\n' "$(basename "$0")" >&2
    printf 'Example: %s mydb.sqlite /path/to/existing.sqlite\n' "$(basename "$0")" >&2
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

target_db=$1
source_db=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sql_script="$script_dir/import_cached_embeddings.sql"

if [[ ! -f "$target_db" ]]; then
    printf 'target DB does not exist: %s\n' "$target_db" >&2
    exit 1
fi

if [[ ! -f "$source_db" ]]; then
    printf 'source DB does not exist: %s\n' "$source_db" >&2
    exit 1
fi

if [[ ! -f "$sql_script" ]]; then
    printf 'SQL import script does not exist: %s\n' "$sql_script" >&2
    exit 1
fi

sqlite_literal() {
    local value=${1//\'/\'\'}
    printf "'%s'" "$value"
}

source_literal=$(sqlite_literal "$source_db")

sqlite3 "$target_db" <<SQL
.bail on
.parameter set @source $source_literal
.read $sql_script
SQL
