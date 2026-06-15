#!/usr/bin/env bash
set -euo pipefail

OVERLEAF_REMOTE_DEFAULT="https://git.overleaf.com/67ef53e839dffff6952cd38b"
DEST_DIR_REL="paper"
KEYCHAIN_SERVICE="overleaf-git-67ef53e839dffff6952cd38b"
KEYCHAIN_ACCOUNT="git"

commit_changes=0
push_changes=0
delete_extra=0
dry_run=0
allow_dirty=0
save_token=0
forget_token=0
remote_url="$OVERLEAF_REMOTE_DEFAULT"
commit_message="Sync Overleaf project into paper"

usage() {
  cat <<'EOF'
Usage:
  scripts/sync-overleaf-paper.sh [options]

Sync the Overleaf project into this repository's paper/ directory.

Options:
  --commit              Stage paper/ changes and create a git commit.
  --push                Push the current branch after committing. Implies --commit.
  --delete              Mirror Overleaf exactly by deleting files in paper/ that
                        are not in Overleaf. Omit this to keep GitHub-only files.
  --dry-run             Show what would be copied without changing files.
  --allow-dirty         Allow running when paper/ already has local changes.
  --save-token          Save the token to macOS Keychain after prompting.
  --forget-token        Delete the saved token from macOS Keychain and exit.
  --remote URL          Override the Overleaf Git remote URL.
  --message TEXT        Commit message for --commit.
  -h, --help            Show this help.

Token handling:
  The script first checks OVERLEAF_GIT_TOKEN, then macOS Keychain, then prompts
  without echoing. Use --save-token once to avoid future prompts.

Common command:
  scripts/sync-overleaf-paper.sh --save-token --commit --push
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)
      commit_changes=1
      shift
      ;;
    --push)
      push_changes=1
      commit_changes=1
      shift
      ;;
    --delete)
      delete_extra=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --allow-dirty)
      allow_dirty=1
      shift
      ;;
    --save-token)
      save_token=1
      shift
      ;;
    --forget-token)
      forget_token=1
      shift
      ;;
    --remote)
      remote_url="${2:-}"
      if [[ -z "$remote_url" ]]; then
        echo "error: --remote requires a URL" >&2
        exit 2
      fi
      shift 2
      ;;
    --message)
      commit_message="${2:-}"
      if [[ -z "$commit_message" ]]; then
        echo "error: --message requires text" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

require_cmd git
require_cmd rsync
require_cmd mktemp

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/.." rev-parse --show-toplevel)"
dest_dir="$repo_root/$DEST_DIR_REL"

if [[ "$forget_token" -eq 1 ]]; then
  if command -v security >/dev/null 2>&1; then
    if security delete-generic-password -s "$KEYCHAIN_SERVICE" -a "$KEYCHAIN_ACCOUNT" >/dev/null 2>&1; then
      echo "Deleted saved Overleaf Git token from macOS Keychain."
    else
      echo "No saved Overleaf Git token found in macOS Keychain."
    fi
  else
    echo "error: macOS security command not found; cannot use Keychain" >&2
    exit 1
  fi
  exit 0
fi

if [[ ! -d "$dest_dir" ]]; then
  echo "error: destination directory not found: $dest_dir" >&2
  exit 1
fi

if [[ "$allow_dirty" -eq 0 && -n "$(git -C "$repo_root" status --porcelain -- "$DEST_DIR_REL")" ]]; then
  echo "error: $DEST_DIR_REL/ has local changes. Commit/stash them first, or pass --allow-dirty." >&2
  git -C "$repo_root" status --short -- "$DEST_DIR_REL" >&2
  exit 1
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/overleaf-paper-sync.XXXXXX")"
token_file="$tmp_dir/token"
askpass="$tmp_dir/git-askpass.sh"
overleaf_dir="$tmp_dir/overleaf"

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

if [[ -n "${OVERLEAF_GIT_TOKEN:-}" ]]; then
  token="$OVERLEAF_GIT_TOKEN"
elif command -v security >/dev/null 2>&1 && token="$(security find-generic-password -w -s "$KEYCHAIN_SERVICE" -a "$KEYCHAIN_ACCOUNT" 2>/dev/null)"; then
  echo "Using Overleaf Git token from macOS Keychain."
else
  printf "Overleaf Git authentication token: " >&2
  old_stty="$(stty -g)"
  stty -echo
  IFS= read -r token
  stty "$old_stty"
  printf "\n" >&2
fi

if [[ -z "$token" ]]; then
  echo "error: empty Overleaf token" >&2
  exit 1
fi

if [[ "$save_token" -eq 1 ]]; then
  if command -v security >/dev/null 2>&1; then
    security add-generic-password -U -s "$KEYCHAIN_SERVICE" -a "$KEYCHAIN_ACCOUNT" -w "$token"
    echo "Saved Overleaf Git token to macOS Keychain."
  else
    echo "warning: macOS security command not found; token was not saved" >&2
  fi
fi

printf "%s" "$token" > "$token_file"
chmod 600 "$token_file"
token=""

cat > "$askpass" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf "git" ;;
  *Password*) cat "$OVERLEAF_SYNC_TOKEN_FILE" ;;
  *) printf "git" ;;
esac
EOF
chmod 700 "$askpass"

echo "Cloning Overleaf project..."
GIT_TERMINAL_PROMPT=0 \
GIT_ASKPASS="$askpass" \
OVERLEAF_SYNC_TOKEN_FILE="$token_file" \
git clone --quiet "$remote_url" "$overleaf_dir"

overleaf_head="$(git -C "$overleaf_dir" rev-parse --short HEAD)"
overleaf_file_count="$(git -C "$overleaf_dir" ls-files | wc -l | tr -d ' ')"
echo "Overleaf HEAD: $overleaf_head"
echo "Overleaf tracked files: $overleaf_file_count"

rsync_args=(-a --exclude='.git/')
if [[ "$delete_extra" -eq 1 ]]; then
  rsync_args+=(--delete)
fi
if [[ "$dry_run" -eq 1 ]]; then
  rsync_args+=(--dry-run --itemize-changes)
fi

echo "Syncing into $DEST_DIR_REL/..."
rsync "${rsync_args[@]}" "$overleaf_dir/" "$dest_dir/"

if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry run complete. No files were changed."
  exit 0
fi

mismatches=0
while IFS= read -r -d '' file; do
  if ! cmp -s "$overleaf_dir/$file" "$dest_dir/$file"; then
    echo "mismatch: $DEST_DIR_REL/$file" >&2
    mismatches=$((mismatches + 1))
  fi
done < <(git -C "$overleaf_dir" ls-files -z)

if [[ "$mismatches" -ne 0 ]]; then
  echo "error: verification failed with $mismatches mismatched file(s)" >&2
  exit 1
fi
echo "Verified $overleaf_file_count Overleaf file(s) in $DEST_DIR_REL/."

if [[ "$commit_changes" -eq 1 ]]; then
  if [[ "$push_changes" -eq 1 ]]; then
    branch="$(git -C "$repo_root" rev-parse --abbrev-ref HEAD)"
    if [[ "$branch" == "HEAD" ]]; then
      echo "error: cannot push from detached HEAD" >&2
      exit 1
    fi
    git -C "$repo_root" fetch origin "$branch"
    if ! git -C "$repo_root" merge-base --is-ancestor "origin/$branch" HEAD; then
      echo "error: local branch is behind origin/$branch; pull/rebase before pushing." >&2
      exit 1
    fi
  fi

  if [[ "$delete_extra" -eq 1 ]]; then
    git -C "$repo_root" add -A -- "$DEST_DIR_REL"
  fi

  while IFS= read -r -d '' file; do
    git -C "$repo_root" add -f -- "$DEST_DIR_REL/$file"
  done < <(git -C "$overleaf_dir" ls-files -z)

  if git -C "$repo_root" diff --cached --quiet -- "$DEST_DIR_REL"; then
    echo "No paper/ changes to commit."
  else
    git -C "$repo_root" commit -m "$commit_message"
  fi

  if [[ "$push_changes" -eq 1 ]]; then
    git -C "$repo_root" push origin "$branch"
  fi
else
  echo "Local sync complete. Review with:"
  echo "  git -C \"$repo_root\" status --short -- $DEST_DIR_REL"
fi
