.PHONY: paper-sync paper-sync-once paper-sync-dry-run paper-status paper-forget-token

paper-sync:
	scripts/sync-overleaf-paper.sh --save-token --commit --push

paper-sync-once:
	scripts/sync-overleaf-paper.sh --commit --push

paper-sync-dry-run:
	scripts/sync-overleaf-paper.sh --dry-run --allow-dirty

paper-status:
	git status --short -- paper

paper-forget-token:
	scripts/sync-overleaf-paper.sh --forget-token
