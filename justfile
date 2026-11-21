# acetate — self-hosted Docker Compose helper recipes.
# Run from your workstation; recipes SSH into your server using .env.deploy
# (gitignored — stays off GitHub). Example .env.deploy:
#
#   ACETATE_DEPLOY_HOST=myserver        # SSH alias from ~/.ssh/config
#   ACETATE_DEPLOY_DIR=~/acetate        # path to the cloned repo on the server
#   ACETATE_COMPOSE_DIR=~/acetate       # set if docker-compose.yml lives elsewhere
#   ACETATE_SERVICES=                   # blank = all services in the compose file
#
# Defaults assume this repo's docker-compose.yml run from ~/acetate.
# Override per-run:  just --set host my-other-box logs

set dotenv-load := true
set dotenv-filename := ".env.deploy"

host        := env_var_or_default("ACETATE_DEPLOY_HOST", "acetate")
dir         := env_var_or_default("ACETATE_DEPLOY_DIR", "~/acetate")
compose_dir := env_var_or_default("ACETATE_COMPOSE_DIR", dir)
services    := env_var_or_default("ACETATE_SERVICES", "")

# List recipes.
default:
    @just --list

# Pull main and rebuild/recreate containers on the server.
deploy:
    ssh -t {{host}} 'cd {{dir}} && git pull --ff-only && cd {{compose_dir}} && docker compose up -d --build {{services}}'

# Tail logs on the server (default: scheduler, where periodic jobs report).
logs service="scheduler":
    ssh -t {{host}} 'cd {{compose_dir}} && docker compose logs -f --tail=100 {{service}}'

# Container status on the server.
ps:
    ssh -t {{host}} 'cd {{compose_dir}} && docker compose ps {{services}}'

# Restart without rebuilding — e.g. after editing .env on the server.
restart:
    ssh -t {{host}} 'cd {{compose_dir}} && docker compose restart {{services}}'

# Read-only reconciliation plan. Compares local/remote ownership-aware row counts
# and previews additive audio sync without modifying anything.
db-reconcile:
    uv run python scripts/reconcile_vps.py plan --host "{{host}}" --compose-dir "{{compose_dir}}" --remote-data-dir "{{dir}}/data" --services "{{services}}"

# Promote locally computed embeddings/enrichment into the remote server DB.
# Remote stays authoritative for plays, snapshots, chat, and review verdicts.
db-reconcile-apply:
    uv run python scripts/reconcile_vps.py apply --host "{{host}}" --compose-dir "{{compose_dir}}" --remote-data-dir "{{dir}}/data" --services "{{services}}" --confirm PROMOTE_LOCAL_ENRICHMENT

# Roll back to a backup filename reported by db-reconcile-apply.
db-reconcile-restore backup:
    uv run python scripts/reconcile_vps.py restore --host "{{host}}" --compose-dir "{{compose_dir}}" --remote-data-dir "{{dir}}/data" --services "{{services}}" --backup "{{backup}}" --confirm RESTORE_VPS_BACKUP
