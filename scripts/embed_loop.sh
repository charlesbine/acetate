#!/bin/sh
# One request controller; inference locks are also enforced in both workers.
set -eu
exec python /app/scripts/embed_loop.py
