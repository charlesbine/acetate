# Authelia setup for Acetate

This guide covers running Acetate behind Authelia forward-auth and a reverse proxy
(Traefik, Caddy, or NGINX). Acetate implements trusted-header identity and role checks;
your proxy authenticates requests, copies Authelia's identity headers into the upstream
request, and prevents direct access to Acetate.

## Access model

Acetate recognizes three cumulative roles:

| Acetate role | Authelia membership | May do |
|---|---|---|
| `viewer` | authenticated, but neither group below | Browse the collection, playlists, stats, previews, and maps |
| `query` | `acetate-query` | Everything above, plus chat and CLAP text search |
| `admin` | `acetate-admin`, or a configured username | Everything, including jobs, smart-playlist changes, and Spotify mutations |

Create the owner's account with `acetate-admin`. Put friends who may use model-backed
queries in `acetate-query`. Other authenticated Authelia users remain read-only.
Authelia is self-hosted and does not charge per account.

Example conceptual user records (adapt to the existing file/LDAP backend):

```yaml
users:
  admin:
    displayname: Alice
    email: alice@example.com
    password: <argon2 password hash>
    groups: [acetate-admin]
  friend:
    displayname: Bob
    email: bob@example.com
    password: <argon2 password hash>
    groups: [acetate-query]
```

Generate hashes with the Authelia CLI/version already deployed; do not put plaintext
passwords in the users database.

## Authelia policy

Require authentication for the entire Acetate host. A typical rule is:

```yaml
access_control:
  rules:
    - domain: acetate.example.com
      policy: two_factor
```

It is fine to use `one_factor` on a Tailscale-only host if that is the established
policy, but Acetate itself assumes Authelia has already authenticated the request.
Do not make individual read routes `bypass`: Acetate needs an identity even for a
viewer.

## Forward-auth header contract

Configure the reverse proxy to call Authelia's current authorization endpoint:

```text
/api/authz/forward-auth
```

On a successful authorization response, copy these response headers into the request
sent to Acetate:

```text
Remote-User
Remote-Groups
Remote-Email
Remote-Name
```

For Traefik, the important portion of the middleware is:

```yaml
forwardAuth:
  address: http://authelia:9091/api/authz/forward-auth
  trustForwardHeader: true
  authResponseHeaders:
    - Remote-User
    - Remote-Groups
    - Remote-Email
    - Remote-Name
```

For Caddy, the equivalent is:

```caddyfile
forward_auth authelia:9091 {
    uri /api/authz/forward-auth
    copy_headers Remote-User Remote-Groups Remote-Email Remote-Name
}
reverse_proxy 127.0.0.1:8765
```

For NGINX, capture the authorization response and overwrite the upstream headers:

```nginx
auth_request_set $user   $upstream_http_remote_user;
auth_request_set $groups $upstream_http_remote_groups;
auth_request_set $email  $upstream_http_remote_email;
auth_request_set $name   $upstream_http_remote_name;

proxy_set_header Remote-User   $user;
proxy_set_header Remote-Groups $groups;
proxy_set_header Remote-Email  $email;
proxy_set_header Remote-Name   $name;
```

Use the full official Authelia integration snippet for the proxy already installed;
the fragments above only document Acetate's required contract.

## Acetate environment

Set these values in the VPS `.env` and rebuild/restart the app container:

```dotenv
ACETATE_AUTH_MODE=authelia
ACETATE_ADMIN_GROUPS=acetate-admin
ACETATE_QUERY_GROUPS=acetate-query
# Optional emergency/additional admins, comma-separated Authelia usernames:
ACETATE_ADMIN_USERS=admin
```

Group and username comparisons are case-insensitive. In Authelia mode, a request
without `Remote-User` is rejected with `401`. Docker Compose also defaults the app
container to `authelia` mode, so a missing `.env` setting does not silently enable the
local bypass on the VPS.

## Security invariants

1. Keep Acetate bound to VPS loopback (`127.0.0.1:8765`) or on a private container
   network reachable only by the reverse proxy.
2. Never expose the app container's port directly through Tailscale or the public
   interface. A client able to reach it could otherwise manufacture identity headers.
3. Strip/overwrite client-supplied `Remote-*` headers. Only values returned by
   Authelia's forward-auth response may reach Acetate.
4. Protect the whole hostname with forward-auth. `/healthz` is the sole app-level
   exception so the internal Docker health check works; it contains no private data.
5. Preserve the existing localhost Docker port binding in `docker-compose.yml`.

## Verification

Run these checks through the normal hostname, not directly against port 8765:

1. Logged out: redirected to Authelia.
2. Viewer account: pages render; query forms and mutation buttons are absent; direct
   POSTs to `/chat/new` and `/run/poll` return `403`.
3. `acetate-query` account: chat and sound search work; `/run/poll` still returns `403`.
4. `acetate-admin` account: pipeline and smart-playlist controls render and work.
5. From any non-proxy host, confirm TCP port 8765 cannot be reached.

Useful diagnostic: temporarily proxy a request inspector instead of Acetate and
confirm the four `Remote-*` headers have the expected values. Remove the inspector
after verification.

Official references:

- <https://www.authelia.com/integration/trusted-header-sso/introduction/>
- <https://www.authelia.com/integration/proxies/traefik/>
- <https://www.authelia.com/integration/proxies/caddy/>
- <https://www.authelia.com/integration/proxies/nginx/>
