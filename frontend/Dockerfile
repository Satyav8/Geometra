# The chat UI. Build context is frontend/:
#   docker build -t sam-frontend ./frontend
#
# Unlike the backend image this one runs no application code - Vite compiles the React
# app to static files and nginx serves them, which is also how the UI is deployed on
# Render. The difference is that nginx here also proxies the API, so the browser sees one
# origin and CORS never enters into it.

# ----------------------------------------------------------------------------- build
FROM node:20.18.0-alpine AS build

WORKDIR /app

# package files alone first: this layer is rebuilt only when dependencies change, so an
# edit to a component reuses the cached install.
COPY package.json package-lock.json ./
# npm ci, not npm install: installs exactly what the lockfile pins, and fails rather than
# silently resolving something newer. A handover image should build the same way in a
# year as it does today.
RUN npm ci

COPY . .

# Empty on purpose, and it must be EXPORTED rather than left unset: the app reads
# VITE_API_URL with ?? , so "" means "call the same origin this page came from" and
# unset would fall back to the dev default. nginx then proxies those calls to the
# backend. Override it only if you serve the UI from a different origin than the API.
ARG VITE_API_URL=""
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# ----------------------------------------------------------------------------- runtime
FROM nginx:1.27-alpine AS runtime

# Replaces the default site, which would otherwise serve nginx's welcome page at /.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

# 8080 rather than 80 so the server can run unprivileged.
EXPOSE 8080

# wget is in the alpine base, so unlike the backend image this needs no Python one-liner.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
