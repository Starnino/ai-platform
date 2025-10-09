# === Compose file paths ===
SERVING_COMPOSE := apps/serving/docker-compose.yaml
MONITORING_COMPOSE := apps/monitoring/docker-compose.yaml
DC := docker compose -f

.PHONY: up down logs \
        up-serving down-serving logs-serving \
		up-monitoring down-monitoring logs-monitoring \

# --- all services ---
up: up-serving up-monitoring
down: down-monitoring down-serving
logs:
	-$(DC) $(SERVING_COMPOSE) logs -f || true &
	-$(DC) $(MONITORING_COMPOSE) logs -f || true &

# --- serving ---
up-serving:
	$(DC) $(SERVING_COMPOSE) up -d --build
down-serving:
	$(DC) $(SERVING_COMPOSE) down
logs-serving:
	$(DC) $(SERVING_COMPOSE) logs -f

# --- monitoring ---
up-monitoring:
	$(DC) $(MONITORING_COMPOSE) up -d --build
down-monitoring:
	$(DC) $(MONITORING_COMPOSE) down -v
logs-monitoring:
	$(DC) $(MONITORING_COMPOSE) logs -f

# --- clean up ---
clean:
	docker system prune -af --volumes