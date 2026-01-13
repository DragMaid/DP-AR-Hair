db:
	docker compose up db

backend:
	poetry run uvicorn server:app --host 0.0.0.0 --port 8000

cron:
	docker compose up cronjob

nginx:
	docker compose up nginx

seed:
	poetry run python seed.py

debug:
	poetry run python seed.py -d

clear:
	docker compose down -v
