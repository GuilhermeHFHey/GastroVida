web: gunicorn projeto.wsgi --log-file -
web: python manage.py collectstatic --no-input; gunicorn projeto.wsgi --log-file - --log-level debug