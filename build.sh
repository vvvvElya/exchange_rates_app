#!/bin/bash
# build.sh

echo "📦 Inštalácia balíkov..."
pip install -r requirements.txt

echo "🔧 Migrácia databázy..."
python manage.py migrate

echo "🎨 Zber statických súborov..."
python manage.py collectstatic --noinput
