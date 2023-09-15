from apps.config import config
from flask import Flask
from flask_wtf.csrf import CSRFProtect
from pathlib import Path
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()

csrf = CSRFProtect()


def create_app(config_key):
    app = Flask(__name__)
    from apps.crud import views as crud_views

    app.config.from_object(config[config_key])

    db.init_app(app)
    Migrate(app, db)

    csrf.init_app(app)

    app.register_blueprint(crud_views.crud, url_prefix="/crud")
    return app
