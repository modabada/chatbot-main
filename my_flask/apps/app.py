from apps.config import config
from flask import Flask
from flask_wtf.csrf import CSRFProtect
from pathlib import Path
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager


db = SQLAlchemy()

csrf = CSRFProtect()

login_manager = LoginManager()
# 미 로그인이 리다이렉트 할 주소
login_manager.login_view = "auth.signup"
# 로그인 후에 표시할 메세지
login_manager.login_message = ""


def create_app(config_key):
    app = Flask(__name__)
    from apps.crud import views as crud_views
    from apps.auth import views as auth_views
    from apps.detector import views as dt_views

    app.config.from_object(config[config_key])

    db.init_app(app)
    Migrate(app, db)

    csrf.init_app(app)

    login_manager.init_app(app)

    app.register_blueprint(crud_views.crud, url_prefix="/crud")
    app.register_blueprint(auth_views.auth, url_prefix="/auth")
    app.register_blueprint(dt_views.dt)
    return app
