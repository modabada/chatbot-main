from pathlib import Path


basedir = Path(__file__).parent.parent


class BaseConfig:
    SECRET_KEY = "TestingSecretKey"
    WTF_CSRF_SECRET_KEY = "TestingCSRFSecretKey"


class LocalConfig(BaseConfig):
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{basedir / 'local.sqlite'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # SQL 콘솔 로그 출력 설정
    SQLALCHEMY_ECHO = True


class TestingConfig(BaseConfig):
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{basedir / 'local.sqlite'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # SQL 콘솔 로그 출력 설정
    WTF_CSRF_ENABLED = False


config = {
    "testing": TestingConfig,
    "local": LocalConfig,
}
