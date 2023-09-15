import os
from flask_mail import Mail, Message
from flask_debugtoolbar import DebugToolbarExtension
from email_validator import EmailNotValidError, validate_email
from flask import (
    Flask,
    flash,
    session,
    redirect,
    make_response,
    render_template,
    request,
    url_for,
)


app = Flask(__name__)
app.config["SECRET_KEY"] = "2AZSMss3p5QPbcY2hBsJ"
app.logger.setLevel(level=10)

# 디버그 툴바가 리다이렉트를 중단하지 않도록 한다
app.config["DEBUG_TB_INTERCEPT_REDIRECTS"] = False

# DebugToolbarExtension 에 애플리케이션을 설정한다
toolbar = DebugToolbarExtension(app)

# Mail 클래스의 설정 추가
app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER")
app.config["MAIL_PORT"] = os.environ.get("MAIL_PORT")
app.config["MAIL_USE_TLS"] = os.environ.get("MAIL_USE_TLS")
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER")

# flask-mail 확장 등록
mail = Mail(app)


@app.route("/")
def index():
    return "hello flask"


@app.route(
    "/hello/<name>",
    methods=["GET", "POST"],
    endpoint="/hello-endpoint",
)
def hello(name: str):
    return f"<h1> Hello {name}!</h1>"


@app.route("/name/<name>")
def show_name(name):
    return render_template("index.html", name=name)


@app.route("/contact", methods=["GET"])
def contact():
    # response 객체 가져오기
    response = make_response(render_template("contact.html"))

    # 쿠키 및 세션 설정
    response.set_cookie("flask key", "flask value")
    session["username"] = "ak"

    # response 반환
    return response


@app.route("/contact/complete", methods=["GET", "POST"])
def contact_complete():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        description = request.form["description"]

        is_valid = True

        # 유저네임 미입력 검증
        if not username:
            flash("사용자명은 필수입니다.")
            is_valid = False

        # 이메일 미입력 검증
        if not email:
            flash("메일 주소는 필수입니다.")
            is_valid = False

        # 이메일 형식 검증
        try:
            validate_email(email)
        except EmailNotValidError:
            flash("메일 주소의 형식으로 입력해 주세요")
            is_valid = False

        # 문의 내용 미입력 검증
        if not description:
            flash("문의 내용을 적어주세요.")
            is_valid = False

        # 검증 결과 올바르지 않다면.
        if not is_valid:
            return redirect(url_for("contact"))

        # 이메일 발송
        send_email(
            email,
            "문의 감사합니다",
            "contact_mail",
            username=username,
            description=description,
        )

        return redirect(url_for("contact_complete"))
    return render_template("contact_complete.html")


def send_email(to, subject, template, **kwargs):
    # 메일 송신 함수
    msg = Message(subject, recipients=[to])
    msg.body = render_template(template + ".txt", **kwargs)
    msg.html = render_template(template + ".html", **kwargs)
    mail.send(msg)
