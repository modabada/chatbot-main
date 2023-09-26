import random
import uuid
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
from apps.app import db
from apps.crud.models import User
from apps.detector.forms import UploadImageForm, DetectorForm, DeleteForm
from apps.detector.models import UserImage, UserImageTag
from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    send_from_directory,
    url_for,
    request,
)


from flask_login import current_user, login_required
from PIL import Image


dt = Blueprint(
    "detector",
    __name__,
    template_folder="templates",
)


@dt.route("/")
def index():
    user_images = (
        db.session.query(User, UserImage)
        .join(UserImage)
        .filter(User.id == UserImage.user_id)
        .all()
    )

    user_image_tag_dict = {}
    for user_image in user_images:
        user_image_tags = (
            db.session.query(UserImageTag)
            .filter(UserImageTag.user_image_id == user_image.UserImage.id)
            .all()
        )
        user_image_tag_dict[user_image.UserImage.id] = user_image_tags

    detector_form = DetectorForm()

    delete_form = DeleteForm()

    return render_template(
        "detector/index.html",
        user_images=user_images,
        user_image_tag_dict=user_image_tag_dict,
        detector_form=detector_form,
        delete_form=delete_form,
    )


@dt.route("/images/<path:filename>")
def image_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@dt.route("/upload", methods=["GET", "POST"])
@login_required
def upload_image():
    form = UploadImageForm()
    if form.validate_on_submit():
        file = form.image.data
        # 파일의 파일명과 확장자를 얻고 ,파읾을 uuid 로 변경
        ext = Path(file.filename).suffix
        image_uuid_file_name = str(uuid.uuid4()) + ext
        # 이미지 저장
        image_path = Path(
            current_app.config["UPLOAD_FOLDER"],
            image_uuid_file_name,
        )
        file.save(image_path)

        # db 에 저장
        user_image = UserImage(
            user_id=current_user.id,
            image_path=image_uuid_file_name,
        )
        db.session.add(user_image)
        db.session.commit()

        return redirect(url_for("detector.index"))
    return render_template("detector/upload.html", form=form)


@dt.route("/detect/<string:image_id>", methods=["POST"])
@login_required
def detect(image_id):
    user_image = (
        db.session.query(UserImage)
        .filter(
            UserImage.id == image_id,
        )
        .first()
    )
    if user_image is None:
        flash("감지대상의 이미지가 존재하지 않습니다")
        return redirect(url_for("detector.index"))

    target_image_path = Path(current_app.config["UPLOAD_FOLDER"], user_image.image_path)
    tags, detected_image_file_name = exec_detect(target_image_path)

    try:
        # 데이터베이스에 태그와 변환 후의 이미지 경로 정보를 저장한다
        save_detected_image_tags(user_image, tags, detected_image_file_name)
    except Exception as e:
        flash("물체 감지 처리에서 오류가 발생했습니다.")
        db.session.rollback()
        current_app.logger.error(e)
        return redirect(url_for("detector.index"))

    return redirect(url_for("detector.index"))


@dt.route("/images/delete/<string:image_id>", methods=["POST"])
@login_required
def delete_image(image_id):
    try:
        db.session.query(UserImageTag).filter(
            UserImageTag.user_image_id == image_id
        ).delete()
        db.session.query(UserImage).filter(UserImage.id == image_id).delete()

        db.session.commit()
    except Exception as e:
        flash("이미지 삭제중 오류가 발생했습니다")
        current_app.logger.error(e)
        db.session.rollback()
    return redirect(url_for("detector.index"))


@dt.route("/images/search", methods=["GET"])
def search():
    user_images = db.session.query(User, UserImage).join(
        UserImage,
        User.id == UserImage.user_id,
    )

    search_text = request.args.get("search")
    user_image_tag_dict = dict()
    filtered_user_image = list()

    for user_image in user_images:
        # 검색어가 빈 경우는 모든 결과 반환
        if not search_text:
            user_image_tag = (
                db.session.query(UserImageTag)
                .filter(UserImageTag.user_image_id == user_image.UserImage.id)
                .all()
            )
        else:
            user_image_tag = (
                db.session.query(UserImageTag)
                .filter(UserImageTag.user_image_id == user_image.UserImage.id)
                .filter(UserImageTag.tag_name.like("%" + search_text + "%"))
                .all()
            )

            # 검색 결과가 없다면 반환 안함
            if not user_image_tag:
                continue

            # 결과가 나왔다면 결과의 모든 태그 가져오기
            user_image_tags = (
                db.session.query(UserImageTag)
                .filter(UserImageTag.user_image_id == user_image.UserImage.id)
                .all()
            )

        user_image_tag_dict[user_image.UserImage.id] = user_image_tags
        filtered_user_image.append(user_image)

    detector_form = DetectorForm()
    delete_form = DeleteForm()

    return render_template(
        "detector/index.html",
        user_images=filtered_user_image,
        user_image_tag_dict=user_image_tag_dict,
        detector_form=detector_form,
        delete_form=delete_form,
    )


def make_color(labels):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in labels]
    color = random.choice(colors)
    return color


def make_line(result_image):
    line = round(0.002 * max(result_image.shape[0:2])) + 1
    return line


def draw_lines(c1, c2, result_image, line, color):
    cv2.rectangle(result_image, c1, c2, color, thickness=line)
    return cv2


def draw_text(result_image, line, c1, cv2, color, labels, label):
    display_txt = f"{labels[label]}"
    font = max(line - 1, 1)
    t_size = cv2.getTextSize(display_txt, 0, fontScale=line / 3, thickness=font)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(result_image, c1, c2, color, -1)
    cv2.putText(
        result_image,
        display_txt,
        (c1[0], c1[1] - 2),
        0,
        line / 3,
        [255, 255, 255],
        thickness=font,
        lineType=cv2.LINE_AA,
    )
    return cv2


def exec_detect(target_image_path):
    labels = current_app.config["LABELS"]
    image = Image.open(target_image_path).convert("RGB")
    image_tensor = torchvision.transforms.functional.to_tensor(image)

    model = torch.load(Path(current_app.root_path, "detector", "model.pt"))
    model = model.eval()

    output = model([image_tensor])[0]
    tags = []
    result_image = np.array(image.copy())

    # 모델이 감지한 각 물체만큼 반복하여 이미지 위에 그리기
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        if score > 0.5 and labels[label] not in tags:
            print(score)
            print(labels[label])
            color = make_color(labels)
            line = make_line(result_image)
            c1 = (int(box[0]), int(box[1]))
            c2 = (int(box[2]), int(box[3]))
            cv2 = draw_lines(c1, c2, result_image, line, color)
            cv2 = draw_text(result_image, line, c1, cv2, color, labels, label)
            tags.append(labels[label])

    detected_image_file_name = str(uuid.uuid4()) + ".jpg"

    detected_image_file_path = str(
        Path(current_app.config["UPLOAD_FOLDER"], detected_image_file_name)
    )

    cv2.imwrite(detected_image_file_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    return tags, detected_image_file_name


def save_detected_image_tags(user_image, tags, detected_image_file_name):
    user_image.image_path = detected_image_file_name

    user_image.is_detected = True
    db.session.add(user_image)
    for tag in tags:
        user_image_tag = UserImageTag(user_image_id=user_image.id, tag_name=tag)
        db.session.add(user_image_tag)
    db.session.commit()
