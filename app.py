import os
import uuid
import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from PIL import Image, ImageOps

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle, KeepTogether
)

from flask import Flask, render_template, request, send_file, abort, session, redirect, url_for
from werkzeug.utils import secure_filename


# --------------------------------------------------
# APP AND PATH CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

# Used for one-time result storage between POST and redirected GET.
# Set FLASK_SECRET_KEY as an environment variable when deploying.
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "perfecthue-development-secret-change-before-deployment"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR, "static", "uploads"
)

app.config["DOWNLOAD_FOLDER"] = os.path.join(
    BASE_DIR, "static", "downloads"
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["DOWNLOAD_FOLDER"], exist_ok=True)

DATASET_PATH = os.path.join(BASE_DIR, "foundation_dataset.csv")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place foundation_dataset.csv beside app.py."
    )

df = pd.read_csv(DATASET_PATH)

required_columns = {
    "brand", "product", "name", "R", "G", "B", "in_india"
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        "Dataset is missing columns: "
        + ", ".join(sorted(missing_columns))
    )

df[["R", "G", "B"]] = df[["R", "G", "B"]].apply(
    pd.to_numeric, errors="coerce"
)

df = df.dropna(subset=["R", "G", "B"]).copy()

df[["R", "G", "B"]] = (
    df[["R", "G", "B"]].clip(0, 255).astype(int)
)


# --------------------------------------------------
# MEDIAPIPE FACEMESH
# --------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    refine_landmarks=False,
    min_detection_confidence=0.5
)


# --------------------------------------------------
# IMAGE HELPERS
# --------------------------------------------------

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def validate_image(file_stream):
    try:
        file_stream.seek(0)
        image = Image.open(file_stream)
        image.verify()
        file_stream.seek(0)
        return True
    except Exception:
        file_stream.seek(0)
        return False


def load_image_as_bgr(image_path):
    with Image.open(image_path) as pil_image:
        pil_image = ImageOps.exif_transpose(
            pil_image
        ).convert("RGB")

        image_rgb = np.array(pil_image)

    return cv2.cvtColor(
        image_rgb, cv2.COLOR_RGB2BGR
    )


def save_bgr_image(image_bgr, output_path):
    image_rgb = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2RGB
    )

    Image.fromarray(image_rgb).save(
        output_path, format="JPEG", quality=95
    )


# --------------------------------------------------
# SKIN COLOUR AND UNDERTONE HELPERS
# --------------------------------------------------


def detect_undertone(rgb):
    """
    Estimate undertone using CIELAB colour information.
    This is an image-based estimate and may be affected
    by lighting, makeup and camera settings.
    """

    rgb_pixel = np.uint8([[list(map(int, rgb))]])

    lab_pixel = cv2.cvtColor(
        rgb_pixel, cv2.COLOR_RGB2LAB
    )[0][0].astype(float)

    a_star = lab_pixel[1] - 128.0
    b_star = lab_pixel[2] - 128.0

    chroma = float(
        np.sqrt(a_star ** 2 + b_star ** 2)
    )

    if chroma < 8:
        return "Neutral"

    hue_angle = float(
        np.degrees(np.arctan2(b_star, a_star))
    )
    print("DEBUG HUE ANGLE:", hue_angle)

    # Updated classification thresholds
    if hue_angle >= 50:
        return "Warm"

    elif hue_angle <= 35:
        return "Cool"

    else:
        return "Neutral"
    


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0]),
        int(rgb[1]),
        int(rgb[2])
    )


def check_india_availability(value):

    if pd.isna(value):
        return "Not available in India"

    if isinstance(value, (bool, np.bool_)):
        is_available = bool(value)

    elif isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):
        is_available = value == 1

    else:
        is_available = str(value).strip().lower() in {
            "true", "1", "yes", "available"
        }

    if is_available:
        return "Available in India"

    return "Not available in India"


# --------------------------------------------------
# SKIN TONE DETECTION
# --------------------------------------------------


def get_skin_tone(image_path):
    try:
        image = load_image_as_bgr(image_path)
    except Exception:
        return None, None, None, "Could not read the uploaded image."

    if image is None:
        return None, None, None, "Could not read the uploaded image."

    max_dimension = 640
    height, width = image.shape[:2]

    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale))
        )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None, None, None, (
            "No face detected. Please upload a clear, "
            "front-facing photograph."
        )

    if len(results.multi_face_landmarks) > 1:
        return None, None, None, (
            "Multiple faces detected. Please upload "
            "a photograph containing only one face."
        )

    landmarks = results.multi_face_landmarks[0].landmark

    height, width, _ = image.shape
    sample_landmarks = [109, 338, 117, 346, 50, 280]

    patch_radius = max(3, int(width * 0.012))
    skin_pixels = []

    for landmark_index in sample_landmarks:
        landmark = landmarks[landmark_index]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        x1 = max(0, x - patch_radius)
        x2 = min(width, x + patch_radius + 1)
        y1 = max(0, y - patch_radius)
        y2 = min(height, y + patch_radius + 1)

        patch = image_rgb[y1:y2, x1:x2]

        if patch.size == 0:
            continue

        hsv_patch = cv2.cvtColor(
            patch.astype(np.uint8),
            cv2.COLOR_RGB2HSV
        )

        value_channel = hsv_patch[:, :, 2]

        valid_pixels = patch[
            (value_channel >= 45) & (value_channel <= 235)
        ]

        if len(valid_pixels) > 0:
            skin_pixels.extend(valid_pixels.tolist())

    if len(skin_pixels) < 20:
        return None, None, None, (
            "Could not sample enough clear skin areas. "
            "Please upload a well-lit, front-facing photograph."
        )

    average_colour = np.median(
        np.asarray(skin_pixels, dtype=np.uint8),
        axis=0
    ).astype(int)

    return image, average_colour, landmarks, None


# --------------------------------------------------
# FOUNDATION RECOMMENDATIONS
# --------------------------------------------------

def recommend_foundation(avg_color, top_n=3):
    """Find closest foundation shades using RGB distance."""

    dataset = df.copy()

    dataset["distance"] = np.sqrt(
        (dataset["R"] - int(avg_color[0])) ** 2
        + (dataset["G"] - int(avg_color[1])) ** 2
        + (dataset["B"] - int(avg_color[2])) ** 2
    )

    return (
        dataset.sort_values(by="distance")
        .head(top_n)
        .reset_index(drop=True)
    )


# --------------------------------------------------
# APPLY FOUNDATION TO FACE
# --------------------------------------------------


def apply_foundation(image, foundation_rgb, landmarks, mode="half"):

    height, width, _ = image.shape

    mask = np.zeros((height, width), dtype=np.uint8)

    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in landmarks
    ]

    face_contour = [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ]

    face_points = np.array(
        [points[index] for index in face_contour],
        dtype=np.int32
    )

    cv2.fillConvexPoly(mask, face_points, 255)

    left_eye = [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246
    ]

    right_eye = [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398
    ]

    mouth = [
        78, 95, 88, 178, 87, 14, 317, 402,
        318, 324, 308, 415, 310, 311, 312,
        13, 82, 81, 80
    ]

    for excluded_region in [left_eye, right_eye, mouth]:
        excluded_points = np.array(
            [points[index] for index in excluded_region],
            dtype=np.int32
        )
        cv2.fillConvexPoly(mask, excluded_points, 0)

    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    if mode == "half":
        half_mask = np.zeros_like(mask)
        half_mask[:, :width // 2] = 255
        mask = cv2.bitwise_and(mask, half_mask)

    foundation_bgr = np.array(
        [
            int(foundation_rgb[2]),
            int(foundation_rgb[1]),
            int(foundation_rgb[0])
        ],
        dtype=np.uint8
    )

    foundation_layer = np.full_like(image, foundation_bgr)

    alpha = mask.astype(float) / 255.0
    alpha = np.expand_dims(alpha, axis=2) * 0.30

    blended = (
        image.astype(float) * (1 - alpha)
        + foundation_layer.astype(float) * alpha
    )

    return np.clip(blended, 0, 255).astype(np.uint8)


# --------------------------------------------------
# CREATE A SINGLE, DESIGNED PDF RESULT
# --------------------------------------------------

def create_result_pdf(
    uploaded_path,
    uploaded_name,
    skin_tone,
    skin_tone_hex,
    undertone,
    recommendations,
    shades_output
):
    """Create one compact, coordinated PerfectHue PDF report."""

    from xml.sax.saxutils import escape

    result_id = uuid.uuid4().hex
    pdf_path = os.path.join(
        app.config["DOWNLOAD_FOLDER"],
        f"PerfectHue_Result_{result_id}.pdf"
    )

    page_width, page_height = A4
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=12 * mm,
        bottomMargin=16 * mm,
        title="PerfectHue Personal Shade Report",
        author="PerfectHue"
    )

    # PerfectHue colour palette
    plum = colors.HexColor("#59434D")
    rose = colors.HexColor("#B87983")
    pale = colors.HexColor("#FBF4F1")
    muted = colors.HexColor("#806B70")
    border = colors.HexColor("#EAD8D8")
    green = colors.HexColor("#47856B")
    red = colors.HexColor("#B66C78")
    white = colors.white

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PHBrandCompact",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=26,
        textColor=plum,
        alignment=TA_CENTER,
        spaceAfter=2
    ))
    styles.add(ParagraphStyle(
        name="PHSubCompact",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=muted,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name="PHSectionCompact",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=plum,
        spaceBefore=5,
        spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name="PHBodyCompact",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=muted,
        spaceAfter=1
    ))
    styles.add(ParagraphStyle(
        name="PHShadeName",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=rose
    ))
    styles.add(ParagraphStyle(
        name="PHMicro",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8,
        textColor=muted,
        alignment=TA_CENTER
    ))

    def fit_image(path, max_width, max_height):
        """Return a proportionally scaled ReportLab image."""
        with Image.open(path) as source_image:
            image_width, image_height = source_image.size

        scale = min(
            max_width / float(image_width),
            max_height / float(image_height)
        )
        image_flowable = RLImage(
            path,
            width=image_width * scale,
            height=image_height * scale
        )
        image_flowable.hAlign = "CENTER"
        return image_flowable

    story = [
        Spacer(1, 1 * mm),
        Paragraph("PERFECTHUE", styles["PHBrandCompact"]),
        Paragraph(
            "YOUR PERSONAL FOUNDATION SHADE REPORT",
            styles["PHSubCompact"]
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            f"Created on {datetime.now().strftime('%d %B %Y')}",
            styles["PHSubCompact"]
        ),
        Spacer(1, 4 * mm)
    ]

    # Uploaded photo and skin analysis in one compact panel.
    if os.path.isfile(uploaded_path):
        try:
            photo_flowable = fit_image(
                uploaded_path, 34 * mm, 43 * mm
            )
        except Exception:
            photo_flowable = Paragraph(
                "Uploaded photograph",
                styles["PHMicro"]
            )
    else:
        photo_flowable = Paragraph(
            "Uploaded photograph",
            styles["PHMicro"]
        )

    safe_hex = escape(str(skin_tone_hex))
    analysis_text = Paragraph(
        "<b>YOUR SKIN ANALYSIS</b><br/><br/>"
        f"<font size='17' color='{safe_hex}'><b>{safe_hex}</b></font><br/>"
        f"RGB: {int(skin_tone[0])}, {int(skin_tone[1])}, "
        f"{int(skin_tone[2])}<br/><br/>"
        f"<b>Estimated undertone: {escape(str(undertone))}</b><br/><br/>"
        "<font size='7'>This is an image-based estimate. Lighting, "
        "makeup and camera settings may affect the result.</font>",
        styles["PHBodyCompact"]
    )

    analysis_panel = Table(
        [[photo_flowable, analysis_text]],
        colWidths=[43 * mm, 121 * mm]
    )
    analysis_panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale),
        ("BOX", (0, 0), (-1, -1), 0.7, border),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6)
    ]))

    story.extend([
        analysis_panel,
        Spacer(1, 3 * mm),
        Paragraph(
            "YOUR CLOSEST FOUNDATION MATCHES",
            styles["PHSectionCompact"]
        ),
        Paragraph(
            "Ranked by colour distance from the detected skin colour "
            "in the foundation dataset.",
            styles["PHBodyCompact"]
        ),
        Spacer(1, 2 * mm)
    ])

    # A compact row for each match. The two preview images are placed
    # side by side in the same row as the shade details.
    for index, shade in enumerate(recommendations):
        brand = escape(str(shade.get("brand", "Unknown brand")))
        shade_name = escape(str(shade.get("name", "Unnamed shade")))
        product = escape(str(shade.get("product", "")))
        shade_hex = str(shade.get("hex", "#C49A83"))

        try:
            swatch_colour = colors.HexColor(shade_hex)
        except Exception:
            swatch_colour = colors.HexColor("#C49A83")

        availability = str(shade.get(
            "in_india_label",
            "Availability not specified"
        ))
        availability_colour = (
            green if availability == "Available in India" else red
        )

        details = Paragraph(
            f"<b>MATCH {index + 1:02d}  |  {brand}</b><br/>"
            f"<font size='10' color='{rose.hexval()}'><b>{shade_name}</b></font><br/>"
            f"{product}<br/>"
            f"<b>HEX:</b> {escape(shade_hex)}<br/>"
            f"RGB: {int(shade['R'])} / {int(shade['G'])} / "
            f"{int(shade['B'])}<br/>"
            f"<font color='{availability_colour.hexval()}'>"
            f"<b>{escape(availability)}</b></font>",
            styles["PHBodyCompact"]
        )

        swatch = Table([[""]], colWidths=[14 * mm], rowHeights=[14 * mm])
        swatch.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), swatch_colour),
            ("BOX", (0, 0), (-1, -1), 1, white)
        ]))

        previews_by_mode = {}
        for preview in shades_output:
            if preview.get("shade_index") != shade.get("shade_index", index):
                continue

            relative_path = str(preview.get("img", ""))
            preview_path = os.path.join(
                BASE_DIR,
                "static",
                relative_path.replace("/", os.sep)
            )

            if not os.path.isfile(preview_path):
                continue

            try:
                previews_by_mode[preview.get("mode", "")] = fit_image(
                    preview_path, 22 * mm, 29 * mm
                )
            except Exception:
                app.logger.exception(
                    "Could not add a preview image to the PDF: %s",
                    preview_path
                )

        preview_columns = []
        for mode, label in (
            ("half", "HALF-FACE"),
            ("full", "FULL-FACE")
        ):
            image_flowable = previews_by_mode.get(mode)
            if image_flowable is None:
                image_flowable = Paragraph(
                    "Preview unavailable",
                    styles["PHMicro"]
                )

            preview_card = Table(
                [[Paragraph(label, styles["PHMicro"])],
                 [image_flowable]],
                colWidths=[29 * mm]
            )
            preview_card.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1)
            ]))
            preview_columns.append(preview_card)

        preview_pair = Table(
            [[preview_columns[0], preview_columns[1]]],
            colWidths=[30 * mm, 30 * mm]
        )
        preview_pair.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0)
        ]))

        recommendation_row = Table(
            [[details, swatch, preview_pair]],
            colWidths=[79 * mm, 18 * mm, 67 * mm]
        )
        recommendation_row.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), pale),
            ("BOX", (0, 0), (-1, -1), 0.6, border),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
        ]))

        story.extend([
            recommendation_row,
            Spacer(1, 2.5 * mm)
        ])

    story.extend([
        Spacer(1, 2 * mm),
        Paragraph(
            "PerfectHue  •  Beauty is personal. Your shade should be too.",
            styles["PHSubCompact"]
        )
    ])

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(border)
        canvas.setLineWidth(0.5)
        canvas.line(
            16 * mm, 12 * mm,
            page_width - 16 * mm, 12 * mm
        )
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(muted)
        canvas.drawCentredString(
            page_width / 2,
            7 * mm,
            f"PerfectHue  •  Page {document.page}"
        )
        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    return result_id


# --------------------------------------------------
# MAIN FLASK ROUTES
# --------------------------------------------------

def render_homepage(**extra_context):
    """Render the normal homepage, with no previous analysis by default."""
    context = {
        "recommendations": None,
        "shades_output": [],
        "error": None,
        "uploaded_file": None,
        "skin_tone": None,
        "skin_tone_hex": None,
        "undertone": None,
        "download_url": None
    }
    context.update(extra_context)
    return render_template("index.html", **context)


@app.route("/", methods=["GET", "POST"])
def index():

    # A normal visit or refresh of "/" always shows the clean homepage.
    if request.method == "GET":
        return render_homepage()

    # Validate the uploaded file.
    if (
        "file" not in request.files
        or request.files["file"].filename == ""
    ):
        return render_homepage(error="Please upload an image.")

    file = request.files["file"]

    if not allowed_file(file.filename):
        return render_homepage(
            error="Invalid file type. Only JPG, JPEG and PNG are allowed."
        )

    if not validate_image(file.stream):
        return render_homepage(
            error="Invalid image file. Please upload a valid photograph."
        )

    original_name = secure_filename(file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"

    absolute_upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_name
    )

    file.save(absolute_upload_path)
    uploaded_file = "uploads/" + unique_name

    image, avg_color, face_landmarks, face_error = get_skin_tone(
        absolute_upload_path
    )

    if face_error:
        return render_homepage(
            error=face_error,
            uploaded_file=uploaded_file
        )

    skin_tone = [int(value) for value in avg_color]
    skin_tone_hex = rgb_to_hex(skin_tone)
    undertone = detect_undertone(skin_tone)

    # Find the closest foundation shades.
    recommendations_df = recommend_foundation(avg_color, top_n=3)
    recommendations = []
    shades_output = []

    for index, row in recommendations_df.iterrows():

        availability_label = check_india_availability(
            row.get("in_india")
        )

        shade = {
            "brand": str(row.get("brand", "Unknown brand")),
            "product": str(row.get("product", "")),
            "name": str(row.get("name", "Unnamed shade")),
            "in_india_label": availability_label,
            "R": int(row["R"]),
            "G": int(row["G"]),
            "B": int(row["B"]),
            "hex": rgb_to_hex([
                row["R"],
                row["G"],
                row["B"]
            ]),
            "shade_index": int(index)
        }

        recommendations.append(shade)

        shade_name = secure_filename(
            str(row["name"]).replace(" ", "_")
        ) or "shade"

        # Generate half-face and full-face previews.
        for mode in ["half", "full"]:

            preview_filename = (
                f"{mode}_{index}_{shade_name}_"
                f"{uuid.uuid4().hex[:8]}.jpg"
            )

            output_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                preview_filename
            )

            applied_image = apply_foundation(
                image,
                [
                    int(row["R"]),
                    int(row["G"]),
                    int(row["B"])
                ],
                face_landmarks,
                mode
            )

            try:
                save_bgr_image(applied_image, output_path)

                shades_output.append({
                    "img": "uploads/" + preview_filename,
                    "label": (
                        f"{row['brand']} - "
                        f"{row['name']} ({mode})"
                    ),
                    "mode": mode,
                    "shade_index": int(index)
                })

            except Exception:
                app.logger.exception(
                    "Could not save preview for shade %s",
                    row["name"]
                )

    download_url = None

    # Create one PDF download after successful analysis.
    if recommendations and skin_tone and undertone:
        result_id = create_result_pdf(
            absolute_upload_path,
            original_name,
            skin_tone,
            skin_tone_hex,
            undertone,
            recommendations,
            shades_output
        )

        download_url = f"/download/{result_id}"

    # Store the result briefly in the signed session cookie, then redirect.
    # The redirected GET consumes it once. Refreshing that GET will show
    # the clean homepage rather than repeating the previous result.
    session["perfecthue_result"] = {
        "recommendations": recommendations,
        "shades_output": shades_output,
        "uploaded_file": uploaded_file,
        "skin_tone": skin_tone,
        "skin_tone_hex": skin_tone_hex,
        "undertone": undertone,
        "download_url": download_url
    }

    return redirect(url_for("show_result"))


@app.route("/result", methods=["GET"])
def show_result():
    """Display one successful analysis once; refresh returns to homepage."""
    result = session.pop("perfecthue_result", None)

    if not result:
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        error=None,
        **result
    )


# --------------------------------------------------
# DOWNLOAD ROUTE
# --------------------------------------------------

@app.route("/download/<result_id>")
def download_result(result_id):

    if not re.fullmatch(r"[a-f0-9]{32}", result_id):
        abort(404)

    pdf_path = os.path.join(
        app.config["DOWNLOAD_FOLDER"],
        f"PerfectHue_Result_{result_id}.pdf"
    )

    if not os.path.isfile(pdf_path):
        abort(404)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="My_PerfectHue_Results.pdf",
        mimetype="application/pdf"
    )


# --------------------------------------------------
# ERROR HANDLER
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(_error):

    return render_template(
        "index.html",
        error=(
            "The image is too large. "
            "Please upload an image smaller than 10 MB."
        )
    ), 413


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
