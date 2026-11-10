"""Fimpo chrome on twin stills — LinkedIn 4:5 and 16:9 canvases."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
REG = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

NAVY = (9, 12, 20)
INK = (248, 250, 252)
MUTED = (148, 163, 184)
LINE = (30, 41, 59)
BLUE = (28, 116, 233)
GREEN = (52, 211, 153)
RED = (248, 113, 113)

W4, H4 = 1080, 1350
W16, H16 = 1920, 1080
W9, H9 = 1080, 1920


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = BOLD if bold else REG
    return ImageFont.truetype(str(path), size)


def _fit(im: Image.Image, box_w: int, box_h: int) -> Image.Image:
    im = im.convert("RGB")
    scale = min(box_w / im.width, box_h / im.height)
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def _text_right(draw: ImageDraw.ImageDraw, y: int, text: str, font, fill, width: int, pad: int = 48) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((width - pad - tw, y), text, font=font, fill=fill)


def frame_4x5(
    sim: Image.Image,
    *,
    kicker: str = "",
    verdict: str | None = None,
) -> Image.Image:
    """1080×1350 (4:5) — LinkedIn feed, takes the phone screen."""
    canvas = Image.new("RGB", (W4, H4), NAVY)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 118, W4, 120], fill=BLUE)
    draw.text((48, 28), "FIMPO", font=_font(42, bold=True), fill=INK)
    draw.text((48, 76), "fimpo.net  ·  private AI platform", font=_font(20), fill=MUTED)
    _text_right(draw, 36, "50 / 50", _font(36, bold=True), INK, W4)
    _text_right(draw, 78, "randomised fence exam", _font(18), MUTED, W4)

    fitted = _fit(sim, 1080, 1088)
    canvas.paste(fitted, ((W4 - fitted.width) // 2, 136 + (1088 - fitted.height) // 2))

    draw.rectangle([0, 1236, W4, H4], fill=(6, 8, 14))
    draw.rectangle([0, 1236, W4, 1238], fill=LINE)
    left = (kicker or "digital twin  ·  lift over the fence")[:64]
    draw.text((48, 1256), left, font=_font(22), fill=INK)
    draw.text((48, 1296), "define  ·  practise  ·  test  ·  sign    ·    promote ≥ 80%", font=_font(18), fill=MUTED)
    if verdict:
        ok = verdict.upper().startswith("PASS")
        _text_right(draw, 1268, verdict, _font(28, bold=True), GREEN if ok else RED, W4)
    return canvas


def frame_16x9(
    sim: Image.Image,
    *,
    kicker: str = "",
    verdict: str | None = None,
) -> Image.Image:
    """1920×1080 — LinkedIn landscape / slides."""
    canvas = Image.new("RGB", (W16, H16), NAVY)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, 8, H16], fill=BLUE)
    draw.text((40, 28), "FIMPO", font=_font(36, bold=True), fill=INK)
    draw.text((40, 72), "fimpo.net  ·  private AI platform", font=_font(18), fill=MUTED)
    _text_right(draw, 32, "50 / 50  ·  randomised fence exam", _font(24, bold=True), INK, W16)
    _text_right(draw, 68, "contact = fail    ·    promote ≥ 80%", _font(18), MUTED, W16)

    fitted = _fit(sim, 1840, 880)
    canvas.paste(fitted, ((W16 - fitted.width) // 2, 110 + (880 - fitted.height) // 2))

    draw.rectangle([0, 1008, W16, H16], fill=(6, 8, 14))
    draw.text((40, 1028), (kicker or "digital twin  ·  lift over the fence")[:80], font=_font(22), fill=INK)
    if verdict:
        ok = verdict.upper().startswith("PASS")
        _text_right(draw, 1024, verdict, _font(26, bold=True), GREEN if ok else RED, W16)
    return canvas


def title_4x5() -> Image.Image:
    canvas = Image.new("RGB", (W4, H4), NAVY)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, 8, H4], fill=BLUE)
    draw.text((72, 420), "FIMPO", font=_font(72, bold=True), fill=INK)
    draw.text((72, 510), "fimpo.net  ·  digital twin  ·  fence exam", font=_font(28), fill=MUTED)
    draw.text((72, 620), "50 / 50", font=_font(88, bold=True), fill=INK)
    draw.text((72, 730), "on randomised fences", font=_font(32), fill=MUTED)
    draw.text((72, 820), "Touch the fence and the run fails.", font=_font(26), fill=INK)
    draw.text((72, 1180), "define  ·  practise  ·  test  ·  sign", font=_font(22), fill=MUTED)
    return canvas


def end_4x5() -> Image.Image:
    canvas = Image.new("RGB", (W4, H4), NAVY)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, 8, H4], fill=BLUE)
    draw.text((72, 480), "FIMPO", font=_font(64, bold=True), fill=INK)
    draw.text((72, 548), "fimpo.net", font=_font(24), fill=MUTED)
    draw.text((72, 620), "A skill is a contract.", font=_font(34), fill=INK)
    draw.text((72, 730), "Promote only at ≥ 80%.", font=_font(28), fill=MUTED)
    draw.text((72, 810), "No motors until the exam holds.", font=_font(28), fill=MUTED)
    draw.text((72, 1180), "private stack  ·  digital twins  ·  signed skills", font=_font(20), fill=MUTED)
    return canvas
