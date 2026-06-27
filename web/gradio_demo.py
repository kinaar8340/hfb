#!/usr/bin/env python3
"""Gradio web demo for Hopf Flux Bubble — flux metric control panel."""

from __future__ import annotations

import logging
import os
import re
import time
import traceback
from collections.abc import Callable, Iterator

import gradio as gr

from demo_core import (
    GITHUB_URL,
    HF_SPACE_URL,
    HFB_WALLPAPER_URL,
    VQC_URL,
    default_run_params,
    get_build_label,
    is_hf_space,
    run_flux_bubble_demo,
    run_parameter_sweep_summary,
    run_warp_compare_figure,
)

logger = logging.getLogger(__name__)


def _patch_gradio_client_bool_schema() -> None:
    try:
        from gradio_client import utils as client_utils

        if getattr(client_utils, "_hfb_bool_patch", False):
            return

        orig_get_type = client_utils.get_type

        def get_type(schema):  # noqa: ANN001
            if isinstance(schema, bool):
                return "boolean"
            return orig_get_type(schema)

        client_utils.get_type = get_type
        client_utils._hfb_bool_patch = True
        logger.info("Patched gradio_client bool JSON-schema handling")
    except Exception:
        logger.warning("Could not patch gradio_client", exc_info=True)


_patch_gradio_client_bool_schema()

_DEFAULTS = default_run_params()
_GALLERY_FLUX_URL = "https://raw.githubusercontent.com/kinaar8340/hfb/main/outputs/flux_bubble_demo.png"
_GALLERY_3D_URL = "https://raw.githubusercontent.com/kinaar8340/hfb/main/outputs/flux_bubble_3d.png"
_GALLERY_WARP_URL = "https://raw.githubusercontent.com/kinaar8340/hfb/main/outputs/warp_compare.png"

_HFB_ACCENT = "#2dd4bf"
_HFB_FIELD_FILL = "rgba(8, 18, 16, 0.50)"
_HFB_TAB_GREEN_BG = "#14532d"
_HFB_TAB_GREEN_BORDER = "#1ed760"
_HFB_TAB_GREEN_TEXT = "#86efac"
_HFB_TAB_ORANGE_BG = "#134e4a"
_HFB_TAB_ORANGE_BORDER = "#2dd4bf"
_HFB_TAB_ORANGE_TEXT = "#99f6e4"
_HFB_MATRIX_GREEN = "#33ff66"
_HFB_LOGO_GOLD = "#c9a227"
_HFB_HOME_KEY_BG = "#000000"

CAVEATS_MD = """
> **Effective analog only** — not literal GR curvature or superluminal transport.
> Stability metrics (max |R|, ergo fraction, curvature flux Φ_R) are heuristic proxies
> for tabletop exploration, not experimental claims.
"""

FLUX_BANNER_MD = """
> **Flux bubble demo** — conformal factor Ω from defect wall + vortex flow, effective shift
> (warp analog), acoustic c_eff² − v², and conformal geodesic rays. Topology diagnostics
> report stable_proxy, Φ_R, and linking proxies from `bubble/stability.py`.
"""

WARP_BANNER_MD = """
> **Warp compare** — numeric L¹ fidelity between the analog shift profile and a symbolic
> Alcubierre β field (`analog_gravity/warp_compare.py`). Tune vs, rs, σ for the reference warp.
"""

STABILITY_BANNER_MD = """
> **Stability sweep** — grid over bubble radius × circulation from `configs/default.yaml`.
> Each cell reports stable_proxy, max |R|, and ergo_fraction for quick parameter scouting.
"""

OPTICS_LOGO_HTML = """
<div class="hfb-optics-logo" role="img" aria-label="Hopf Flux Bubble Flux Metric Control Panel">
  <span class="hfb-optics-brand">HOPF FLUX BUBBLE</span>
  <span class="hfb-optics-panel-title">Flux Metric Control Panel</span>
  <span class="hfb-optics-subtitle">DEFECT · HOPF · FLOW · ANALOG GRAVITY</span>
</div>
"""

_OPTICS_TERM_BAR = "─" * 48
_OPTICS_TERM_CHAR_DELAY_S = 0.014
_OPTICS_TERM_NEWLINE_DELAY_S = 0.048
_OPTICS_TERM_UPLINK_DELAY_S = 0.22
_OPTICS_TERM_CURSOR = "▌"
_OPTICS_TERM_RELEASE_DELAY_S = 0.25

TERM_KEYPAD_PROG_COLS = 12
TERM_KEYPAD_PROG_ROWS = 2
TERM_KEYPAD_COUNT = TERM_KEYPAD_PROG_COLS * TERM_KEYPAD_PROG_ROWS
TERM_KEYPAD_DEFINED: dict[int, str] = {
    1: "home",
    2: "status",
    3: "flux",
    4: "warp",
    5: "help",
}
TERM_KEYPAD_HOME_KEY = "key01"
TERM_KEYPAD_DESCRIPTIONS: dict[int, str] = {
    1: "Return to selection menu — momentary",
    2: "Status — pipeline & environment",
    3: "Flux bubble — metric & ray trace",
    4: "Warp & stability — compare + sweep",
    5: "Help — keypad & caveats",
}
TERM_UI_MENU = "menu"
TERM_UI_PAGE = "page"
TERM_NAV_KEYS: tuple[str, ...] = (
    "dpad_select",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "clear",
)
TERM_DPAD_HOLD_KEYS: tuple[str, ...] = (
    "dpad_select",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
)
TERM_NAV_DEFINED: dict[str, str] = {
    "dpad_select": "Enter — confirm menu item",
    "dpad_up": "Up — previous menu item",
    "dpad_down": "Down — next menu item",
    "dpad_left": "Left — previous menu item",
    "dpad_right": "Right — next menu item",
    "clear": "Clear — blank display",
}
TERM_KEYPAD_CONTROL_ORDER: tuple[str, ...] = (
    *TERM_NAV_KEYS,
    *(f"key{i:02d}" for i in range(1, TERM_KEYPAD_COUNT + 1)),
)


def _strip_md_plain(text: str) -> str:
    plain = re.sub(r"^>\s*", "", text.strip(), flags=re.MULTILINE)
    plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", plain)
    plain = re.sub(r"`([^`]+)`", r"\1", plain)
    return plain.strip()


def _optics_terminal_frame(title: str, body: str) -> str:
    return f"{title}\n{_OPTICS_TERM_BAR}\n{body}"


def _optics_assigned_keypad_lines() -> str:
    lines = []
    for index in sorted(TERM_KEYPAD_DEFINED):
        tag = "01 Home" if index == 1 else f"{index:02d}"
        lines.append(f"  [{tag}]  {TERM_KEYPAD_DESCRIPTIONS[index]}")
    for nav_key in TERM_NAV_KEYS:
        if nav_key in TERM_NAV_DEFINED:
            tag = "CLR" if nav_key == "clear" else nav_key.removeprefix("dpad_").upper()
            lines.append(f"  [{tag}]  {TERM_NAV_DEFINED[nav_key]}")
    return "\n".join(lines)


def _optics_terminal_home() -> str:
    return _optics_terminal_frame("PROGRAMMABLE KEYPAD", _optics_assigned_keypad_lines())


def _default_term_ui_state() -> dict:
    return {"mode": TERM_UI_MENU, "index": 0}


def _optics_terminal_menu(menu_index: int) -> str:
    lines = [
        "▲▼ ◀▶ move highlight · enter confirm · 01 Home",
        "",
    ]
    for index, (_action, keypad_key, label, _stream) in enumerate(_term_menu_items()):
        mark = "▶" if index == menu_index else " "
        lines.append(f"{keypad_key:02d} --- [{mark}] {label}")
    return _optics_terminal_frame("SELECTION MENU", "\n".join(lines))


def _term_menu_items() -> tuple[tuple[str, int, str, Callable[[], Iterator[str]]], ...]:
    return (
        ("home", 1, "Home Keypad Legend", _stream_optics_terminal_home),
        ("status", 2, "Status Pipeline & Environment", _stream_optics_terminal_status),
        ("flux", 3, "Flux Bubble Metric & Ray Trace", _stream_optics_terminal_flux),
        ("warp", 4, "Warp Compare & Stability Sweep", _stream_optics_terminal_warp),
        ("help", 5, "Help Keypad & Caveats", _stream_optics_terminal_help),
    )


def _term_menu_index_for_action(action: str) -> int:
    for index, (key, _keypad, _label, _stream) in enumerate(_term_menu_items()):
        if key == action:
            return index
    return 0


def _term_menu_step(menu_index: int, delta: int) -> int:
    return (menu_index + delta) % len(_term_menu_items())


def _optics_terminal_status() -> str:
    on_hf = is_hf_space()
    env = "Hugging Face Space" if on_hf else "Local Gradio"
    grid_note = "Grid 96×96 (HF fast path)" if on_hf else "Grid 128×128 (local)"
    return _optics_terminal_frame(
        "SYSTEM STATUS",
        "\n".join(
            [
                f"Environment : {env}",
                f"Package     : hopf-flux-bubble v0.1.4",
                f"Grid        : {grid_note}",
                "Pipeline    : defect wall → vortex flow → acoustic metric",
                "Modules     : bubble/ · hopf/ · analog_gravity/ · optics/",
                "",
                get_build_label().replace("`", ""),
                "",
                "Tune dials below, then RUN FLUX BUBBLE / WARP / SWEEP.",
            ]
        ),
    )


def _optics_terminal_flux() -> str:
    return _optics_terminal_frame("FLUX BUBBLE DEMO", _strip_md_plain(FLUX_BANNER_MD))


def _optics_terminal_warp() -> str:
    body = "\n\n".join(
        [
            _strip_md_plain(WARP_BANNER_MD),
            _strip_md_plain(STABILITY_BANNER_MD),
        ]
    )
    return _optics_terminal_frame("WARP & STABILITY", body)


def _optics_terminal_help() -> str:
    return _optics_terminal_frame(
        "KEYPAD REFERENCE",
        "\n".join(
            [
                "D-pad TUI — ▲▼ ◀▶ move · enter opens highlighted item",
                "Prog keys 02–05 mirror menu items · 01 Home → menu",
                "",
                _optics_assigned_keypad_lines(),
                "",
                _strip_md_plain(CAVEATS_MD),
            ]
        ),
    )


def _stream_optics_terminal_text(full_text: str) -> Iterator[str]:
    shown = ""
    for ch in full_text:
        shown += ch
        yield shown + _OPTICS_TERM_CURSOR
        time.sleep(_OPTICS_TERM_NEWLINE_DELAY_S if ch == "\n" else _OPTICS_TERM_CHAR_DELAY_S)
    yield shown


def _optics_terminal_uplink_banner(mode: str) -> str:
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    return f"> UPLINK {mode.upper()} @ {stamp} UTC…\n"


def _optics_terminal_stream(builder: Callable[[], str], *, mode: str) -> Iterator[str]:
    banner = _optics_terminal_uplink_banner(mode)
    yield banner + _OPTICS_TERM_CURSOR
    time.sleep(_OPTICS_TERM_UPLINK_DELAY_S)
    yield from _stream_optics_terminal_text(banner + builder())


def _stream_optics_terminal_home() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_home, mode="home")


def _stream_optics_terminal_status() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_status, mode="status")


def _stream_optics_terminal_flux() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_flux, mode="flux")


def _stream_optics_terminal_warp() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_warp, mode="warp")


def _stream_optics_terminal_help() -> Iterator[str]:
    yield from _optics_terminal_stream(_optics_terminal_help, mode="help")


def _stream_optics_terminal_clear(current: str) -> Iterator[str]:
    text = current or ""
    if not text:
        yield ""
        return
    chunk = max(1, len(text) // 36)
    for end in range(len(text), -1, -chunk):
        yield text[:end] + (_OPTICS_TERM_CURSOR if end else "")
        time.sleep(0.01)
    yield ""


TERM_KEYPAD_STREAMERS: dict[str, Callable[[], Iterator[str]]] = {}


def _term_key_id(index: int) -> str:
    return f"key{index:02d}"


def _term_keypad_label(index: int) -> str:
    if index == 1:
        return "01 Home"
    return f"{index:02d}"


def _term_key_is_defined_prog(key: str) -> bool:
    for index in TERM_KEYPAD_DEFINED:
        if index == 1:
            continue
        if _term_key_id(index) == key:
            return True
    return False


def _term_key_btn_classes(key: str, active: str) -> list[str]:
    classes = ["hfb-optics-key"]
    if key in TERM_NAV_KEYS:
        classes.append("hfb-optics-dpad-key")
    if key == TERM_KEYPAD_HOME_KEY:
        classes.append("hfb-optics-key-home")
    elif key.startswith("dpad_"):
        classes.append("hfb-optics-key-dpad")
    if key == "clear":
        classes.append("hfb-optics-key-clear")
    if _term_key_is_defined_prog(key):
        classes.append("hfb-optics-key-defined")
    if key == active and key != TERM_KEYPAD_HOME_KEY:
        classes.append("active")
    return classes


def _term_keypad_btn_updates(active: str) -> tuple:
    return tuple(
        gr.update(elem_classes=_term_key_btn_classes(key_id, active))
        for key_id in TERM_KEYPAD_CONTROL_ORDER
    )


def _term_keypad_outputs(terminal_text: str, active: str, ui_state: dict | None = None) -> tuple:
    state = _default_term_ui_state() if ui_state is None else ui_state
    return (terminal_text, *_term_keypad_btn_updates(active), active, state)


def _term_yield_stream_then_release(
    stream: Iterator[str],
    *,
    active: str,
    ui_state: dict,
    release_delay: float | None = None,
) -> Iterator[tuple]:
    delay = _OPTICS_TERM_RELEASE_DELAY_S if release_delay is None else release_delay
    last_partial = ""
    for partial in stream:
        last_partial = partial
        yield _term_keypad_outputs(partial, active, ui_state)
    time.sleep(delay)
    yield _term_keypad_outputs(last_partial, "", ui_state)


def _term_stream_with_latch(
    stream_fn: Callable[[], Iterator[str]],
    *,
    active: str,
    ui_state: dict,
) -> Iterator[tuple]:
    yield from _term_yield_stream_then_release(stream_fn(), active=active, ui_state=ui_state)


def _make_term_stream_click(
    active_key: str,
    stream_fn: Callable[[], Iterator[str]],
    *,
    menu_action: str | None = None,
):
    def handler(ui_state: dict) -> Iterator[tuple]:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        if menu_action is not None:
            state = {
                "mode": TERM_UI_PAGE,
                "index": _term_menu_index_for_action(menu_action),
            }
        yield from _term_stream_with_latch(stream_fn, active=active_key, ui_state=state)

    return handler


def _make_term_clear_click(active_key: str):
    def handler(current: str, ui_state: dict) -> Iterator[tuple]:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        yield from _term_yield_stream_then_release(
            _stream_optics_terminal_clear(current),
            active=active_key,
            ui_state=state,
        )

    return handler


def _make_term_dpad_click(active_key: str):
    def handler(_current: str, ui_state: dict) -> Iterator[tuple]:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        mode = state.get("mode", TERM_UI_MENU)
        menu_index = int(state.get("index", 0))
        nav_delta = {
            "dpad_up": -1,
            "dpad_left": -1,
            "dpad_down": 1,
            "dpad_right": 1,
        }

        if active_key in nav_delta:
            if mode == TERM_UI_PAGE:
                menu_state = {"mode": TERM_UI_MENU, "index": menu_index}
                text = _optics_terminal_menu(menu_index)
            else:
                new_index = _term_menu_step(menu_index, nav_delta[active_key])
                menu_state = {"mode": TERM_UI_MENU, "index": new_index}
                text = _optics_terminal_menu(new_index)
            yield _term_keypad_outputs(text, active_key, menu_state)
            time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
            yield _term_keypad_outputs(text, "", menu_state)
            return

        if active_key == "dpad_select":
            if mode == TERM_UI_MENU:
                _action, _keypad, _label, stream_fn = _term_menu_items()[menu_index]
                page_state = {"mode": TERM_UI_PAGE, "index": menu_index}
                yield from _term_yield_stream_then_release(
                    stream_fn(),
                    active="dpad_select",
                    ui_state=page_state,
                )
                return
            menu_state = {"mode": TERM_UI_MENU, "index": menu_index}
            text = _optics_terminal_menu(menu_index)
            yield _term_keypad_outputs(text, active_key, menu_state)
            time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
            yield _term_keypad_outputs(text, "", menu_state)

    return handler


def _make_term_latch_click(active_key: str):
    def handler(current: str, ui_state: dict) -> tuple:
        state = dict(ui_state) if ui_state else _default_term_ui_state()
        return _term_keypad_outputs(current, active_key, state)

    return handler


def _make_term_home_momentary():
    def handler(current_active: str, ui_state: dict) -> Iterator[tuple]:
        menu_state = {"mode": TERM_UI_MENU, "index": 0}
        menu_text = _optics_terminal_menu(0)
        yield _term_keypad_outputs(menu_text, current_active, menu_state)
        time.sleep(_OPTICS_TERM_RELEASE_DELAY_S)
        yield _term_keypad_outputs(menu_text, "", menu_state)

    return handler


def _term_boot_home() -> tuple:
    boot_state = _default_term_ui_state()
    return _term_keypad_outputs(_optics_terminal_menu(0), "", boot_state)


def _register_term_keypad_streamers() -> None:
    TERM_KEYPAD_STREAMERS.update(
        {
            "home": _stream_optics_terminal_home,
            "status": _stream_optics_terminal_status,
            "flux": _stream_optics_terminal_flux,
            "warp": _stream_optics_terminal_warp,
            "help": _stream_optics_terminal_help,
        }
    )


_register_term_keypad_streamers()


def _external_tab_html(label: str, url: str, tab_id: str) -> str:
    return (
        f'<a href="{url}" class="hfb-source-tab" data-tab="{tab_id}" '
        f'target="_blank" rel="noopener noreferrer">{label}</a>'
    )


def _source_tab_btn_update(*, active: bool) -> gr.Update:
    if active:
        return gr.update(interactive=False, elem_classes=["hfb-source-tab", "active"])
    return gr.update(interactive=True, elem_classes=["hfb-source-tab"], variant="secondary")


def _home_tab_update(*, on_demo_page: bool) -> gr.Update:
    if on_demo_page:
        return gr.update(interactive=False, elem_classes=["hfb-source-tab", "active"], variant="secondary")
    return gr.update(interactive=True, elem_classes=["hfb-source-tab"], variant="secondary")


def _close_links_panels() -> tuple:
    return (
        gr.update(visible=False),
        _source_tab_btn_update(active=False),
        False,
        gr.update(visible=False),
        _source_tab_btn_update(active=False),
        False,
    )


def _nav_to_page(page: str) -> tuple:
    on_demo = page == "demo"
    closed = _close_links_panels()
    return (
        gr.update(visible=on_demo),
        gr.update(visible=not on_demo),
        _home_tab_update(on_demo_page=on_demo),
        _source_tab_btn_update(active=not on_demo),
        *closed,
        _home_tab_update(on_demo_page=on_demo),
        _source_tab_btn_update(active=not on_demo),
        page,
    )


def _toggle_caveats(is_open: bool) -> tuple:
    show = not is_open
    return (
        gr.update(visible=show),
        _source_tab_btn_update(active=show),
        show,
        gr.update(visible=False),
        _source_tab_btn_update(active=False),
        False,
    )


def _minimize_caveats() -> tuple:
    return (
        gr.update(visible=False),
        _source_tab_btn_update(active=False),
        False,
    )


def _gallery_grid_html() -> str:
    panels = (
        ("Flux bubble 4-panel", _GALLERY_FLUX_URL),
        ("3D torus surface", _GALLERY_3D_URL),
        ("Warp compare", _GALLERY_WARP_URL),
    )
    imgs = "".join(
        f'<figure class="hfb-gallery-figure">'
        f'<img src="{url}" alt="{title}" loading="lazy" />'
        f'<figcaption>{title}</figcaption></figure>'
        for title, url in panels
    )
    return f'<div class="hfb-gallery-wrap">{imgs}</div>'


def _build_hfb_theme() -> gr.themes.Base:
    return (
        gr.themes.Base(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.zinc,
            neutral_hue=gr.themes.colors.zinc,
        )
        .set(
            body_background_fill="transparent",
            body_background_fill_dark="transparent",
            background_fill_primary="transparent",
            background_fill_primary_dark="transparent",
            background_fill_secondary="transparent",
            background_fill_secondary_dark="transparent",
            block_background_fill=_HFB_FIELD_FILL,
            block_background_fill_dark=_HFB_FIELD_FILL,
            panel_background_fill=_HFB_FIELD_FILL,
            panel_background_fill_dark=_HFB_FIELD_FILL,
            input_background_fill=_HFB_FIELD_FILL,
            input_background_fill_dark=_HFB_FIELD_FILL,
            body_text_color="#e0f2f1",
            body_text_color_dark="#e0f2f1",
            block_label_text_color="#99f6e4",
            block_label_text_color_dark="#99f6e4",
            block_title_text_color="#ccfbf1",
            block_title_text_color_dark="#ccfbf1",
            border_color_primary="rgba(255, 255, 255, 0.12)",
            border_color_primary_dark="rgba(255, 255, 255, 0.12)",
            button_primary_background_fill="#0d9488",
            button_primary_background_fill_dark="#0d9488",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="rgba(12, 28, 24, 0.92)",
            button_secondary_background_fill_dark="rgba(12, 28, 24, 0.92)",
            button_secondary_text_color="#e0f2f1",
            button_secondary_text_color_dark="#e0f2f1",
            checkbox_label_background_fill="transparent",
            checkbox_label_background_fill_dark="transparent",
            slider_color=_HFB_ACCENT,
            slider_color_dark=_HFB_ACCENT,
            link_text_color=_HFB_ACCENT,
            link_text_color_dark=_HFB_ACCENT,
            link_text_color_hover="#5eead4",
            link_text_color_hover_dark="#5eead4",
            link_text_color_active=_HFB_ACCENT,
            link_text_color_active_dark=_HFB_ACCENT,
            link_text_color_visited=_HFB_ACCENT,
            link_text_color_visited_dark=_HFB_ACCENT,
        )
    )


WALLPAPER_HEAD = f"""
<style id="hfb-wallpaper-style">
#hfb-wallpaper {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: -9999 !important;
    pointer-events: none !important;
    background-color: #061210 !important;
    background-image: url('{HFB_WALLPAPER_URL}') !important;
    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
}}
</style>
<script>
(function() {{
    function mountWallpaper() {{
        if (document.getElementById('hfb-wallpaper')) return;
        var wp = document.createElement('div');
        wp.id = 'hfb-wallpaper';
        wp.setAttribute('aria-hidden', 'true');
        document.body.insertBefore(wp, document.body.firstChild);
    }}
    if (document.body) mountWallpaper();
    document.addEventListener('DOMContentLoaded', mountWallpaper);
    window.addEventListener('load', mountWallpaper);
}})();
</script>
"""

HFB_CSS = f"""
:root, :root .dark {{
    --body-background-fill: transparent !important;
    --background-fill-primary: transparent !important;
    --background-fill-secondary: transparent !important;
    --block-background-fill: {_HFB_FIELD_FILL} !important;
    --panel-background-fill: {_HFB_FIELD_FILL} !important;
    --input-background-fill: {_HFB_FIELD_FILL} !important;
    --body-text-color: #e0f2f1 !important;
    --block-label-text-color: #99f6e4 !important;
    --block-title-text-color: #ccfbf1 !important;
    --border-color-primary: rgba(255, 255, 255, 0.12) !important;
    --link-text-color: {_HFB_ACCENT} !important;
    color-scheme: dark;
}}
html {{ background-color: #061210 !important; min-height: 100% !important; }}
body {{
    background: transparent !important;
    color: #e0f2f1 !important;
    min-height: 100vh !important;
    width: 100% !important;
    overflow-x: hidden !important;
}}
body::before {{
    content: "" !important;
    position: fixed !important;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: -9998 !important;
    pointer-events: none !important;
    background-color: #061210 !important;
    background-image: url('{HFB_WALLPAPER_URL}') !important;
    background-size: cover !important;
    background-position: center center !important;
}}
.gradio-container {{
    position: relative !important;
    width: 100% !important;
    max-width: 100% !important;
    background: transparent !important;
}}
.gradio-container .block {{
    background-color: {_HFB_FIELD_FILL} !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(4px);
}}
.gradio-container .markdown, .gradio-container .prose, .gradio-container .markdown p {{
    color: #e0f2f1 !important;
}}
.gradio-container .hfb-source-tab,
.gradio-container .hfb-source-tabs-row button.hfb-source-tab,
.gradio-container .hfb-nav-cell a.hfb-source-tab {{
    color: {_HFB_MATRIX_GREEN} !important;
    -webkit-text-fill-color: {_HFB_MATRIX_GREEN} !important;
    text-decoration: underline !important;
    text-decoration-color: {_HFB_MATRIX_GREEN} !important;
    background: transparent !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    box-shadow: none !important;
    padding: 0 !important;
}}
.gradio-container .hfb-source-tab.active,
.gradio-container .hfb-source-tabs-row button.hfb-source-tab.active {{
    color: {_HFB_LOGO_GOLD} !important;
    -webkit-text-fill-color: {_HFB_LOGO_GOLD} !important;
    text-decoration-color: {_HFB_LOGO_GOLD} !important;
    text-decoration-thickness: 2px !important;
}}
.gradio-container .hfb-optics-panel {{
    background: linear-gradient(165deg, #102820 0%, #0a1a14 38%, #061210 100%) !important;
    border: 3px solid #1d6b5c !important;
    border-radius: 14px !important;
    padding: 0 1rem 1rem !important;
    margin: 0.5rem 0 0.75rem 0 !important;
}}
.gradio-container .hfb-optics-panel-header {{
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 0.75rem 1.1rem !important;
    padding: 0.7rem 0.85rem 1.35rem !important;
    border-bottom: 1px solid rgba(29, 107, 92, 0.65) !important;
    background: linear-gradient(180deg, #0f2018 0%, #061210 100%) !important;
    min-height: 5.25rem !important;
}}
.gradio-container .hfb-optics-panel-nav {{
    flex: 1 1 18rem !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0.28rem !important;
}}
.gradio-container .hfb-nav-spreadsheet-row {{
    display: grid !important;
    grid-template-columns: 4.75rem repeat(5, minmax(4.5rem, 1fr)) !important;
    gap: 0.2rem 0.45rem !important;
    align-items: center !important;
    width: 100% !important;
}}
.gradio-container .hfb-nav-row-label {{
    justify-self: end !important;
    text-align: right !important;
    color: #e0f2f1 !important;
    font-weight: 600 !important;
}}
.gradio-container .hfb-optics-logo {{
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    gap: 0.1rem !important;
    min-width: 10.5rem !important;
    padding-right: 0.65rem !important;
    border-right: 1px solid rgba(29, 107, 92, 0.45) !important;
}}
.gradio-container .hfb-optics-brand {{
    font-size: 0.62rem !important;
    letter-spacing: 0.28em !important;
    color: {_HFB_LOGO_GOLD} !important;
    font-weight: 700 !important;
}}
.gradio-container .hfb-optics-panel-title {{
    font-size: 1.15rem !important;
    letter-spacing: 0.12em !important;
    color: #ccfbf1 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
}}
.gradio-container .hfb-optics-subtitle {{
    font-size: 0.68rem !important;
    letter-spacing: 0.22em !important;
    color: #5eead4 !important;
}}
.gradio-container .hfb-optics-panel .hfb-optics-terminal textarea {{
    background: rgba(2, 10, 4, 0.1) !important;
    border: 2px inset #1a4d2a !important;
    color: #33ff66 !important;
    font-family: "Courier New", Courier, monospace !important;
    font-size: 0.78rem !important;
    min-height: 13.5rem !important;
}}
.gradio-container .hfb-optics-keypad {{
    background: linear-gradient(180deg, #0c1814 0%, #061210 100%) !important;
    border: 2px inset #1d4d3f !important;
    border-radius: 10px !important;
    padding: 0.42rem 0.38rem 0.48rem !important;
}}
.gradio-container .hfb-optics-keypad button.hfb-optics-key {{
    flex: 1 1 0 !important;
    min-height: 3rem !important;
    background: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: "Courier New", Courier, monospace !important;
    font-size: 1.44rem !important;
    font-weight: 700 !important;
}}
.gradio-container button.hfb-optics-key-home,
.gradio-container button.hfb-optics-key-home span {{
    color: {_HFB_MATRIX_GREEN} !important;
    background: {_HFB_HOME_KEY_BG} !important;
}}
.gradio-container button.hfb-optics-key-defined:not(.active),
.gradio-container button.hfb-optics-key-defined:not(.active) span {{
    color: {_HFB_MATRIX_GREEN} !important;
}}
.gradio-container button.hfb-optics-key.active {{
    background: {_HFB_MATRIX_GREEN} !important;
    color: #000000 !important;
}}
.gradio-container .hfb-optics-dial-wrap {{
    background: rgba(0, 0, 0, 0.22) !important;
    border: 1px solid #1d4d3f !important;
    border-radius: 10px !important;
    padding: 0.55rem 0.65rem 0.45rem !important;
}}
.gradio-container .hfb-gallery-wrap {{
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)) !important;
    gap: 0.75rem !important;
    width: 100% !important;
}}
.gradio-container .hfb-gallery-figure img {{
    width: 100% !important;
    border-radius: 8px !important;
    background: rgba(6, 18, 16, 0.35) !important;
}}
.gradio-container .hfb-gallery-figure figcaption {{
    color: #99f6e4 !important;
    font-size: 0.82rem !important;
    text-align: center !important;
    margin-top: 0.35rem !important;
}}
.gradio-container button.hfb-panel-minimize {{
    border: 1px solid {_HFB_TAB_GREEN_BORDER} !important;
    background: {_HFB_TAB_GREEN_BG} !important;
    color: {_HFB_TAB_GREEN_TEXT} !important;
}}
.gradio-container .hfb-links-panel {{
    margin: 0 0 0.35rem 0 !important;
    padding: 0.65rem 0.85rem !important;
}}
.gradio-container .hfb-figure-panel img {{
    width: 100% !important;
    object-fit: contain !important;
}}
footer {{ visibility: hidden; }}
"""


def _run_params_from_ui(
    bubble_radius: float,
    wall_width: float,
    defect_amplitude: float,
    circulation: float,
    sound_speed: float,
    defect_profile: str,
    use_3d_torus: bool,
    hopf_index: float,
    major_radius: float,
    minor_radius: float,
) -> dict:
    return {
        "bubble_radius": float(bubble_radius),
        "wall_width": float(wall_width),
        "defect_amplitude": float(defect_amplitude),
        "circulation": float(circulation),
        "sound_speed": float(sound_speed),
        "defect_profile": str(defect_profile),
        "use_3d_torus": bool(use_3d_torus),
        "hopf_index": int(hopf_index),
        "major_radius": float(major_radius),
        "minor_radius": float(minor_radius),
    }


def run_flux(
    bubble_radius: float,
    wall_width: float,
    defect_amplitude: float,
    circulation: float,
    sound_speed: float,
    defect_profile: str,
    use_3d_torus: bool,
    hopf_index: float,
    major_radius: float,
    minor_radius: float,
    include_3d: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str | None, str | None]:
    params = _run_params_from_ui(
        bubble_radius,
        wall_width,
        defect_amplitude,
        circulation,
        sound_speed,
        defect_profile,
        use_3d_torus,
        hopf_index,
        major_radius,
        minor_radius,
    )
    if is_hf_space():
        include_3d = include_3d and True
    try:
        progress(0.1, desc="Building flux bubble metric…")
        metrics, panel_path, path_3d = run_flux_bubble_demo(**params, include_3d=include_3d)
        progress(1.0, desc="Done")
        return metrics, panel_path, path_3d
    except Exception as exc:
        logger.exception("run_flux failed")
        err = f"Error: {exc}\n\n{traceback.format_exc()}"
        return err, None, None


def run_warp(
    bubble_radius: float,
    wall_width: float,
    defect_amplitude: float,
    circulation: float,
    sound_speed: float,
    defect_profile: str,
    use_3d_torus: bool,
    hopf_index: float,
    major_radius: float,
    minor_radius: float,
    vs: float,
    rs: float,
    sigma: float,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, str | None]:
    params = _run_params_from_ui(
        bubble_radius,
        wall_width,
        defect_amplitude,
        circulation,
        sound_speed,
        defect_profile,
        use_3d_torus,
        hopf_index,
        major_radius,
        minor_radius,
    )
    try:
        progress(0.1, desc="Comparing warp profiles…")
        summary, out_path = run_warp_compare_figure(**params, vs=vs, rs=rs, sigma=sigma)
        progress(1.0, desc="Done")
        return summary, out_path
    except Exception as exc:
        logger.exception("run_warp failed")
        return f"Error: {exc}\n\n{traceback.format_exc()}", None


def run_sweep(
    defect_profile: str,
    use_3d_torus: bool,
    hopf_index: float,
    major_radius: float,
    minor_radius: float,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> str:
    try:
        progress(0.1, desc="Running parameter sweep…")
        summary = run_parameter_sweep_summary(
            defect_profile=defect_profile,
            use_3d_torus=bool(use_3d_torus),
            hopf_index=int(hopf_index),
            major_radius=float(major_radius),
            minor_radius=float(minor_radius),
        )
        progress(1.0, desc="Done")
        return summary
    except Exception as exc:
        logger.exception("run_sweep failed")
        return f"Error: {exc}\n\n{traceback.format_exc()}"


def build_app() -> gr.Blocks:
    on_hf = is_hf_space()
    viz3d_info = (
        "3D surface disabled on HF by default — enable for pseudo-3D torus slice (slower)"
        if on_hf
        else "Adds flux_bubble_3d surface plot alongside the 4-panel figure"
    )

    with gr.Blocks(
        title="Hopf Flux Bubble — Live Demo",
        analytics_enabled=False,
        theme=_build_hfb_theme(),
        head=WALLPAPER_HEAD,
        css=HFB_CSS,
        fill_width=True,
    ) as demo:
        current_page = gr.State("demo")
        caveats_open = gr.State(False)

        with gr.Column(visible=False, elem_classes=["hfb-links-panel"]) as panel_caveats:
            with gr.Row():
                gr.Markdown("### Caveats — effective analog only")
                caveats_minimize_btn = gr.Button("▲", elem_classes=["hfb-panel-minimize"], scale=0)
            gr.Markdown(CAVEATS_MD)

        with gr.Column(visible=True) as page_demo:
            with gr.Group(elem_classes=["hfb-optics-panel"]):
                with gr.Row(elem_classes=["hfb-optics-panel-header"]):
                    gr.HTML(OPTICS_LOGO_HTML)
                    with gr.Column(elem_classes=["hfb-optics-panel-nav"], scale=1):
                        with gr.Row(elem_classes=["hfb-nav-spreadsheet-row"]):
                            gr.HTML('<span class="hfb-nav-row-label">Source:</span>')
                            with gr.Column(elem_classes=["hfb-nav-cell"], scale=1, min_width=72):
                                tab_demo_btn = gr.Button(
                                    "Live Demo",
                                    elem_classes=["hfb-source-tab", "active"],
                                    interactive=False,
                                    scale=0,
                                    variant="secondary",
                                )
                            with gr.Column(elem_classes=["hfb-nav-cell"], scale=1, min_width=72):
                                tab_gallery_btn = gr.Button(
                                    "Gallery",
                                    elem_classes=["hfb-source-tab"],
                                    scale=0,
                                    variant="secondary",
                                )
                            with gr.Column(elem_classes=["hfb-nav-cell"], scale=1, min_width=72):
                                tab_caveats_btn = gr.Button(
                                    "Caveats",
                                    elem_classes=["hfb-source-tab"],
                                    scale=0,
                                    variant="secondary",
                                )
                            with gr.Column(elem_classes=["hfb-nav-cell"], scale=1, min_width=72):
                                gr.HTML('<span class="hfb-nav-cell-empty" aria-hidden="true">&nbsp;</span>')
                            with gr.Column(elem_classes=["hfb-nav-cell"], scale=1, min_width=72):
                                gr.HTML('<span class="hfb-nav-cell-empty" aria-hidden="true">&nbsp;</span>')
                        with gr.Row(elem_classes=["hfb-nav-spreadsheet-row"]):
                            gr.HTML('<span class="hfb-nav-row-label">Links:</span>')
                            with gr.Column(elem_classes=["hfb-nav-cell"], scale=1, min_width=72):
                                gr.HTML(_external_tab_html("GitHub", GITHUB_URL, "github"))
                            with gr.Column(elem_classes=["hfb-nav-cell"], scale=1, min_width=72):
                                gr.HTML(_external_tab_html("vqc_proto", VQC_URL, "vqc"))
                            with gr.Column(elem_classes=["hfb-nav-cell"], scale=1, min_width=72):
                                gr.HTML(_external_tab_html("CLI README", f"{GITHUB_URL}#quick-start", "cli"))
                            with gr.Column(elem_classes=["hfb-nav-cell"], scale=1, min_width=72):
                                gr.HTML('<span class="hfb-nav-cell-empty" aria-hidden="true">&nbsp;</span>')
                            with gr.Column(elem_classes=["hfb-nav-cell"], scale=1, min_width=72):
                                gr.HTML('<span class="hfb-nav-cell-empty" aria-hidden="true">&nbsp;</span>')

                optics_terminal = gr.Textbox(
                    label="Matrix status display — selection menu · d-pad nav",
                    value=_optics_terminal_menu(0),
                    lines=12,
                    max_lines=16,
                    interactive=False,
                    elem_classes=["hfb-optics-terminal-wrap", "hfb-optics-terminal"],
                )
                term_active_key = gr.State("")
                term_ui_state = gr.State(_default_term_ui_state())
                term_all_btns: dict[str, gr.Button] = {}
                _dpad_row_labels = {
                    "dpad_select": "enter",
                    "dpad_up": "▲",
                    "dpad_down": "▼",
                    "dpad_left": "◀",
                    "dpad_right": "▶",
                    "clear": "clear",
                }

                with gr.Column(elem_classes=["hfb-optics-keypad"]):
                    with gr.Row(elem_classes=["hfb-optics-dpad-row"], equal_height=True):
                        for nav_key in TERM_NAV_KEYS:
                            term_all_btns[nav_key] = gr.Button(
                                _dpad_row_labels[nav_key],
                                elem_classes=_term_key_btn_classes(nav_key, ""),
                                scale=1,
                                variant="secondary",
                            )
                    with gr.Row(elem_classes=["hfb-optics-prog-row"], equal_height=True):
                        for index in range(1, 13):
                            key_id = _term_key_id(index)
                            term_all_btns[key_id] = gr.Button(
                                _term_keypad_label(index),
                                elem_classes=_term_key_btn_classes(key_id, ""),
                                scale=1,
                                variant="secondary",
                            )
                    with gr.Row(elem_classes=["hfb-optics-prog-row"], equal_height=True):
                        for index in range(13, 25):
                            key_id = _term_key_id(index)
                            term_all_btns[key_id] = gr.Button(
                                _term_keypad_label(index),
                                elem_classes=_term_key_btn_classes(key_id, ""),
                                scale=1,
                                variant="secondary",
                            )

                term_keypad_outputs = [
                    optics_terminal,
                    *[term_all_btns[key_id] for key_id in TERM_KEYPAD_CONTROL_ORDER],
                    term_active_key,
                    term_ui_state,
                ]

                with gr.Row(elem_classes=["hfb-optics-tune-row"]):
                    bubble_radius = gr.Slider(
                        0.5,
                        2.0,
                        value=_DEFAULTS["bubble_radius"],
                        step=0.05,
                        label="Bubble radius",
                        elem_classes=["hfb-optics-dial-wrap"],
                    )
                    wall_width = gr.Slider(
                        0.05,
                        0.6,
                        value=_DEFAULTS["wall_width"],
                        step=0.01,
                        label="Wall width",
                        elem_classes=["hfb-optics-dial-wrap"],
                    )
                    circulation = gr.Slider(
                        0.05,
                        1.0,
                        value=_DEFAULTS["circulation"],
                        step=0.05,
                        label="Circulation",
                        elem_classes=["hfb-optics-dial-wrap"],
                    )
                    defect_amplitude = gr.Slider(
                        0.2,
                        2.0,
                        value=_DEFAULTS["defect_amplitude"],
                        step=0.1,
                        label="Defect amplitude",
                        elem_classes=["hfb-optics-dial-wrap"],
                    )

                with gr.Row(elem_classes=["hfb-optics-tune-row"]):
                    sound_speed = gr.Slider(
                        0.5,
                        2.0,
                        value=_DEFAULTS["sound_speed"],
                        step=0.1,
                        label="Sound speed c₀",
                        elem_classes=["hfb-optics-dial-wrap"],
                    )
                    major_radius = gr.Slider(
                        0.4,
                        2.0,
                        value=_DEFAULTS["major_radius"],
                        step=0.05,
                        label="Hopf major radius",
                        elem_classes=["hfb-optics-dial-wrap"],
                    )
                    minor_radius = gr.Slider(
                        0.1,
                        0.8,
                        value=_DEFAULTS["minor_radius"],
                        step=0.02,
                        label="Hopf minor radius",
                        elem_classes=["hfb-optics-dial-wrap"],
                    )
                    hopf_index = gr.Slider(
                        1,
                        3,
                        value=_DEFAULTS["hopf_index"],
                        step=1,
                        label="Hopf index",
                        elem_classes=["hfb-optics-dial-wrap"],
                    )

                with gr.Row(elem_classes=["hfb-optics-tune-row"]):
                    defect_profile = gr.Dropdown(
                        choices=["toroidal_bubble_wall", "exponential_ring"],
                        value=_DEFAULTS["defect_profile"],
                        label="Defect profile",
                        elem_classes=["hfb-optics-dial-wrap"],
                    )
                    use_3d_torus = gr.Checkbox(
                        label="3D torus slice (hopf texture)",
                        value=_DEFAULTS["use_3d_torus"],
                        elem_classes=["hfb-optics-dial-wrap"],
                    )
                    include_3d = gr.Checkbox(
                        label="Include 3D surface plot",
                        value=False,
                        info=viz3d_info,
                        elem_classes=["hfb-optics-dial-wrap"],
                    )

                with gr.Row(elem_classes=["hfb-optics-tune-row"]):
                    vs = gr.Slider(0.05, 0.8, value=0.3, step=0.05, label="Alcubierre vs", elem_classes=["hfb-optics-dial-wrap"])
                    rs = gr.Slider(0.2, 2.0, value=1.0, step=0.1, label="Warp rs", elem_classes=["hfb-optics-dial-wrap"])
                    sigma = gr.Slider(0.05, 0.8, value=0.25, step=0.05, label="Warp σ", elem_classes=["hfb-optics-dial-wrap"])

            term_all_btns["clear"].click(
                _make_term_clear_click("clear"),
                inputs=[optics_terminal, term_ui_state],
                outputs=term_keypad_outputs,
            )
            for hold_key in TERM_DPAD_HOLD_KEYS:
                term_all_btns[hold_key].click(
                    _make_term_dpad_click(hold_key),
                    inputs=[optics_terminal, term_ui_state],
                    outputs=term_keypad_outputs,
                )
            term_all_btns[TERM_KEYPAD_HOME_KEY].click(
                _make_term_home_momentary(),
                inputs=[term_active_key, term_ui_state],
                outputs=term_keypad_outputs,
            )
            for index in range(1, TERM_KEYPAD_COUNT + 1):
                key_id = _term_key_id(index)
                if index == 1:
                    continue
                if index in TERM_KEYPAD_DEFINED:
                    action = TERM_KEYPAD_DEFINED[index]
                    term_all_btns[key_id].click(
                        _make_term_stream_click(
                            key_id,
                            TERM_KEYPAD_STREAMERS[action],
                            menu_action=action,
                        ),
                        inputs=[term_ui_state],
                        outputs=term_keypad_outputs,
                    )
                else:
                    term_all_btns[key_id].click(
                        _make_term_latch_click(key_id),
                        inputs=[optics_terminal, term_ui_state],
                        outputs=term_keypad_outputs,
                    )

            bubble_inputs = [
                bubble_radius,
                wall_width,
                defect_amplitude,
                circulation,
                sound_speed,
                defect_profile,
                use_3d_torus,
                hopf_index,
                major_radius,
                minor_radius,
            ]
            flux_btn = gr.Button("Run flux bubble", variant="primary")
            warp_btn = gr.Button("Run warp compare", variant="secondary")
            sweep_btn = gr.Button("Run stability sweep", variant="secondary")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    metrics_out = gr.Textbox(label="Stability / warp metrics", lines=14)
                with gr.Column(scale=2):
                    flux_figure = gr.Image(label="Flux bubble 4-panel", type="filepath", elem_classes=["hfb-figure-panel"])
            figure_3d = gr.Image(label="3D torus surface (optional)", type="filepath", elem_classes=["hfb-figure-panel"])
            warp_figure = gr.Image(label="Warp comparison", type="filepath", elem_classes=["hfb-figure-panel"])

            flux_btn.click(
                run_flux,
                inputs=[*bubble_inputs, include_3d],
                outputs=[metrics_out, flux_figure, figure_3d],
            )
            warp_btn.click(
                run_warp,
                inputs=[*bubble_inputs, vs, rs, sigma],
                outputs=[metrics_out, warp_figure],
            )
            sweep_btn.click(
                run_sweep,
                inputs=[defect_profile, use_3d_torus, hopf_index, major_radius, minor_radius],
                outputs=[metrics_out],
            )

        with gr.Column(visible=False, elem_classes=["hfb-gallery-page"]) as page_gallery:
            with gr.Row(elem_classes=["hfb-source-tabs-row"]):
                gr.HTML('<span class="hfb-nav-row-label">Source:</span>')
                gal_tab_demo_btn = gr.Button("Live Demo", elem_classes=["hfb-source-tab"], scale=0, variant="secondary")
                gal_tab_gallery_btn = gr.Button(
                    "Gallery",
                    elem_classes=["hfb-source-tab", "active"],
                    interactive=False,
                    scale=0,
                    variant="secondary",
                )
            gr.Markdown("## Gallery — reference outputs from `hfb-demo`")
            gr.HTML(_gallery_grid_html())
            gr.Markdown(
                f"[Flux bubble]({_GALLERY_FLUX_URL}) · "
                f"[3D surface]({_GALLERY_3D_URL}) · "
                f"[Warp compare]({_GALLERY_WARP_URL})"
            )

        caveats_outputs = [panel_caveats, tab_caveats_btn, caveats_open]
        nav_outputs = [
            page_demo,
            page_gallery,
            tab_demo_btn,
            tab_gallery_btn,
            panel_caveats,
            tab_caveats_btn,
            caveats_open,
            gal_tab_demo_btn,
            gal_tab_gallery_btn,
            current_page,
        ]
        tab_demo_btn.click(lambda: _nav_to_page("demo"), outputs=nav_outputs)
        tab_gallery_btn.click(lambda: _nav_to_page("gallery"), outputs=nav_outputs)
        gal_tab_demo_btn.click(lambda: _nav_to_page("demo"), outputs=nav_outputs)
        gal_tab_gallery_btn.click(lambda: _nav_to_page("gallery"), outputs=nav_outputs)
        tab_caveats_btn.click(_toggle_caveats, inputs=[caveats_open], outputs=caveats_outputs)
        caveats_minimize_btn.click(_minimize_caveats, outputs=caveats_outputs[:3])
        demo.load(_term_boot_home, outputs=term_keypad_outputs)

        gr.Markdown(
            f"MIT license · effective analog exploration only · "
            f"[hfb on GitHub]({GITHUB_URL}) · [HF Space]({HF_SPACE_URL})"
        )
    return demo


demo = build_app()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        demo.get_api_info()
        logger.info("Gradio API info check passed")
    except Exception:
        logger.exception("Gradio API info check failed")

    on_hf = bool(os.environ.get("SPACE_ID"))
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    launch_kwargs: dict = {
        "server_name": "0.0.0.0",
        "server_port": port,
        "show_error": True,
        "show_api": False,
        "inbrowser": False,
        "share": False if on_hf else True,
    }
    demo.queue(default_concurrency_limit=2).launch(**launch_kwargs)


if __name__ == "__main__":
    main()