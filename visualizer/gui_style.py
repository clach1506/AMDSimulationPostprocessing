"""Shared ttk theme for the whole GUI — colors, fonts, and widget styles.
Split out of app.py so the color palette has one home instead of being
duplicated wherever a panel needs to match it."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

BG = "#f4f5f7"
PANEL_BG = "#ffffff"
ACCENT = "#2f6feb"
TEXT = "#1f2328"
MUTED = "#57606a"


def apply_style(root: tk.Tk) -> None:
    """A light, flat, modern-ish theme built on ttk's "clam" base — no
    external theming dependency, just consistent spacing/colors/fonts."""
    root.configure(bg=BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    base_font = ("Helvetica Neue", 11)
    header_font = ("Helvetica Neue", 15, "bold")
    label_font = ("Helvetica Neue", 11, "bold")

    style.configure(".", background=BG, foreground=TEXT, font=base_font)
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL_BG)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Panel.TLabel", background=PANEL_BG, foreground=TEXT)
    style.configure("Muted.TLabel", background=BG, foreground=MUTED)
    style.configure("Header.TLabel", background=BG, foreground=TEXT, font=header_font)
    style.configure("TLabelframe", background=BG, borderwidth=0)
    style.configure("TLabelframe.Label", background=BG, foreground=TEXT, font=label_font)
    style.configure("TCheckbutton", background=BG, foreground=TEXT)
    style.configure("TButton", padding=(10, 6))
    style.configure("Accent.TButton", padding=(10, 6), foreground="white", background=ACCENT)
    style.map("Accent.TButton", background=[("active", ACCENT), ("pressed", ACCENT)])
    style.configure("TEntry", padding=4)
    style.configure("TCombobox", padding=4)
    style.configure("TScale", background=BG)
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(14, 8))
    style.configure("Horizontal.TSeparator", background=MUTED)
